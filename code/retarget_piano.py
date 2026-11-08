"""Explore a closed-form piano-to-flute edit of Magenta's style adapter.

Calibration uses frozen encoder forward passes. No gradients or optimizer.
This adapts established linear model-editing ideas, not a new editing method.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from edit_piano import EMBEDDING, TARGET
from compare_piano import load_cases, embed_case
from magenta_smoke import REPO, REVISION
from pilot import ROOT, audio_metrics, sha256

TEMPLATES = (
    "Solo {} playing a gentle melody.",
    "A recording of {} playing a slow classical piece.",
    "Solo {}, fast repeated notes and arpeggios.",
    "A jazz improvisation played on {}.",
    "Warm expressive melody played on {}.",
    "Soft {} music recorded in a quiet room.",
    "A lively folk tune performed on {} alone.",
    "A musician plays a beautiful {} solo.",
)
CONTROLS = ("acoustic guitar", "violin", "flute", "cello", "harp", "saxophone", "synthesizer", "drums")
SEEDS = (3, 17, 29)
CASES = {
    "grand": {"prompt": "A concert grand piano recital, flowing romantic phrases, solo performance.", "seed": 101, "style_mapper_seed": 101, "target": "piano"},
    "upright": {"prompt": "An old upright piano plays a jaunty ragtime tune in an empty room.", "seed": 103, "style_mapper_seed": 103, "target": "piano"},
    "guitar": {"prompt": "Intimate solo acoustic guitar, fingerpicked strings, a delicate melody.", "seed": 101, "style_mapper_seed": 101, "target": "acoustic guitar"},
    "violin": {"prompt": "Unaccompanied violin solo, a slow lyrical tune played with a bow.", "seed": 103, "style_mapper_seed": 103, "target": "violin"},
}


def retarget_delta(weight, source, destination, preserve):
    """Minimum-norm update fitting source->destination and preservation rows."""
    weight = np.asarray(weight, dtype=np.float64)
    source, destination, preserve = (np.asarray(x, dtype=np.float64) for x in (source, destination, preserve))
    if (weight.ndim != 2 or source.ndim != 2 or source.shape != destination.shape
            or preserve.ndim != 2 or source.shape[1] != weight.shape[0]
            or preserve.shape[1] != weight.shape[0] or not len(source)
            or not all(np.isfinite(x).all() for x in (weight, source, destination, preserve))):
        raise ValueError("Expected finite, compatible weight and calibration matrices")
    constraints = np.concatenate((source, preserve))
    desired_change = np.concatenate(((destination - source) @ weight,
                                     np.zeros((len(preserve), weight.shape[1]))))
    return np.linalg.pinv(constraints, rcond=1e-4) @ desired_change


def style_vector(tokens, codebook):
    from magenta_rt.config import MUSICCOCA, NUM_RESERVED_TOKENS

    tokens = np.asarray(tokens)
    if (tokens.shape != (MUSICCOCA.rvq_truncation_level,)
            or not np.issubdtype(tokens.dtype, np.integer)
            or np.any((tokens < 0) | (tokens >= MUSICCOCA.codebook_size))):
        raise ValueError("Unexpected style tokens")
    indices = tokens + NUM_RESERVED_TOKENS + 1 + np.arange(len(tokens)) * MUSICCOCA.per_rvq_vocab_size
    return codebook[indices].sum(axis=0, dtype=np.float64)


def adapter_layer(model, original):
    """Find the same Dense layer used by the pinned official weight loader."""
    import mlx.core as mx
    from magenta_rt.mlx.load_weights import _get_inner

    branch = model._sampler.depthformer.encoder.body.layers[0].layers[0].layers[1]
    layers = [_get_inner(layer) for layer in branch.layers]
    dense = [layer._linear for layer in layers if "Dense" in type(layer).__name__]
    if len(dense) != 1 or not bool(mx.array_equal(dense[0].weight, mx.array(original.T).astype(mx.bfloat16)).item()):
        raise RuntimeError("Runtime adapter does not match the original checkpoint")
    return dense[0]


def generate_case(model, case, directory, name, variant, patch_hash):
    import mlx.core as mx
    from magenta_rt.config import MUSICCOCA

    embedding = embed_case(model, case)
    np.random.seed(case["seed"])
    mx.random.seed(case["seed"])
    started = time.perf_counter()
    waveform, _ = model.generate(conditioning={MUSICCOCA.key: embedding}, frames=300)
    metrics = audio_metrics(waveform.samples, waveform.sample_rate)
    path = directory / f"{name}.wav"
    waveform.write(str(path))
    record = {**case, "variant": variant, "case": name, "frames": 300,
              "repo_id": REPO, "revision": REVISION, "patch_sha256": patch_hash,
              "temperature": model.temperature, "top_k": model.top_k, "cfg_scales": model.cfg_scales,
              "audio": metrics, "wall_seconds": time.perf_counter() - started,
              "wav_sha256": sha256(path)}
    path.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"variant": variant, "case": name, "seconds": record["wall_seconds"],
                      "rms_dbfs": metrics["rms_dbfs"]}), flush=True)
    mx.clear_cache()


def main():
    import mlx.core as mx
    from huggingface_hub import snapshot_download
    from magenta_rt import MagentaRT2Mlx, paths
    from magenta_rt.musiccoca import MusicCoCa
    from safetensors import safe_open
    from safetensors.numpy import save_file

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "piano_retarget_search")
    parser.add_argument("--calibration", type=Path, help="Reuse a recorded calibration manifest")
    parser.add_argument("--centroids", action="store_true", help="Fit instrument means instead of individual examples")
    parser.add_argument("--cases", type=Path, help="Text-only exploratory cases")
    args = parser.parse_args()
    cases = load_cases(args.cases) if args.cases else CASES
    if any("prompt" not in case for case in cases.values()):
        parser.error("This exploratory runner expects text cases")
    directory = args.output
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "runner.py").write_bytes(Path(__file__).read_bytes())
    (directory / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    snapshot = Path(snapshot_download(REPO, revision=REVISION, local_files_only=True,
                                     allow_patterns=["resources/*", "checkpoints/mrt2_base.safetensors"]))
    paths.set_magenta_home(snapshot)
    checkpoint = snapshot / "checkpoints" / "mrt2_base.safetensors"
    with safe_open(checkpoint, framework="numpy") as weights:
        original = weights.get_tensor(TARGET)
        codebook = weights.get_tensor(EMBEDDING)
    encoder = MusicCoCa()
    calibration, source, destination, preserve = [], [], [], []

    def encode(prompt, seed, group):
        tokens = encoder.tokenize(encoder.embed(prompt, use_mapper=True, seed=seed))
        calibration.append({"prompt": prompt, "mapper_seed": seed, "group": group, "tokens": tokens.tolist()})
        return style_vector(tokens, codebook)

    if args.calibration:
        previous = json.loads(args.calibration.read_text())
        if previous["repo_id"] != REPO or previous["revision"] != REVISION:
            raise ValueError("Calibration came from a different checkpoint")
        calibration = previous["calibration"]
        for group, values in (("source", source), ("destination", destination), ("preserve", preserve)):
            values.extend(style_vector(row["tokens"], codebook) for row in calibration if row["group"] == group)
    else:
        for seed in SEEDS:
            for template in TEMPLATES:
                source.append(encode(template.format("piano"), seed, "source"))
                destination.append(encode(template.format("flute"), seed, "destination"))
            for instrument in CONTROLS:
                for template in TEMPLATES[:3]:
                    preserve.append(encode(template.format(instrument), seed, "preserve"))
    source, destination, preserve = map(np.array, (source, destination, preserve))
    if args.centroids:
        source, destination = source.mean(axis=0, keepdims=True), destination.mean(axis=0, keepdims=True)
        # Group by the recorded prompt, not an implicit array ordering.
        preserved_means = []
        for instrument in CONTROLS:
            prompts = {template.format(instrument) for template in TEMPLATES[:3]}
            vectors = [style_vector(row["tokens"], codebook) for row in calibration
                       if row["group"] == "preserve" and row["prompt"] in prompts]
            if not vectors:
                raise ValueError(f"Missing preservation examples: {instrument}")
            preserved_means.append(np.mean(vectors, axis=0))
        preserve = np.array(preserved_means)
    delta = retarget_delta(original, source, destination, preserve)
    # Rotate the input basis: random control has the same rank and singular values.
    rotation, _ = np.linalg.qr(np.random.default_rng(20261108).normal(size=(len(original), len(original))))
    candidates = {"retarget": delta, "stronger": 1.5 * delta, "random": rotation @ delta}
    manifest = {"repo_id": REPO, "revision": REVISION, "target": TARGET,
                "calibration": calibration, "source_checkpoint": str(checkpoint),
                "formula": "delta = pinv([piano; preserve], rcond=1e-4) @ [(flute-piano) @ W; 0]",
                "fit_instrument_centroids": args.centroids,
                "reused_calibration_sha256": sha256(args.calibration) if args.calibration else None,
                "calibration_relative_residual": float(np.linalg.norm(source @ (original + delta) - destination @ original) / np.linalg.norm(destination @ original)),
                "preserve_relative_residual": float(np.linalg.norm(preserve @ delta) / np.linalg.norm(preserve @ original)),
                "candidates": {}, "claim": "Exploratory calibration-based model editing; no gradients. Not a novel method or evidence of full unlearning."}
    for name, change in candidates.items():
        edited = (original.astype(np.float64) + change).astype(np.float32)
        patch = directory / f"{name}.safetensors"
        save_file({TARGET: edited}, patch)
        manifest["candidates"][name] = {"patch_sha256": sha256(patch),
            "changed_parameters": int(np.count_nonzero(edited != original)),
            "relative_adapter_change": float(np.linalg.norm(edited - original) / np.linalg.norm(original))}
    (directory / "edit.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "calibration"}), flush=True)
    mx.set_cache_limit(1024**3)
    model = MagentaRT2Mlx(size="mrt2_base", bits=None, checkpoint=str(checkpoint), style_model=encoder)
    adapter = adapter_layer(model, original)
    for name in ("baseline", *candidates):
        if name == "baseline":
            weight, digest = original, None
        else:
            with safe_open(directory / f"{name}.safetensors", framework="numpy") as patch:
                weight = patch.get_tensor(TARGET)
            digest = manifest["candidates"][name]["patch_sha256"]
        adapter.weight = mx.array(weight.T).astype(mx.bfloat16)
        mx.eval(adapter.weight)
        output = directory / name
        output.mkdir()
        (output / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
        for case_name, case in cases.items():
            generate_case(model, case, output, case_name, name, digest)


if __name__ == "__main__":
    main()
