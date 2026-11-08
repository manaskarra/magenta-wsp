"""First hypothesis: erase one piano-associated direction in the style adapter.

Uses frozen text-encoder forward passes and checkpoint tensors, with no gradients.
This is a standard projection baseline, not a claim of novel or complete unlearning.
"""

import json
from pathlib import Path

import numpy as np

from pilot import ROOT, sha256

PREFIX = "params/depthformer/encoder/body/layers_1/branched_mulan_embedder/mulan_embedder/"
TARGET = PREFIX + "depth_input_adapter/kernel"
EMBEDDING = PREFIX + "mulan_dequantizer/embedding"
PAIRS = (
    ("Solo {} playing a gentle melody.", "piano", "acoustic guitar"),
    ("A recording of {} playing a slow classical piece.", "grand piano", "violin"),
    ("Solo {}, fast repeated notes and arpeggios.", "piano", "harp"),
    ("A jazz improvisation played on {}.", "piano", "saxophone"),
    ("Warm expressive melody played on {}.", "grand piano", "flute"),
    ("Soft {} music recorded in a quiet room.", "piano", "cello"),
)


def remove_direction(weight, direction):
    """For row inputs x @ W, remove the component parallel to direction."""
    weight = np.asarray(weight)
    direction = np.asarray(direction, dtype=np.float64)
    if weight.ndim != 2 or direction.shape != (weight.shape[0],):
        raise ValueError("Direction must match the adapter's input dimension")
    norm = np.linalg.norm(direction)
    if not np.isfinite(weight).all() or not np.isfinite(norm) or norm == 0:
        raise ValueError("Finite weights and a nonzero finite direction are required")
    unit = direction / norm
    return (weight.astype(np.float64) - np.outer(unit, unit @ weight)).astype(weight.dtype)


def main():
    from huggingface_hub import snapshot_download
    from magenta_rt import paths
    from magenta_rt.config import MUSICCOCA, NUM_RESERVED_TOKENS
    from magenta_rt.musiccoca import MusicCoCa
    from safetensors import safe_open
    from safetensors.numpy import save_file
    from magenta_smoke import REPO, REVISION

    directory = ROOT / "results" / "piano_projection"
    if directory.exists():
        raise FileExistsError(f"Preserving previous edit: {directory}")
    snapshot = Path(snapshot_download(REPO, revision=REVISION, local_files_only=True,
                                     allow_patterns=["resources/*", "checkpoints/mrt2_base.safetensors"]))
    paths.set_magenta_home(snapshot)
    source = snapshot / "checkpoints" / "mrt2_base.safetensors"
    with safe_open(source, framework="numpy") as checkpoint:
        codebook = checkpoint.get_tensor(EMBEDDING)
        original = checkpoint.get_tensor(TARGET)
    encoder = MusicCoCa()
    calibration = []
    differences = []
    for template, positive, negative in PAIRS:
        vectors = []
        pair = []
        for instrument in (positive, negative):
            prompt = template.format(instrument)
            embedding = encoder.embed(prompt, use_mapper=True, seed=7)
            tokens = encoder.tokenize(embedding)
            if tokens.shape != (MUSICCOCA.rvq_truncation_level,) or np.any((tokens < 0) | (tokens >= MUSICCOCA.codebook_size)):
                raise ValueError("Unexpected MusicCoCa tokens")
            # Match _build_conditioning and the checkpoint's branched embedder.
            indices = tokens + NUM_RESERVED_TOKENS + 1 + np.arange(len(tokens)) * MUSICCOCA.per_rvq_vocab_size
            vectors.append(codebook[indices].sum(axis=0, dtype=np.float64))
            pair.append({"prompt": prompt, "tokens": tokens.tolist()})
        differences.append(vectors[0] - vectors[1])
        calibration.append(pair)
    direction = np.mean(differences, axis=0)
    edited = remove_direction(original, direction)
    with safe_open(source, framework="numpy") as checkpoint:
        metadata = checkpoint.metadata()
        weights = {key: checkpoint.get_tensor(key) for key in checkpoint.keys()}
    total_parameters = sum(value.size for value in weights.values())
    weights[TARGET] = edited
    directory.mkdir(parents=True, exist_ok=False)
    destination = directory / "mrt2_base.safetensors"
    save_file(weights, destination, metadata=metadata)
    del weights
    # Verify serialization and that every other tensor is exactly unchanged.
    changed = []
    with safe_open(source, framework="numpy") as before, safe_open(destination, framework="numpy") as after:
        if before.keys() != after.keys():
            raise RuntimeError("Checkpoint keys changed")
        for key in before.keys():
            if not np.array_equal(before.get_tensor(key), after.get_tensor(key)):
                changed.append(key)
    if changed != [TARGET]:
        raise RuntimeError(f"Unexpected modified tensors: {changed}")
    record = {
        "source_repo": REPO, "source_revision": REVISION, "checkpoint": str(destination),
        "checkpoint_sha256": sha256(destination), "changed_tensors": changed,
        "changed_parameters": int(np.count_nonzero(original != edited)),
        "total_parameters": total_parameters,
        "relative_adapter_change": float(np.linalg.norm(edited - original) / np.linalg.norm(original)),
        "formula": "W_edited = W - u (u^T W); u = normalized mean of six paired style-codebook differences",
        "calibration": calibration, "runner_sha256": sha256(Path(__file__)),
        "claim": "One style-adapter projection; not verified piano erasure or novel unlearning.",
    }
    np.save(directory / "direction.npy", direction)
    (directory / "runner.py").write_bytes(Path(__file__).read_bytes())
    (directory / "edit.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k != "calibration"}), flush=True)


if __name__ == "__main__":
    main()
