"""Fresh-process baseline/edited generations with identical evaluation settings."""

import argparse
import json
from pathlib import Path
import re
import time

import numpy as np

from magenta_smoke import REPO, REVISION, PROMPT
from pilot import ROOT, audio_metrics, sha256

CASES = {
    "piano_demo": (PROMPT, 7),
    "piano_heldout": ("Bright solo upright piano playing staccato jazz chords.", 19),
    "guitar_control": ("Solo acoustic guitar, a clear gentle guitar melody, no other instruments.", 7),
}


def load_cases(path):
    cases = json.loads(path.read_text())
    if not isinstance(cases, dict) or not cases:
        raise ValueError("Expected a nonempty object of named cases")
    for name, case in cases.items():
        if not re.fullmatch(r"[a-z0-9_]+", name):
            raise ValueError(f"Unsafe case name: {name!r}")
        text_fields = {"prompt", "seed", "style_mapper_seed", "target"}
        audio_fields = {"audio_reference", "audio_sha256", "seed", "target"}
        if not isinstance(case, dict) or set(case) not in (text_fields, audio_fields):
            raise ValueError(f"Unexpected case fields: {name}")
        string_fields = ("prompt", "target") if "prompt" in case else ("audio_reference", "audio_sha256", "target")
        if any(not isinstance(case[k], str) or not case[k].strip() for k in string_fields):
            raise ValueError(f"Missing prompt or target: {name}")
        seed_fields = ("seed", "style_mapper_seed") if "prompt" in case else ("seed",)
        if any(type(case[k]) is not int or not 0 <= case[k] < 2**32 for k in seed_fields):
            raise ValueError(f"Invalid random seed: {name}")
        if "audio_reference" in case and (not Path(case["audio_reference"]).is_absolute()
                or not re.fullmatch(r"[0-9a-f]{64}", case["audio_sha256"])):
            raise ValueError(f"Audio cases require an absolute path and SHA-256: {name}")
    return cases


def embed_case(model, case):
    """Use exactly the declared conditioning source; evaluation labels stay out."""
    if "prompt" in case:
        return model.embed_style(case["prompt"], use_mapper=True, seed=case["style_mapper_seed"])
    reference = Path(case["audio_reference"])
    if sha256(reference) != case["audio_sha256"]:
        raise ValueError(f"Audio reference hash mismatch: {reference}")
    from magenta_rt.audio import Waveform
    return model.embed_style(Waveform.from_file(str(reference)), use_mapper=False)


def main():
    import mlx.core as mx
    from huggingface_hub import snapshot_download
    from magenta_rt import MagentaRT2Mlx, paths
    from magenta_rt.config import MUSICCOCA

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edited", action="store_true")
    parser.add_argument("--variant", choices=("baseline", "old", "edited"), help="Label for three-way comparisons")
    parser.add_argument("--edit-record", type=Path, help="Edited checkpoint metadata; requires --edited")
    parser.add_argument("--cases", type=Path, help="Frozen JSON evaluation cases")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "piano_comparison")
    parser.add_argument("--seconds", type=int, choices=(12, 30), default=12)
    args = parser.parse_args()
    if args.edit_record and not args.edited:
        parser.error("--edit-record requires --edited")
    variant = args.variant or ("edited" if args.edited else "baseline")
    if (variant == "baseline") == args.edited:
        parser.error("baseline requires original weights; old/edited require --edited")
    cases = load_cases(args.cases) if args.cases else {
        name: {"prompt": prompt, "seed": seed, "style_mapper_seed": 7,
               "target": "acoustic guitar" if name == "guitar_control" else "piano"}
        for name, (prompt, seed) in CASES.items()
    }
    cases_hash = sha256(args.cases) if args.cases else None
    directory = args.output / variant
    if directory.exists():
        raise FileExistsError(f"Preserving previous evaluation: {directory}")
    snapshot = Path(snapshot_download(REPO, revision=REVISION, local_files_only=True,
                                     allow_patterns=["resources/*", "checkpoints/mrt2_base.safetensors"]))
    paths.set_magenta_home(snapshot)
    checkpoint = snapshot / "checkpoints" / "mrt2_base.safetensors"
    edit_record = None
    if args.edited:
        edit_record = json.loads((args.edit_record or ROOT / "results" / "piano_projection" / "edit.json").read_text())
        checkpoint = Path(edit_record["checkpoint"])
        if sha256(checkpoint) != edit_record["checkpoint_sha256"]:
            raise ValueError("Edited checkpoint does not match its recorded hash")
    mx.set_cache_limit(1024**3)
    model = MagentaRT2Mlx(size="mrt2_base", bits=None, checkpoint=str(checkpoint))
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "runner.py").write_bytes(Path(__file__).read_bytes())
    (directory / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    for name, case in cases.items():
        seed = case["seed"]
        print(f"Generating {variant}/{name}", flush=True)
        embedding = embed_case(model, case)
        np.random.seed(seed)
        mx.random.seed(seed)
        started = time.perf_counter()
        frames = args.seconds * 25
        wav, _ = model.generate(conditioning={MUSICCOCA.key: embedding}, frames=frames)
        seconds = time.perf_counter() - started
        metrics = audio_metrics(wav.samples, wav.sample_rate)
        audio_path = directory / f"{name}.wav"
        wav.write(str(audio_path))
        audio_hash = sha256(audio_path)
        if not args.cases and args.seconds == 12 and variant == "baseline" and name == "piano_demo":
            previous = json.loads((ROOT / "results" / "magenta_base_smoke" / "result.json").read_text())
            if audio_hash != previous["wav_sha256"]:
                raise RuntimeError("Baseline repeat differed from the original smoke sample; investigate before comparing edits")
        record = {
            "variant": variant, "case": name, **case,
            "cases_sha256": cases_hash, "frames": frames, "size": "mrt2_base",
            "temperature": model.temperature, "top_k": model.top_k, "cfg_scales": model.cfg_scales,
            "repo_id": REPO, "revision": REVISION, "checkpoint": str(checkpoint),
            "edit": edit_record, "runner_sha256": sha256(Path(__file__)),
            "audio": metrics, "wall_seconds": seconds, "wav_sha256": audio_hash,
        }
        (directory / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps({"case": name, "wall_seconds": seconds, **metrics}), flush=True)
        mx.clear_cache()


if __name__ == "__main__":
    main()
