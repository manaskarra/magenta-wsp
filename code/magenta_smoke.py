"""Measure unquantized Magenta before attempting piano erasure."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np

from pilot import ROOT, audio_metrics, sha256

REPO = "google/magenta-realtime-2"
REVISION = "010aa0dcb0dfd27b24f0ad07b4dad63e8f9521cc"
PROMPT = "Solo acoustic grand piano, a clear gentle piano melody, no other instruments."


def main():
    import mlx.core as mx
    from huggingface_hub import snapshot_download
    from magenta_rt import MagentaRT2Mlx, paths
    from magenta_rt.config import MUSICCOCA

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=("mrt2_small", "mrt2_base"), default="mrt2_small")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "magenta_smoke")
    args = parser.parse_args()
    directory = args.output
    if directory.exists():
        raise FileExistsError(f"Preserving existing evidence: {directory}")
    checkpoint = snapshot_download(
        REPO, revision=REVISION,
        allow_patterns=["resources/*", f"checkpoints/{args.size}.safetensors", "README.md", "LICENSE"],
        max_workers=4,
    )
    paths.set_magenta_home(checkpoint)
    mx.set_cache_limit(1024**3)
    started = time.perf_counter()
    model = MagentaRT2Mlx(size=args.size, bits=None)
    load_seconds = time.perf_counter() - started
    print(f"Magenta loaded and warmed up in {load_seconds:.1f}s", flush=True)
    embedding = model.embed_style(PROMPT, use_mapper=True, seed=7)
    np.random.seed(7)
    mx.random.seed(7)
    started = time.perf_counter()
    waveform, _ = model.generate(conditioning={MUSICCOCA.key: embedding}, frames=300)
    wall_seconds = time.perf_counter() - started
    metrics = audio_metrics(waveform.samples, waveform.sample_rate)
    directory.mkdir(parents=True, exist_ok=False)
    wav = directory / "audio.wav"
    waveform.write(str(wav))
    record = {
        "repo_id": REPO, "revision": REVISION, "size": args.size,
        "quantization_bits": None, "prompt": PROMPT, "seed": 7,
        "frames": 300, "requested_seconds": 12.0,
        "temperature": model.temperature, "top_k": model.top_k,
        "cfg_scales": model.cfg_scales, "style_mapper": True,
        "style_embedding": embedding.tolist(),
        "runner_sha256": sha256(Path(__file__)),
        "packages": {p: importlib.metadata.version(p) for p in ("magenta-rt", "mlx", "numpy", "ai-edge-litert")},
        "audio": metrics, "load_and_warmup_seconds": load_seconds,
        "wall_seconds": wall_seconds, "peak_mlx_memory_gb": mx.get_peak_memory() / 1e9,
        "wav_sha256": sha256(wav),
        "assessment": "Unedited feasibility sample; instrument identity and quality need listening.",
    }
    (directory / "runner.py").write_bytes(Path(__file__).read_bytes())
    (directory / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k != "style_embedding"}), flush=True)


if __name__ == "__main__":
    main()
