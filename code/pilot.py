"""Reproduce MiniMax's vocal behavior without changing any model weights."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
STYLES = {
    "ambient": (
        "ambient, 60 BPM, D minor, calm and spacious",
        "Slow evolving synthesizer pads, soft electric piano and a gentle bass pulse.",
    ),
    "acoustic": (
        "acoustic pop, 96 BPM, C major, warm and intimate",
        "Fingerpicked acoustic guitar, soft piano, upright bass and brushed drums.",
    ),
}
LYRICS = (
    "[Verse]\nMorning light across the room\nSilver clouds are drifting through\n"
    "Every quiet road runs free\nCarry this small song with me\n"
    "[Chorus]\nLet the daylight find its way\nThrough the windows of the day\n"
    "Every note a place to start\nEvery road an open heart"
)


def cases():
    """Paired seeds and arrangements; only the vocal request changes."""
    result = {}
    for style, (metadata, arrangement) in STYLES.items():
        for condition in ("plain", "explicit", "singing_control"):
            if condition == "singing_control":
                vocal = "An intimate female lead singer clearly sings the supplied lyrics."
                text = f"Global Metadata: {metadata}.\nVocal Details: {vocal}"
                lyrics = LYRICS
            else:
                text = f"Global Metadata: Instrumental {metadata}."
                if condition == "explicit":
                    text += (
                        "\nVocal Details: Instrumental only. No vocals, no singing, "
                        "no humming, no chanting, no spoken words, no choir."
                    )
                lyrics = "[Instrumental]"
            result[f"{style}_{condition}"] = {
                "text": text + f"\nArrangement: {arrangement}",
                "lyrics": lyrics,
                "duration": 40.0,
                "steps": 30,
                "seed": 7,
            }
    result["smoke"] = {**result["acoustic_singing_control"], "duration": 8.0}
    return result


def audio_metrics(audio, sample_rate):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 2 or audio.shape[1] != 2 or not audio.shape[0]:
        raise ValueError("Expected nonempty stereo audio, shaped [samples, 2]")
    if sample_rate <= 0 or not np.isfinite(audio).all():
        raise ValueError("Invalid sample rate or non-finite audio")
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    if rms == 0:
        raise ValueError("Silent output is not successful instrumental generation")
    return {
        "sample_rate": sample_rate,
        "samples": len(audio),
        "duration_seconds": len(audio) / sample_rate,
        "rms_dbfs": 20 * np.log10(rms),
        "peak": float(np.max(np.abs(audio))),
        "clipped_fraction": float(np.mean(np.abs(audio) >= 0.999)),
    }


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def completed_case(directory, request):
    """Resume only if the exact request and its audio artifact still match."""
    metadata_path = directory / "result.json"
    if not metadata_path.exists():
        if directory.exists():
            raise RuntimeError(f"Incomplete run at {directory}; use a fresh --output")
        return False
    saved = json.loads(metadata_path.read_text())
    wav = directory / "audio.wav"
    if saved["request"] != request or not wav.exists() or sha256(wav) != saved["wav_sha256"]:
        raise RuntimeError(f"Existing run differs or is damaged: {directory}")
    return True


def main():
    choices = cases()
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--case", choices=choices, default="smoke")
    selection.add_argument("--pilot", action="store_true", help="Run all six 40-second cases")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "baseline")
    parser.add_argument("--list", action="store_true", help="Print frozen prompts without loading a model")
    args = parser.parse_args()
    selected = {k: v for k, v in choices.items() if k != "smoke"} if args.pilot else {args.case: choices[args.case]}
    if args.list:
        print(json.dumps(selected, indent=2))
        return

    import mlx.core as mx
    from mlx_audio.audio_io import write
    from mlx_audio.music import load
    from mlx_audio.music.models.minimax_music3.config import (
        AR_CFG_SCALE, AR_CFG_TOP_K, AR_SAMPLING_TOP_K, DIT_CFG_SCALE,
    )

    checkpoint = json.loads((ROOT / "checkpoint.json").read_text())
    model_path = Path(checkpoint["path"])
    # The upstream runtime has a synthetic tokenizer fallback for tiny tests.
    # Never let a missing real tokenizer silently enter our experiment.
    if not (model_path / "tokenizer" / "tokenizer.json").is_file():
        raise FileNotFoundError("The real checkpoint tokenizer is missing")
    config = json.loads((model_path / "config.json").read_text())
    if "quantization" in config or "quantization_config" in config:
        raise ValueError("This baseline requires the unquantized checkpoint")
    provenance = {
        "checkpoint": checkpoint,
        "runner_sha256": sha256(Path(__file__)),
        "packages": {p: importlib.metadata.version(p) for p in ("mlx", "mlx-audio", "numpy", "transformers")},
        "sampling": {
            "ar_cfg_scale": AR_CFG_SCALE, "ar_cfg_top_k": AR_CFG_TOP_K,
            "ar_sampling_top_k": AR_SAMPLING_TOP_K, "dit_cfg_scale": DIT_CFG_SCALE,
        },
    }
    pending = []
    for name, parameters in selected.items():
        request = {**provenance, "parameters": parameters}
        directory = args.output / name
        if completed_case(directory, request):
            print(f"Verified existing result: {name}", flush=True)
        else:
            pending.append((name, parameters, directory, request))
    if not pending:
        return
    mx.set_cache_limit(2 * 1024**3)
    mx.set_memory_limit(48 * 1024**3)
    print("Loading pinned unquantized checkpoint", flush=True)
    started = time.perf_counter()
    model = load(model_path, strict=True)
    print(f"Loaded in {time.perf_counter() - started:.1f}s; active {mx.get_active_memory()/1e9:.2f} GB", flush=True)
    for name, parameters, directory, request in pending:
        print(f"Generating {name}: requested {parameters['duration']}s", flush=True)
        started = time.perf_counter()
        result = next(model.generate(**parameters))
        audio = np.asarray(result.audio)
        metrics = audio_metrics(audio, result.sample_rate)
        directory.mkdir(parents=True, exist_ok=False)
        wav = directory / "audio.wav"
        write(wav, audio, result.sample_rate)
        record = {
            "request": request, "audio": metrics, "wav_sha256": sha256(wav),
            "wall_seconds": time.perf_counter() - started,
            "peak_mlx_memory_gb": mx.get_peak_memory() / 1e9,
            "vocal_assessment": "not assessed; numeric audio checks cannot establish absence of singing",
        }
        (directory / "result.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps({"case": name, **metrics, "wall_seconds": record["wall_seconds"]}), flush=True)
        del result, audio
        mx.clear_cache()


if __name__ == "__main__":
    main()
