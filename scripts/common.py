"""File integrity and audio checks shared by the reproduction commands."""

import hashlib

import numpy as np


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def audio_metrics(audio, sample_rate):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 2 or audio.shape[1] != 2 or not audio.shape[0]:
        raise ValueError("Expected nonempty stereo audio, shaped [samples, 2]")
    if sample_rate <= 0 or not np.isfinite(audio).all():
        raise ValueError("Invalid sample rate or non-finite audio")
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    if rms == 0:
        raise ValueError("Generated audio is silent")
    return {
        "sample_rate": sample_rate,
        "samples": len(audio),
        "duration_seconds": len(audio) / sample_rate,
        "rms_dbfs": 20 * np.log10(rms),
        "peak": float(np.max(np.abs(audio))),
        "clipped_fraction": float(np.mean(np.abs(audio) >= 0.999)),
    }
