"""The published style-adapter edit, calculated from frozen encoder tokens."""

import numpy as np

REPO = "google/magenta-realtime-2"
REVISION = "010aa0dcb0dfd27b24f0ad07b4dad63e8f9521cc"
PREFIX = "params/depthformer/encoder/body/layers_1/branched_mulan_embedder/mulan_embedder/"
TARGET = PREFIX + "depth_input_adapter/kernel"
EMBEDDING = PREFIX + "mulan_dequantizer/embedding"
CONTROLS = ("acoustic guitar", "violin", "flute", "cello", "harp", "saxophone", "synthesizer", "drums")


def style_vector(tokens, codebook):
    from magenta_rt.config import MUSICCOCA, NUM_RESERVED_TOKENS

    tokens = np.asarray(tokens)
    if (tokens.shape != (MUSICCOCA.rvq_truncation_level,)
            or not np.issubdtype(tokens.dtype, np.integer)
            or np.any((tokens < 0) | (tokens >= MUSICCOCA.codebook_size))):
        raise ValueError("Unexpected style tokens")
    indices = tokens + NUM_RESERVED_TOKENS + 1 + np.arange(len(tokens)) * MUSICCOCA.per_rvq_vocab_size
    return codebook[indices].sum(axis=0, dtype=np.float64)


def protected_gate(piano, controls, audio_piano, audio_controls, ridge=1.0):
    """Minimize gate variation while exactly protecting declared mean/audio rows.

    A g = b fixes piano means to 1 and preservation inputs to 0.
    g = Q^-1 A.T (A Q^-1 A.T)^+ b, with a positive ridge on Q.
    """
    piano, controls, audio_piano, audio_controls = [np.asarray(x, dtype=np.float64)
        for x in (piano, controls, audio_piano, audio_controls)]
    if (piano.ndim != 2 or controls.ndim != 3 or audio_piano.ndim != 2 or audio_controls.ndim != 2
            or any(not x.size or not np.isfinite(x).all() for x in (piano, controls, audio_piano, audio_controls))
            or any(x.shape[-1] != piano.shape[1] for x in (controls, audio_piano, audio_controls))
            or not np.isfinite(ridge) or ridge <= 0):
        raise ValueError("Need finite, compatible, nonempty calibration arrays and a positive ridge")
    p_mean, a_mean = piano.mean(0), audio_piano.mean(0)
    centered = piano - p_mean
    flattened = controls.reshape(-1, piano.shape[1])
    audio_centered = audio_piano - a_mean
    covariance = (centered.T @ centered / len(centered)
                  + flattened.T @ flattened / len(flattened)
                  + audio_centered.T @ audio_centered / len(audio_centered))
    scale = np.trace(covariance) / piano.shape[1]
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Calibration has no usable covariance")
    covariance += ridge * scale * np.eye(piano.shape[1])
    constraints = np.vstack((p_mean, a_mean, controls.mean(1), audio_controls))
    desired = np.zeros(len(constraints)); desired[:2] = 1
    whitened = np.linalg.solve(covariance, constraints.T)
    gate = whitened @ np.linalg.pinv(constraints @ whitened, rcond=1e-10) @ desired
    if not np.isfinite(gate).all() or np.max(np.abs(constraints @ gate - desired)) > 1e-6:
        raise ValueError("Calibration constraints are inconsistent or ill-conditioned")
    return gate
