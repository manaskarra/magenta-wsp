"""Build a covariance-aware rank-one edit with explicit audio preservation.

All calibration encoders are frozen. The gate is solved in closed form; there
are no gradients, optimizer steps, or inference-time interventions.
"""

import json
from pathlib import Path

import numpy as np

from edit_piano import TARGET, EMBEDDING
from magenta_smoke import REPO, REVISION
from pilot import ROOT, sha256
from retarget_piano import style_vector, CONTROLS

TEMPLATES = (
    "A solo {} performance, unaccompanied, clear instrument tone.",
    "An intimate recording of {} playing a simple lullaby.",
    "A gentle waltz played on {} alone.",
    "Solo {} playing a blues melody with a swinging rhythm.",
    "A slow expressive {} solo with long pauses between phrases.",
    "Bright {} playing quick ascending and descending melodic runs.",
    "A classical solo performed on {} in a small recital room.",
    "A quiet {} melody, warm tone, no backing instruments.",
    "Fast lively {} music with a clear playful tune.",
    "Low notes and a dark melancholy melody on solo {}.",
    "High delicate notes played softly on {}.",
    "A musician practices a scale on {}, one instrument only.",
)
SEEDS = (5, 23, 41, 61)
AUDIO_PIANO = (
    "piano_validation/baseline/concert_grand_11.wav",
    "piano_validation/baseline/piano_jazz_37.wav",
    "magenta_base_smoke/audio.wav",
)
AUDIO_PRESERVE = (
    "piano_validation/baseline/guitar_control_11.wav",
    "piano_validation/baseline/violin_control_11.wav",
    "piano_validation/baseline/flute_control_37.wav",
    "piano_centroid_search/baseline/guitar.wav",
    "piano_centroid_search/baseline/violin.wav",
    "piano_centroid_search/baseline/flute_prompt.wav",
)


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


def main():
    from magenta_rt import paths
    from magenta_rt.audio import Waveform
    from magenta_rt.musiccoca import MusicCoCa
    from safetensors import safe_open
    from safetensors.numpy import save_file

    output = ROOT / "results" / "piano_protected_search"
    output.mkdir(parents=True, exist_ok=False)
    (output / "runner.py").write_bytes(Path(__file__).read_bytes())
    old = json.loads((ROOT / "results/piano_centroid_search/edit.json").read_text())
    checkpoint = Path(old["source_checkpoint"])
    paths.set_magenta_home(checkpoint.parent.parent)
    with safe_open(checkpoint, framework="numpy") as weights:
        original, codebook = weights.get_tensor(TARGET), weights.get_tensor(EMBEDDING)
    encoder = MusicCoCa()
    calibration = []

    def encode(prompt, seed, group):
        tokens = encoder.tokenize(encoder.embed(prompt, use_mapper=True, seed=seed))
        calibration.append({"prompt": prompt, "mapper_seed": seed, "group": group, "tokens": tokens.tolist()})
        return style_vector(tokens, codebook)

    piano, flute, controls = [], [], []
    for instrument in ("piano", "flute", *[c for c in CONTROLS if c != "flute"]):
        vectors = [encode(template.format(instrument), seed, instrument) for seed in SEEDS for template in TEMPLATES]
        if instrument == "piano":
            piano = vectors
        elif instrument == "flute":
            flute = vectors
            controls.append(vectors)
        else:
            controls.append(vectors)

    def audio_vectors(names, group):
        vectors = []
        for name in names:
            path = ROOT / "results" / name
            tokens = encoder.tokenize(encoder.embed(Waveform.from_file(str(path)), use_mapper=False))
            calibration.append({"audio_reference": str(path), "audio_sha256": sha256(path), "group": group, "tokens": tokens.tolist()})
            vectors.append(style_vector(tokens, codebook))
        return np.array(vectors)

    piano, flute, controls = map(np.array, (piano, flute, controls))
    audio_piano = audio_vectors(AUDIO_PIANO, "audio_piano")
    audio_controls = audio_vectors(AUDIO_PRESERVE, "audio_preserve")
    gate = protected_gate(piano, controls, audio_piano, audio_controls)
    # Keep the already demonstrated output direction while improving where it acts.
    old_source = np.array([style_vector(c["tokens"], codebook) for c in old["calibration"] if c["group"] == "source"])
    old_destination = np.array([style_vector(c["tokens"], codebook) for c in old["calibration"] if c["group"] == "destination"])
    direction = (old_destination.mean(0) - old_source.mean(0)) @ original
    delta = np.outer(gate, direction)
    candidates = {"protected": delta, "stronger": 1.4 * delta}
    manifest = {"repo_id": REPO, "revision": REVISION, "source_checkpoint": str(checkpoint),
        "target": TARGET, "calibration": calibration, "ridge": 1.0,
        "output_direction_source_sha256": sha256(ROOT / "results/piano_centroid_search/edit.json"),
        "formula": "delta = g outer ((old_flute_mean-old_piano_mean) @ W), g = Q^-1 A.T (A Q^-1 A.T)^+ b",
        "audio_preservation_max_gate": float(np.max(np.abs(audio_controls @ gate))),
        "text_piano_gate_mean_std": [float(np.mean(piano @ gate)), float(np.std(piano @ gate))],
        "text_control_gate_rms": float(np.sqrt(np.mean((controls @ gate) ** 2))),
        "candidates": {}}
    for name, change in candidates.items():
        weight = (original + change).astype(np.float32)
        path = output / f"{name}.safetensors"
        save_file({TARGET: weight}, path)
        manifest["candidates"][name] = {"patch_sha256": sha256(path),
            "changed_parameters": int(np.count_nonzero(weight != original)),
            "relative_adapter_change": float(np.linalg.norm(weight - original) / np.linalg.norm(original))}
    np.savez(output / "calibration_vectors.npz", piano=piano, flute=flute, controls=controls,
             audio_piano=audio_piano, audio_controls=audio_controls, gate=gate, direction=direction)
    (output / "edit.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "calibration"}), flush=True)


if __name__ == "__main__":
    main()
