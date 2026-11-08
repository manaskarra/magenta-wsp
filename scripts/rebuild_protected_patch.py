"""Rebuild the selected adapter from recorded frozen-encoder tokens, without training."""

import argparse
import json
from pathlib import Path

import numpy as np

from edit import EMBEDDING, TARGET, CONTROLS, protected_gate, style_vector
from materialize_patch import source_checkpoint
from common import sha256


def rebuild_weight(original, codebook, calibration, old_calibration, ridge):
    def vectors(rows, group):
        return np.array([style_vector(row["tokens"], codebook) for row in rows if row["group"] == group])

    controls = np.array([vectors(calibration, instrument)
        for instrument in ("flute", *[c for c in CONTROLS if c != "flute"])])
    gate = protected_gate(vectors(calibration, "piano"), controls,
        vectors(calibration, "audio_piano"), vectors(calibration, "audio_preserve"), ridge=ridge)
    direction = (vectors(old_calibration, "destination").mean(0)
                 - vectors(old_calibration, "source").mean(0)) @ original
    return (original + 1.4 * np.outer(gate, direction)).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--direction-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserving existing patch")
    manifest = json.loads(args.manifest.read_text())
    old = json.loads(args.direction_manifest.read_text())
    if sha256(args.direction_manifest) != manifest["output_direction_source_sha256"]:
        raise ValueError("Output direction calibration changed")
    source = source_checkpoint(manifest, args.base_checkpoint)
    from safetensors import safe_open
    from safetensors.numpy import save_file

    with safe_open(source, framework="numpy") as weights:
        original, codebook = weights.get_tensor(TARGET), weights.get_tensor(EMBEDDING)
    weight = rebuild_weight(original, codebook, manifest["calibration"], old["calibration"], manifest["ridge"])
    if weight.shape != original.shape or not np.isfinite(weight).all():
        raise ValueError("Invalid rebuilt weight")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file({TARGET: weight}, args.output)
    if sha256(args.output) != manifest["candidates"]["stronger"]["patch_sha256"]:
        raise RuntimeError("Rebuilt patch differs from the recorded selection; do not use it")
    print(json.dumps({"patch": str(args.output), "sha256": sha256(args.output), "exact_match": True}))


if __name__ == "__main__":
    main()
