"""Save and audit a full Magenta checkpoint containing one explored adapter patch."""

import argparse
import json
from pathlib import Path
import re

import numpy as np

from edit_piano import TARGET
from pilot import ROOT, sha256


def source_checkpoint(manifest, override=None):
    source = Path(override or manifest["source_checkpoint"])
    expected = manifest.get("source_checkpoint_sha256")
    if override is not None and not expected:
        raise ValueError("A portable base checkpoint requires its pinned SHA-256 in the manifest")
    if expected and sha256(source) != expected:
        raise ValueError("Base checkpoint hash mismatch")
    return source


def main():
    from safetensors import safe_open
    from safetensors.numpy import save_file

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search", type=Path, default=ROOT / "results" / "piano_retarget_search")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, help="Relocated original checkpoint, verified against the manifest hash")
    args = parser.parse_args()
    args.search = args.search.resolve()
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError(f"Preserving previous checkpoint: {args.output}")
    manifest = json.loads((args.search / "edit.json").read_text())
    if not re.fullmatch(r"[a-z0-9_]+", args.candidate) or args.candidate not in manifest["candidates"]:
        raise ValueError("Candidate must name a patch in the search manifest")
    source = source_checkpoint(manifest, args.base_checkpoint)
    patch_path = args.search / f"{args.candidate}.safetensors"
    if sha256(patch_path) != manifest["candidates"][args.candidate]["patch_sha256"]:
        raise ValueError("Patch hash mismatch")
    with safe_open(patch_path, framework="numpy") as patch, safe_open(source, framework="numpy") as checkpoint:
        if patch.keys() != [TARGET]:
            raise ValueError("Expected exactly one adapter tensor")
        edited, original = patch.get_tensor(TARGET), checkpoint.get_tensor(TARGET)
        if edited.shape != original.shape or edited.dtype != original.dtype or not np.isfinite(edited).all():
            raise ValueError("Invalid edited tensor")
        weights = {key: checkpoint.get_tensor(key) for key in checkpoint.keys()}
        metadata = checkpoint.metadata()
    weights[TARGET] = edited
    total = sum(value.size for value in weights.values())
    args.output.mkdir(parents=True, exist_ok=False)
    destination = args.output / "mrt2_base.safetensors"
    save_file(weights, destination, metadata=metadata)
    del weights
    changed = []
    with safe_open(source, framework="numpy") as before, safe_open(destination, framework="numpy") as after:
        if before.keys() != after.keys():
            raise RuntimeError("Checkpoint keys changed")
        for key in before.keys():
            if not np.array_equal(before.get_tensor(key), after.get_tensor(key)):
                changed.append(key)
    if changed != [TARGET]:
        raise RuntimeError(f"Unexpected changed tensors: {changed}")
    record = {"source_repo": manifest["repo_id"], "source_revision": manifest["revision"],
              "source_checkpoint_sha256": sha256(source),
              "checkpoint": str(destination), "checkpoint_sha256": sha256(destination),
              "changed_tensors": changed, "total_parameters": total,
              "candidate": args.candidate, **manifest["candidates"][args.candidate],
              "search_manifest_sha256": sha256(args.search / "edit.json"),
              "search_manifest": str((args.search / "edit.json").resolve()),
              "fit_instrument_centroids": manifest.get("fit_instrument_centroids", False),
              "calibration": manifest["calibration"], "formula": manifest["formula"],
              "runner_sha256": sha256(Path(__file__))}
    (args.output / "edit.json").write_text(json.dumps(record, indent=2) + "\n")
    (args.output / "runner.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k: v for k, v in record.items() if k != "calibration"}), flush=True)


if __name__ == "__main__":
    main()
