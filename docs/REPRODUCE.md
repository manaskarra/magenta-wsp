# Reproduce the edit

Use Python 3.12 and the pinned official MLX runtime. The original and edited full
checkpoints are each about 9.84 GB. Allow at least 25 GB of free disk space for
both and the model resources. Peak MLX allocation was about 15.7 GB in the local
run; CPU-side memory is additional. Other runtimes have not been verified.

Run these commands from the repository root:

```sh
python3.12 -m venv .env
.env/bin/python -m pip install -r requirements-magenta.txt
.env/bin/hf download google/magenta-realtime-2 --revision 010aa0dcb0dfd27b24f0ad07b4dad63e8f9521cc --include 'resources/*' 'checkpoints/mrt2_base.safetensors' README.md
```

Set `BASE` to the checkpoint inside the returned snapshot directory. Keep that
snapshot in the Hugging Face cache: the comparison script reads its resources
there. The materializer checks the base hash and refuses to overwrite an output.

```sh
BASE=/absolute/path/to/snapshot/checkpoints/mrt2_base.safetensors
.env/bin/python scripts/materialize_patch.py --base-checkpoint "$BASE" --output edited_checkpoint
.env/bin/python scripts/compare_piano.py --cases demo_cases.json --output reproduction
.env/bin/python scripts/compare_piano.py --edited --edit-record edited_checkpoint/edit.json --cases demo_cases.json --output reproduction
```

The last two commands run the original and edited checkpoints separately with
the same prompt, seeds and generation settings. Listen to
`reproduction/baseline/piano_demo.wav` and
`reproduction/edited/piano_demo.wav`. These retain raw output levels; the web
demo uses copies matched by RMS level. Use a fresh output directory for a rerun.
This text-only replay is separate from the three selected examples on the page.

## Recalculate the patch

The recorded encoder tokens are included. Recalculate the gate and output
direction from those tokens and the base weights with:

```sh
.env/bin/python scripts/rebuild_protected_patch.py --base-checkpoint "$BASE" --manifest patch/edit.json --direction-manifest patch/direction_calibration.json --output rebuilt.safetensors
```

The script requires an exact SHA-256 match with the published patch.
[edit.py](../scripts/edit.py) contains the gate calculation and token-to-vector mapping.
The saved tokens are enough to rebuild the patch exactly. Recollecting them
from audio would require the nine original synthetic references, which are not
included. The exploratory collection scripts remain in the research archive.

## Expected hashes

| Artifact | SHA-256 |
| --- | --- |
| Original checkpoint | `60f3e813d9da4a41a166c734a3074e6d54254c2fc14b0817bad6b8d25cddc044` |
| Main patch | `76c9e49fc2bcfead129b7992bddf1c968e02c06d20bccaae26adbaaca081959e` |
| Full edited checkpoint | `5a9e57012d9d78ac405cb969ac63a691f4f47f7386445034131fb875986b1bba` |

## Files and verification

The repo includes patches, saved calibration tokens, reproduction scripts,
selected recordings and evaluation records. Full checkpoints and raw reference
recordings are not included.

[manifest.json](../manifest.json) records published file sizes and hashes.
[demo_audio.json](../evidence/demo_audio.json) records output hashes and playback
level adjustments. Calibration metadata uses portable paths; rebuilding from
it reproduces the published patch exactly.

After changing the edit, rebuild it with the pinned runtime and base checkpoint.
Changes to musical behavior also need paired generation and listening comparisons.
