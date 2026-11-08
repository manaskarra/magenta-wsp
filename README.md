# magenta-wsp

Piano to flute: one matrix, changed. Maintained by [manaskarra](https://github.com/manaskarra).

This bundle contains a permanent weight edit for **Google Magenta RealTime 2**.
One 768 × 1,024 matrix changes — **786,432 values, or 0.032% of the checkpoint**.
The edited model can generate flute from the same input that previously produced
piano, using the original inference code.

The listening demo has three selected examples and an interactive view of the
actual weights. It plays recorded outputs; it does not run the model in your browser.
To view it, clone this repo and run:

```sh
git clone https://github.com/manaskarra/magenta-wsp.git
cd magenta-wsp
python3 -m http.server 8000 --bind 127.0.0.1
```

Open [localhost:8000](http://127.0.0.1:8000/). For the research details, read the
[full weight calculation](METHOD.md) and [evaluation notes](EVALUATION.md).

## How the weights were chosen

The target is the style adapter between the frozen text/audio encoder and the
music generator:

```text
params/depthformer/encoder/body/layers_1/branched_mulan_embedder/mulan_embedder/depth_input_adapter/kernel
```

Its role in style conditioning motivated the choice. Piano and flute examples
provide a direction, *d*, in the adapter's output space. A second calibration
uses 432 text encodings and nine synthetic audio references to calculate a gate,
*g*: piano means should activate the edit, while control inputs should suppress it.

Python solves those constraints and writes the updated matrix:

```math
W' = W + \alpha\, g d^\top
```

For an input style vector *x*, the adapter then produces:

```math
x^\top W' = x^\top W + \alpha\,(x^\top g)\,d^\top
```

Here, *W* is the original 768 × 1,024 matrix and *W′* is the edited matrix.
All vectors are written as columns: *x* and *g* have 768 entries, and *d* has
1,024. The superscript ⊤ means transpose; *g d*⊤ is their outer product.
The scalar *x*⊤*g* controls the strength of the flute shift for that input.
The coefficient *α* = 1.4 was chosen through development comparisons and listening.

This is a rank-one update before FP32 rounding. The patch stores the complete
replacement matrix. After it is saved into a checkpoint, ordinary inference
uses it without an extra prompt or runtime hook. No gradients or optimizer
steps were used; the calculation does use calibration examples.

## What the demo establishes

The same input can produce a different instrument after a weight edit. Audio
references supply style: this is not note-for-note conversion of a recording.
The examples are selected successes, and mixed arrangements can fail. The gate
is unbounded; a negative response can reverse the intended change.

Listening used one project-owner listener. Two blind quality comparisons against
asking the original model for flute split 1–1, so no quality advantage over that
prompt baseline is established. The boogie example includes the baseline.

The two patches are separate:

| Patch | Status |
| --- | --- |
| `patch/stronger.safetensors` | Main edit used for the demo |
| `experimental/gatefix.safetensors` | Four extra audio-preservation constraints; no independent final evaluation |

The [experimental instructions](experimental/README.md) explain that variant and
its limits. Apply either patch to the original model, not to the other patch.

## Reproduce on Apple silicon

Use Python 3.12 and the pinned official MLX runtime. The original and edited full
checkpoints are each about 9.84 GB. Allow at least 25 GB of free disk space for
both and the model resources. Peak MLX allocation was about 15.7 GB in the local
run; CPU-side memory is additional. Other runtimes have not been verified.

Run these commands from the unzipped bundle directory:

```sh
python3.12 -m venv .env
.env/bin/python -m pip install -r code/requirements-magenta.txt
.env/bin/hf download google/magenta-realtime-2 --revision 010aa0dcb0dfd27b24f0ad07b4dad63e8f9521cc --include 'resources/*' 'checkpoints/mrt2_base.safetensors' README.md
```

Set `BASE` to the checkpoint inside the returned snapshot directory. Keep that
snapshot in the Hugging Face cache: the comparison script reads its resources
there. The materializer checks the base hash and refuses to overwrite an output.

```sh
BASE=/absolute/path/to/snapshot/checkpoints/mrt2_base.safetensors
.env/bin/python code/materialize_patch.py --search patch --candidate stronger --base-checkpoint "$BASE" --output edited_checkpoint
.env/bin/python code/compare_piano.py --cases demo_cases.json --output reproduction
.env/bin/python code/compare_piano.py --edited --edit-record edited_checkpoint/edit.json --cases demo_cases.json --output reproduction
```

The last two commands load the original and edited checkpoints in separate
processes with the same prompt, seeds and generation settings. Listen to
`reproduction/baseline/piano_demo.wav` and
`reproduction/edited/piano_demo.wav`. These retain raw output levels; the web
demo uses copies matched by RMS level. Use a fresh output directory for a rerun.
This text-only replay is separate from the three selected examples on the page.

### Recalculate the patch

The bundle includes the original frozen-encoder tokens. To recalculate the gate
and output direction from those tokens and the base weights:

```sh
.env/bin/python code/rebuild_protected_patch.py --base-checkpoint "$BASE" --manifest patch/edit.json --direction-manifest patch/direction_calibration.json --output rebuilt.safetensors
```

The script checks that the resulting patch matches the recorded SHA-256 exactly.
`code/protected_piano.py` shows how calibration was first collected. Collecting it
again requires nine synthetic audio references from the research workspace;
those recordings are not included here.

### Expected hashes

| Artifact | SHA-256 |
| --- | --- |
| Original checkpoint | `60f3e813d9da4a41a166c734a3074e6d54254c2fc14b0817bad6b8d25cddc044` |
| Main patch | `76c9e49fc2bcfead129b7992bddf1c968e02c06d20bccaae26adbaaca081959e` |
| Full edited checkpoint | `5a9e57012d9d78ac405cb969ac63a691f4f47f7386445034131fb875986b1bba` |

## Inspiration and authorship

Inspired by [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/)
by Rikiya Takehi, Jayden Teoh, Alex Zhang and Omar Khattab. **GPT-6 Astra designed
and implemented the main edit**, using Python and frozen encoder examples. The
project owner set the goal and provided listening feedback. The experimental
patch is a separate follow-up.

This is a human-guided, tool-assisted WSP experiment with closed-form calibration.
The original [Qwen experiment logs](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/README.md)
also use Python and forward passes; this project does not replicate that protocol
or demonstrate interpretation from raw weights alone. Closed-form concept editing
has prior art, including [Unified Concept Editing](https://unified.baulab.info/).
The editing algorithm itself is not claimed as new.

## Files and sharing

Unzip the whole folder and open `index.html`, or host the folder as a static site.
Keep its files together. The ZIP contains eight listening clips across three
examples, both patches, reproduction code, calibration tokens and weight-map data.
Full checkpoints, raw reference recordings and the complete research archive are
not included.

`manifest.json` records each file's size and SHA-256. The ZIP sidecar records the
archive hash. [evidence/demo_audio.json](evidence/demo_audio.json) records raw
output hashes and the gains used for the listening copies. See
[ATTRIBUTION.md](ATTRIBUTION.md) for model and reference-audio licenses.

## Tests

The tests cover the gate constraints, patch/checkpoint identity checks, audio
integrity and paired evaluation inputs. They run without downloading the model:

```sh
python3.12 -m venv .test-env
.test-env/bin/python -m pip install -r requirements-test.txt
.test-env/bin/python -m pytest
```

The published calibration metadata uses relative paths; its file hashes reflect
that relocation. The saved patch and audio files are unchanged. This repository
contains the selected release and its reproduction code, not the complete raw
research archive. A model download is required to generate new audio.
