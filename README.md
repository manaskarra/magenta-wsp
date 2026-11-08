# magenta-wsp

A piano-to-flute weight edit for [Google Magenta RealTime 2](https://huggingface.co/google/magenta-realtime-2).
The same prompt or audio-style reference can produce piano with the original
weights and flute with the edited weights.

The edit replaces one 768 × 1,024 matrix: **786,432 values, or 0.032% of the
checkpoint**. It is calculated from piano, flute and control examples without
backpropagation, then saved into the checkpoint. Generation uses the original
inference code.

## Listen

The demo pairs three selected recordings and lets you inspect the actual weight
changes. It plays saved audio; it does not run Magenta in the browser.

```sh
git clone https://github.com/manaskarra/magenta-wsp.git
cd magenta-wsp
python3 -m http.server 8000 --bind 127.0.0.1
```

Open [localhost:8000](http://127.0.0.1:8000/). The [method](METHOD.md) covers the
calculation in detail; the [evaluation notes](docs/EVALUATION.md) record the comparisons,
listening feedback and failures.

## How the weights were chosen

The edited matrix connects the frozen text/audio style encoder to the music
generator:

```text
params/depthformer/encoder/body/layers_1/branched_mulan_embedder/mulan_embedder/depth_input_adapter/kernel
```

This is where the style input enters the generator. Piano and flute examples
define a direction, *d*, in the adapter's output space. A gate, *g*, controls
how strongly that direction applies to each input. It is calculated from 432
text encodings and nine synthetic audio references, with constraints that give
piano means a response of one and control means or references a response of zero.

The saved update is:

```math
W' = W + \alpha\, g d^\top
```

For an input style vector *x*, the adapter then produces:

```math
x^\top W' = x^\top W + \alpha\,(x^\top g)\,d^\top
```

*W* is the original matrix and *W′* is the edited matrix. The vectors are written
as columns: *x* and *g* have 768 entries, and *d* has 1,024. The superscript ⊤
means transpose, so *g d*⊤ is an outer product.

The dot product *x*⊤*g* sets the strength of the flute shift. The multiplier
*α* = 1.4 was selected through development comparisons and listening. The update
is rank one before FP32 rounding; the patch stores the full replacement matrix.
Once saved, it needs no extra prompt or runtime activation hook.

## Results and limits

The selected examples show piano-to-flute changes from text and audio-style
inputs. Audio references guide new music generation; the edit does not convert
a recording note for note. Mixed arrangements can fail, and a negative gate
response can reverse the intended change.

The project owner was the only listener. Two blind quality comparisons against
asking the original model for flute split 1–1. That does not establish a quality
advantage over prompting. The boogie example includes this baseline.

There are two patches:

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

### Recalculate the patch

The recorded encoder tokens are included. Recalculate the gate and output
direction from those tokens and the base weights with:

```sh
.env/bin/python scripts/rebuild_protected_patch.py --base-checkpoint "$BASE" --manifest patch/edit.json --direction-manifest patch/direction_calibration.json --output rebuilt.safetensors
```

The script requires an exact SHA-256 match with the published patch.
[edit.py](scripts/edit.py) contains the gate calculation and token-to-vector mapping.
The saved tokens are enough to rebuild the patch exactly. Recollecting them
from audio would require the nine original synthetic references, which are not
included. The exploratory collection scripts remain in the research archive.

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

The experiment uses Python and encoder examples to calculate an edit; it does
not demonstrate interpreting raw weights alone. It also differs from the
original post's [Qwen protocol](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/README.md).
Closed-form concept editing has prior art, including
[Unified Concept Editing](https://unified.baulab.info/); the editing algorithm
is not claimed as new.

## Repository layout

```text
docs/             Evaluation, attribution and the upstream model card
scripts/          Apply the patch, rebuild it, generate comparisons, build the demo
patch/            Main adapter and the calibration tokens used to calculate it
experimental/     Follow-up adapter and its separate instructions
tests/            Small CPU tests; no model download
audio/            Eight listening clips across three selected examples
evidence/         Checkpoint checks, listening records and weight-map data
```

The demo is `index.html`, `demo.js` and `demo.css`. [METHOD.md](METHOD.md) explains
how the edit works; [EVALUATION.md](docs/EVALUATION.md) records what worked and what
failed. Full checkpoints and raw reference recordings are not included.

[manifest.json](manifest.json) records the published file sizes and hashes.
[demo_audio.json](evidence/demo_audio.json) records output hashes and the level
adjustments used for playback. Calibration metadata uses portable paths;
rebuilding from it reproduces the published patch exactly.

## Development

Use Python 3.12. These checks run on a CPU without the Magenta runtime:

```sh
python3.12 -m venv .test-env
.test-env/bin/python -m pip install -r requirements-test.txt
.test-env/bin/python scripts/build_demo.py
.test-env/bin/python -m pytest
```

Run `build_demo.py` from a Git checkout after changing a published file. It
refreshes the formatted Markdown in the demo and the file checksums. The tests
cover gate constraints, checkpoint application, command-line validation, audio
integrity and the document reader. GitHub Actions runs the same checks.

For a change to the edit itself, also rebuild the real patch with the pinned
runtime and base checkpoint. CPU fixtures do not establish musical quality;
new behavior needs paired generation and listening comparisons.

## License

The project's code, original documentation and demo interface are under the
[MIT license](LICENSE). Model-derived patches, the upstream model card and
reference recordings retain their own terms. See [ATTRIBUTION.md](docs/ATTRIBUTION.md)
for those licenses and the generated audio's provenance.
