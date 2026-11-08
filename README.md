# magenta-wsp

A weight-space programming experiment that edits [Google Magenta RealTime 2](https://huggingface.co/google/magenta-realtime-2)
to shift piano-conditioned music toward flute.
The same prompt or audio-style reference can produce piano with the original
weights and flute with the edited weights.

The edit replaces one 768 × 1,024 matrix: **786,432 values, or 0.032% of the
checkpoint**. It is calculated from piano, flute and control examples without
backpropagation, then saved into the checkpoint. Generation uses the original
inference code.

## Listen

The demo pairs three selected recordings and lets you inspect the actual weight
changes. It plays saved audio; it does not run Magenta in the browser.

The first example gives both models the same prompt:

> A solo honky-tonk piano plays a fast boogie-woogie, only piano.

**Original model** produces piano; **Edited model** produces flute. **Ask for
flute** is a separate comparison: the original model receives a flute prompt.
The demo shows the input used for each clip.

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

## Reproduce

Python 3.12, Apple silicon and at least 25 GB of free disk space.
Follow the [reproduction guide](docs/REPRODUCE.md) to apply the patch, generate
comparisons or rebuild it from the saved calibration tokens.

## Development

CPU tests; no model download:

```sh
python3.12 -m venv .test-env
.test-env/bin/python -m pip install -r requirements-test.txt
.test-env/bin/python scripts/build_demo.py
.test-env/bin/python -m pytest
```

Run the build after edits to refresh the demo documents and checksums.
GitHub Actions checks both.

## Credits

Inspired by [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/)
by Rikiya Takehi, Jayden Teoh, Alex Zhang and Omar Khattab.
**GPT-6 Astra designed and implemented the main weight edit.**
Related work: [Unified Concept Editing](https://unified.baulab.info/).

## License

[MIT](LICENSE) for the code, original docs and demo interface.
[Model, patch and audio terms](docs/ATTRIBUTION.md) are listed separately.
