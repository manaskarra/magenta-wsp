# How the piano-to-flute edit works

The edit changes how Magenta interprets some piano-related style inputs. It
modifies one matrix between the style encoder and the music generator, then
saves that matrix into the model. The generator already knows how to make flute
sounds; the experiment redirects when it uses that ability.

## From a request to music

A model's weights are numbers used in its calculations. Training originally
adjusted those numbers to learn musical patterns. A checkpoint is a saved copy
of the weights. Loading a different checkpoint can change the model's behavior
even when the request and generation code stay the same.

For this edit, the relevant part of Magenta's flow is:

```text
Text prompt or reference recording
              ↓
Frozen MusicCoCa style encoder
              ↓
Style tokens → codebook lookup → 768-number style vector, x
              ↓
Style adapter, W: 768 × 1,024 weights   ← this is what we edit
              ↓
1,024-number conditioning signal
              ↓
Music generator → audio decoder → recording
```

This diagram shows the style path, not every part of the architecture. An audio
reference guides the style of newly generated music. It is not a recording that
gets its instrument replaced note for note.

The encoder represents the request numerically. Its style tokens select vectors
from a learned codebook; adding those vectors gives *x*. Individual entries do
not come with labels such as “piano” or “flute.” The adapter combines these 768
values into 1,024 values the generator can use.

We chose the adapter because of that architectural role. The calculation then
uses examples to decide how its weights should change.

## Two questions: what changes, and when?

The edit needs an **output direction**, *d*, that shifts conditioning toward
flute, and a **gate**, *g*, that controls how strongly to apply that shift to a
particular input.

To calculate the direction, we encode piano and flute descriptions and take the
mean vector for each group. Their difference describes a piano-to-flute shift
in the input space. Passing that difference through the original adapter puts
it in the generator's conditioning space:

```math
d = W^\top (\mu_{\mathrm{flute}} - \mu_{\mathrm{piano}})
```

Here, *μ* means a group average. The superscript ⊤ means transpose. All vectors
in these equations are columns: *x* has 768 entries and *d* has 1,024. The
direction comes from the earlier paired text calibration, whose tokens are
saved in `patch/direction_calibration.json`. It is an empirical direction;
there is no guarantee that it changes only instrument identity.

Applying the same shift to every request would also affect guitars, violins and
other inputs. The gate gives each style vector a scalar response:

```math
s(x) = x^\top g
```

This is a dot product: multiply corresponding entries of *x* and *g*, then add
the products. A response near one applies the intended shift; a response near
zero largely suppresses it. The response is not an instrument probability.

## Calculating the gate

The main gate uses 432 text encodings: nine instruments, 12 descriptions per
instrument and four style-mapper seeds. It also uses nine synthetic audio
references: three piano examples and six controls. These are frozen encoder
forward passes; the encoder's weights stay unchanged.

The constraints ask for a response of one on the piano text mean and piano
audio mean. They ask for zero on the eight other instrument means and each of
the six audio controls. These constraints apply to those calibration vectors;
they do not guarantee the response of every individual or unseen example.

Writing the constraint vectors as rows of *A*, and their requested responses as
*b*, the gate must satisfy *Ag = b*. Among solutions, we penalize variation
across piano examples and response to control examples:

```math
\min_g\; g^\top Qg \qquad \text{subject to}\quad Ag=b
```

A scaled identity term in *Q* regularizes the calculation. The closed-form
solution is:

```math
g = Q^{-1} A^\top \bigl(A Q^{-1} A^\top\bigr)^{+} b
```

The superscript + denotes a pseudoinverse. In [edit.py](scripts/edit.py),
`protected_gate()` uses a linear solve and a pseudoinverse, then checks that the
constraint errors are small. Python computes the coefficients directly. This
is fitting from calibration data, with no backpropagation or iterative optimizer
steps.

## Writing the change into the weights

Once we have *g* and *d*, the saved update is:

```math
W' = W + \alpha\,g d^\top, \qquad \alpha = 1.4
```

*W* is the original adapter and *W′* is its replacement. The outer product
*g d*⊤ makes a 768 × 1,024 matrix: entry *(i, j)* is *gᵢ dⱼ*. This coordinates the
changes across the matrix using one input direction and one output direction,
which is why it is called a rank-one update before FP32 rounding.

The multiplier 1.4 was selected through development comparisons and listening.
It is an experimental choice, not a mathematically guaranteed optimum.

Multiplying an input by the edited adapter gives:

```math
x^\top W' = x^\top W + \alpha\,(x^\top g)\,d^\top
```

Read that as: **original conditioning + strength × gate response × flute
shift**. The identity follows by distributing the matrix multiplication over
the addition. It is also why the edit needs no extra gate code during generation:
the ordinary multiplication by *W′* already produces the whole expression.

For illustration, a gate response of 1 adds 1.4 times the direction; a response
of 0 adds nothing. A response of −0.5 adds −0.7 times the direction, reversing
the intended shift. These are example scores, not measurements of demo clips.
The actual gate is linear and unbounded, and a negative response explains the
piano-and-strings failure recorded in [EVALUATION.md](docs/EVALUATION.md).

## Save, reload, compare

The main patch stores the complete replacement matrix, about 3.15 MB. Applying
it checks the original checkpoint and patch hashes, replaces the target tensor,
and saves a full checkpoint. A comparison verifies that the other 617 tensors
are unchanged.

The changed matrix contains 786,432 values: 0.032% of the full checkpoint's
2,459,163,592 parameters, including the codec. That percentage measures how many
values changed, not the size of each change. Relative to the adapter itself,
the update's Frobenius norm is 5.026% of the original matrix's norm.

The reproduction scripts run original and edited checkpoints separately with
matched prompts, random seeds and generation settings. The saved verification
records show an exact patch rebuild from recorded tokens and a fresh-process
reload that reproduced one development recording byte for byte on the tested
runtime. The model runs its original inference code; no activation hook or
hidden flute prompt is required.

## What the result supports

Selected examples show piano-to-flute changes from text and audio-style inputs.
Some mixed arrangements fail, and preserving a control's instrument label does
not guarantee unchanged sound quality. Two blind quality comparisons against
simply asking for flute split one preference each, with one listener. We have
not established a quality advantage over prompting.

The newer experimental patch adds four zero-response constraints. It keeps the
same direction and strength, and its gate is still unbounded. It needs further
evaluation; it is not a replacement for the main result.

The contribution is a documented, reproducible experiment in permanently
redirecting an existing music model's behavior through a small weight edit.
The linear-algebra method has prior art. GPT-6 Astra designed and implemented
the main edit using Python and encoder examples; the project owner set the goal
and supplied listening feedback. The process does not demonstrate reading the
meaning of arbitrary raw weights alone.

## A short way to present it

> Magenta already knows how to generate piano and flute. I explored whether a
> permanent weight edit could make the same piano request produce flute. The
> edit changes one small matrix that carries style information into the music
> generator. Piano and flute examples define a direction, and control examples
> help determine when that direction should apply. Python calculates the update
> without backpropagation, and it is saved into the checkpoint. The demo shows
> selected before-and-after results from changing 0.032% of the model's weights.
> Mixed arrangements remain a limitation, and it is not proven better than
> simply prompting for flute.
