# The weight update

MusicCoCa, the frozen style encoder, accepts text or reference audio. Its style
tokens select codebook vectors that sum to a 768-dimensional vector, *x*. The
768 × 1,024 adapter matrix, *W*, maps this to a 1,024-dimensional generator input.
The equations below use column vectors; ⊤ denotes transpose.

The edit has two parts:

- **Direction, *d*:** the difference between mean flute and piano vectors,
  projected through the original adapter. This comes from the earlier paired
  text calibration.
- **Gate, *g*:** a vector calculated from 432 text encodings and nine synthetic
  audio references. The scalar *x*⊤*g* controls how much of the direction to add.

Writing the mean style vectors as *μ*, the output direction is:

```math
d = W^\top (\mu_{\mathrm{flute}} - \mu_{\mathrm{piano}})
```

The saved matrix uses an outer-product update, with *α* = 1.4:

```math
W' = W + \alpha\,g d^\top
```

For any input *x*, this gives:

```math
x^\top W' = x^\top W + \alpha\,(x^\top g)\,d^\top
```

The 432 text encodings cover nine instruments, 12 descriptions per instrument,
and four style-mapper seeds. The audio calibration has three piano references
and six controls. Constraints set the piano text-mean and audio-mean responses
to 1, and the eight other instrument means and six audio-control responses to 0.

`protected_gate()` solves these constraints in closed form:

```math
g = Q^{-1} A^\top \bigl(A Q^{-1} A^\top\bigr)^{+} b
```

Here, *A* contains the constraint vectors and *b* their requested responses.
*Q* penalizes variation across piano examples and response to control examples;
a scaled identity term keeps the solve regularized. The implementation uses a
linear solve and a pseudoinverse (the superscript `+`), then checks the constraint
residuals.

The strength of 1.4 was selected through development comparisons and listening.
The update is calculated from calibration examples, with no gradients, optimizer
steps or runtime activation hooks.

The gate is linear and unbounded. If *x*⊤*g* is negative, it applies the direction
backwards. This explains a failure on a piano-and-strings reference and remains
a limitation of both saved patches.
