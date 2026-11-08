# Experimental patch

`gatefix.safetensors` is a separate version of the piano-to-flute edit. It keeps
the same output direction and 1.4 multiplier, but recalculates the gate with four
extra zero-response constraints: ambient, singing, flute-and-guitar, and
harp-and-strings audio references.

It was preferred over the main patch in two direct blind comparisons by one
listener. That does not establish a general quality improvement. The added
references and the inputs used to select this patch are not independent tests.
The gate is still linear and unbounded; the piano-and-strings reversal persists.

## Apply it

From the repository root, after the environment and `BASE` setup in the
[reproduction guide](../docs/REPRODUCE.md):

```sh
.env/bin/python scripts/materialize_patch.py --patch-dir experimental --candidate gatefix --base-checkpoint "$BASE" --output experimental_checkpoint
```

Apply it to the original pinned checkpoint, not on top of the main edit. The
command writes a separate full checkpoint and refuses to overwrite an output.

| Artifact | SHA-256 |
| --- | --- |
| Patch | `cf3e73f8f7a8f6efd7d4673948dde51a25c1747fca50542053a20088cef5585a` |
| Full edited checkpoint | `fea15f24d9e2774e0de16747489afcf56eb41bf0fcdcd06d51eb7b6c23c3dc4a` |

`edit.json` records the base identity, formula and added reference hashes. The
raw reference recordings needed to recalculate this variant are not included in
the bundle. A new evaluation on unseen references is still needed.
