# Evaluation notes

Results through 8 November 2026. The main patch is the 1.4× protected edit,
`stronger.safetensors`. The demo presents selected recordings; this file records
the broader results, failures and selection history.

## Checkpoint verification

The saved patch replaces one 768 × 1,024 matrix. Comparing the full checkpoints
found 786,432 changed FP32 values and no changes to the other 617 tensors. The
update is rank one before rounding and has 5.026% of the original adapter's
Frobenius norm.

Rebuilding from the recorded encoder tokens reproduced the patch hash exactly.
Materializing it against a relocated, hash-checked base reproduced the full
checkpoint hash. A fresh process also reproduced the selected development audio
byte for byte on the local runtime.

| Record | Location |
| --- | --- |
| Portable patch and checkpoint verification | [piano_package_verification/verification.json](../evidence/checkpoint_verification.json) |
| Fresh-process reload | [piano_protected_reload/reload_verification.json](../evidence/reload_verification.json) |
| Selected checkpoint and metadata | piano_protected_checkpoint/ (research archive: `results/piano_protected_checkpoint/`) |

## Fresh comparisons

A separate code review and generation run tested 16 new cases, plus four
flute-prompt baselines. Inputs and seeds were fixed before generation. Both full
checkpoints ran through the unmodified comparison script in separate processes;
a separate implementation of the 16-label CLAP screen scored the results.

| Check | Result | Interpretation |
| --- | --- | --- |
| Eligible piano cases → flute label | 6/6: three solo text, one ensemble, two audio references | Supports conversion on this small subset |
| Solo text control labels retained | 5/5 | Does not measure sound quality or unchanged notes |
| Out-of-domain audio control labels retained | 0/2 | Main edit does not reliably preserve these inputs |
| Blind listening: piano → flute | Four conversions confirmed | One listener; includes a trio with bass and drums retained |
| Blind quality: edit vs flute prompt | One preference each across two pairs | No established quality advantage |

The conversion denominator excludes the montuno request, whose original output
was harp-labelled, and the mixed reference whose original output was cello-labelled.
It is not a 6/6 success rate over every attempted input. CLAP cosine similarities
are neither instrument probabilities nor quality scores.

Both blind rounds used **one listener: the project owner**. Earlier development
listening was informal. Four prompt-baseline pairs were generated and scored,
but only two received a blind human quality comparison.

The [first listening record](../evidence/listening_round_1.json) and
[second listening record](../evidence/listening_round_2.json) are included unchanged.
The full review report remains in the research archive. This was a separate
AI-assisted audit, not external peer review. Two details in that report need
correction: the experimental gate is not bounded, and the singing-control
failure was not confirmed in the first listening round.

## Where it fails

| Input or comparison | Observation |
| --- | --- |
| Piano + strings audio reference | Original output sounded like strings; both edits added piano. The negative gate applies the shift backwards. |
| Piano + guitar text request | Both outputs sounded like piano and guitar. Conversion failed; quality feedback suggested degradation but was ambiguous. |
| Piano + strings, 30-second text request | Piano-with-strings remained the top caption in all three windows. No human confirmation of conversion. |
| Ambient audio reference | Main edit introduced flute. The original sounded piano-like and ambient, so this was not a verified piano-free control. |
| Singing audio reference | Classifier label changed from singing to piano. The original pair was not human-rated. |
| Organ control | Label retained, but the listener preferred the original's quality. |
| Experimental edit on a piano + drums reference | Piano receded and drums remained; the listener did not clearly hear flute. |

The last case uses an audio reference. It is distinct from the successful
text-conditioned piano-and-drums example.

## Earlier main-patch evaluation

The 16-case comparison generated 48 clips: original model, earlier centroid edit,
and protected edit. Each clip was 12 seconds. All eight piano baselines and eight
controls had their expected labels under the 16-instrument CLAP screen.

| Automatic screen | Centroid edit | Protected edit |
| --- | ---: | ---: |
| Piano text → flute | 3/4 | 4/4 |
| Piano audio reference → flute | 2/4 | 4/4 |
| Control labels retained | 7/8 | 8/8 |

These results are real, but the data were not fully independent of calibration.
Three synthetic audio controls reused calibration prompts with new seeds; piano
references came from the same generation batch. Four audio cases in the earlier
development set were the calibration files themselves. One synthetic piano
calibration input was also timbrally ambiguous: the broader classifier preferred
trumpet to piano.

The case manifest preceded generation. The written scoring protocol was saved
after generation began, so the thresholds were not preregistered. A real piano
recording and trumpet recording were outside calibration, but that does not make
the entire set held out.

The protected edit was preferred to the earlier edit on a development example
and a real-piano example. This is a comparison between edits, not evidence that
editing improves on the original model or a flute prompt. Mean CLAP similarity
to original control outputs was 0.950 for the protected edit and 0.957 for the
centroid edit; label retention did not imply closer audio embeddings.

Records: three-model comparison (research archive: `results/piano_protected_validation/`),
development listening (research archive: `results/listening_round_5/`) and
real-piano comparison (research archive: `results/listening_round_6/`).

### Strength ablation

The simple centroid edit was tested at the same 1.4 coefficient and at a 2.29169
coefficient chosen to match the protected update's Frobenius norm. This reused
the 16-case set; it was not a new generalization test.

| Variant | Piano → flute | Control labels retained |
| --- | ---: | ---: |
| Centroid, 1.4× | 7/8 | 7/8 |
| Centroid, matched norm | 7/8 | 8/8 |
| Protected, 1.4× | 8/8 | 8/8 |

The advantage over the matched-norm baseline is one case. Several parts of the
calibration differ, so this does not isolate which part helped. All 64 clips,
including cached comparisons, were checked for input pairing and audio hashes.
Records: piano_strength_ablation/ (research archive: `results/piano_strength_ablation/`).

### Thirty-second ensembles

Six prompts and seeds were fixed before generation; the protected patch had
already been selected. CLAP compared instrument captions across three
non-overlapping ten-second windows per clip.

| Arrangement | Result |
| --- | --- |
| Piano + drums | Listener heard piano → flute with drums retained; quality was not rated separately |
| Piano + guitar | Listener heard both instruments in both outputs; conversion failed |
| Piano + strings | Automatic screen did not support conversion |
| Flute + guitar control | Joint caption retained in all windows; no human confirmation |
| Guitar + drums control | Baseline itself drifted toward flute captions; inconclusive |
| Harp + strings control | Harp label retained; strings preservation unresolved |

Retained accompaniment means audible instruments, not identical notes or timing.
Records: ensemble outputs and scores (research archive: `results/piano_ensemble_validation/`),
listening round 7 (research archive: `results/listening_round_7/`) and
listening round 8 (research archive: `results/listening_round_8/`).

## Experimental patch

`gatefix.safetensors` adds four zero-response constraints for ambient, singing,
flute-and-guitar, and harp-and-strings audio. It retains the main edit's direction
and strength. Despite its name, it does not clip or saturate the gate; negative
responses remain possible.

The ambient and singing cases were used as constraints, so their later
preservation is in-sample. Humming, guitar-and-drums, trumpet and piano gate
values helped select among four variants; they are selection data too.

One listener preferred this patch to the main edit in two direct blind
comparisons. The piano-and-strings reversal persisted, and the original audio
was still preferred on some cases. Treat this as a follow-up candidate, not a
validated replacement. Application instructions and hashes are in the
[bundled experimental README](../experimental/README.md).

## Earlier experiments

These attempts informed the main edit but are not additional evidence for it.
Their scripts and generated records remain in the workspace.

| Attempt | Outcome | Records |
| --- | --- | --- |
| MiniMax unwanted vocals | A plain instrumental request contained vocals; an explicit no-vocals request did not. One seed and different output durations: insufficient reason to pursue a weight repair. | `results/baseline/`, `results/listening.json` |
| Magenta readiness | The 230M and 2.4B generators ran locally. The listener accepted the 2.4B piano sample as a usable baseline. | `results/magenta_smoke/`, `results/magenta_base_smoke/` |
| Piano-direction projection | First demo became much quieter. Independent CLAP kept the piano label for all 12 piano cases in the later screen; reject as piano erasure. | `results/piano_projection/`, `results/piano_validation/` |
| Individual-example retargeting | First listening pair still sounded like piano in both versions. | `results/piano_retarget_search/` |
| Mean-vector retargeting | Demonstrated flute, but later screen converted 3/4 text cases and 2/2 piano references; an audio control failed its label check. | `results/piano_centroid_search/`, `results/piano_centroid_validation/` |
| Protected retargeting | Expanded text/audio calibration and selected the 1.4× version. This is the current main patch. | `results/piano_protected_search/` |

The early projection used MusicCoCa as a scorer; its instrument labels disagreed
with independent CLAP and the first listener's feedback. Later screens used the
separate classifier and a broader instrument list. The early seven-label scores
remain archived rather than replaced by the newer results.

## What is shipped

The repository contains three selected listening examples, eight audio files, the two
labelled patches, saved calibration tokens, reproduction code and actual
weight-map data. It excludes full checkpoints, raw reference recordings, the
complete listening archive. These evaluation notes and four selected verification/listening records are included. The web page performs no
generation. Its file manifest and audio records support checking the packaged
examples, not reproducing every research result from the ZIP alone.

The next useful test is to freeze one patch and evaluate genuinely new inputs
with multiple blind listeners. Compare edited output against a flute prompt at
matched levels, rating instrument identity, quality and accompaniment separately.
For mechanism work, first address the unbounded gate: the current linear matrix
update cannot simply clip its response to an interval.
