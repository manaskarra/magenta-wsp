# Attribution and licenses

The project's code, original documentation and demo interface are covered by
the [MIT license](../LICENSE), copyright 2026 manaskarra. That license does not
replace the model, patch or reference-audio terms below, and does not assert
commercial clearance for the generated recordings.

## Model and patches

**Magenta RealTime 2**, Google DeepMind / Google LLC, copyright 2026 Google LLC.
The base model is from [google/magenta-realtime-2](https://huggingface.co/google/magenta-realtime-2),
revision `010aa0dcb0dfd27b24f0ad07b4dad63e8f9521cc`, under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The original [model card and usage terms](MODEL_CARD.md) are preserved in this bundle.

`patch/stronger.safetensors` and `experimental/gatefix.safetensors` each replace
one style adapter with a matrix calculated by this project. Both contain
modified model material and retain the original attribution and CC BY 4.0 license.
No endorsement by Google or the Magenta authors is implied.

The reproduction code uses the unmodified
[magenta/magenta-realtime runtime](https://github.com/magenta/magenta-realtime),
commit `694a545e4ba0b88bf1150137b129582166d3e07f`, licensed under Apache 2.0.
The runtime is installed from upstream; its source is not bundled.

## Reference audio

| Recording | Creator | License | Use |
| --- | --- | --- | --- |
| [Pistachio Ice Cream Ragtime](https://librosa.org/data/audio/442789__lena-orsa__happy-music-pistachio-ice-cream-ragtime.hq.ogg) | The Piano Lady / Lena Orsa | [CC BY-NC 3.0](https://creativecommons.org/licenses/by-nc/3.0/) | Piano-style input: 12-second excerpts starting at 10 seconds in early tests and 40 seconds in the demo |
| [solo-trumpet-06](https://librosa.org/data/audio/sorohanro_-_solo-trumpet-06.hq.ogg) | Mihai Sorohan | [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/) | Trumpet control in evaluation |

Both recordings are listed in the
[librosa examples](https://librosa.org/doc/main/recordings.html). Their original
attribution text is preserved in `attribution/`.

The WAVs in this bundle are model-generated outputs, not copies of the reference
recordings. They are level-matched listening copies; no external recording was
mixed into them. The demo's `fa_real_piano_40s` case uses the noncommercial piano
reference. Earlier research records use the ID `audio_real_piano`. This bundle
does not assert commercial clearance for outputs conditioned on that reference.
Other audio-style inputs were synthetic model outputs.

## Inspiration and implementation

[Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/)
by Rikiya Takehi, Jayden Teoh, Alex Zhang and Omar Khattab, published 2 November
2026, inspired the experiment. **GPT-6 Astra designed and implemented the main
edit**, using Python and frozen encoder examples. The project owner supplied
the goal and listening feedback. The newer experimental patch is a separate
follow-up; the post's authors did not make either patch.

[Unified Concept Editing](https://unified.baulab.info/) is related prior work on
closed-form concept edits. The [README](../README.md) describes the calculation used
here and the scope of the WSP attribution.
