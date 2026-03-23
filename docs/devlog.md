# Parallel Experiments LSDJ 2.0 - developer's log

## Milestones

- [2015] Initial idea/pass at PELSDJ
- [7/3/2020] exileFaker ["Parallel Experiments LSDJ"](https://exilefaker.bandcamp.com/album/parallel-experiments-lsdj) EP release
- [2/12/2026] Initial commit of v2
- [2/14❤️/2026] Tokenization pipeline working 
- [2/18/2026] First embedding pipeline working 
- [2/20/2026] First model + train loop
- [2/23/2026] Model v2
- [2/26/2026] Model v3-v4
- [2/27/2026] Model v5
- [3/3/2026] First generation pipeline

## Model version history

| Version | Class           | Description                                                  | Works |
|---------|-----------------|--------------------------------------------------------------|-------|
|   v1    | LSDJTransformer | cross-entropy loss used for all outputs including entity IDs | ✅    |
|   v2    | LSDJTransformer | "entity loss" (CE on entity features) used in place of instr, table, groove IDs | ✅ |
|   v3    | LSDJTransformer | hierarchical entity un-embedding covering softsynths and inner tables; MSE loss for continuous values | ✅ |
|   v4    | LSDJTransformer | true hierarchical decoding mirroring embedding structure. Predict individual groove/table steps rather than a single value | ❌ (OOM) |
|   v5    | LSDJTransformer | Conditionally generate FX values | ✅ |
|   v6    | LSDJTransformer | Add constrative loss for entity embedder/decoder | ❌ (OOM) |
|   v5.1  | LSDJTransformer | Adds instr_to_table_proj for conditioning table generation
|   v6    | LSDJTransformer | Add dropout and Gaussian noise on embedding | ✅ |
|   v7    | LSDJTransformer | Use Rotary Position Embeddings (RoPE) | ✅ |
|   v8    | LSDJTransformer | Add global position scalar; jax.nn attention kernel | ✅ |
|   v9    | LSDJTransformer | Replace MSE loss with Gaussian energy loss for continuous outputs | ❌ (training unstable) |
|   v9.1* | LSDJTransformer | Switch to Beta-NLL loss | ✅ |
|   v10   | LSDJTransformer | Helix encoding for note values | ✅ |
|   v11   | LSDJTransformer | Conditional FX value generation | ? |
|   v12   | LSDJTransformer | Helix encoding, MSE loss | ✅ |
|   v13   | LSDJTransformer | Chroma/Octave decoder for notes | ? |


\* current model

## Training methodology notes

### Phrase-aligned crop starts (deferred)
Currently `make_multi_track_batch` samples crop start positions uniformly at random.
A natural alternative is to bias starts toward phrase boundaries (`crop_start % 16 == 0`),
on the grounds that the first few steps of each crop have less attention context, and
steps that happen to land mid-phrase are a lower-quality training signal than steps
with a full preceding phrase in view.

Counterarguments: (1) RoPE positions give the model information about where it is within
a phrase even without prior context; (2) the benefit scales inversely with crop length —
for long crops (256+ steps) the effect is diluted. Suggested implementation when the time
comes: a `phrase_aligned_crops: bool` flag in `make_multi_track_batch` that rounds
`crop_start` down to the nearest multiple of 16. Probably most useful for short crop
lengths or early in training.

### Phrase-level H (Hop) commands
Phrase-level H commands cause a channel to jump to another phrase (or a specific
step within the current phrase). This creates a mismatch between the static phrase
layout the model sees and the actual playback sequence.

Two cases in practice:
1. **Synchronized hops** (3/4 time signatures, uniform phrase shortening): all channels
   hop together, so cross-channel alignment is preserved up to the hop. Not a problem.
2. **Desynchronized hops** (one channel "catches up" to the others later): the model
   sees incorrect cross-channel combinations until realignment. This is the only
   genuinely problematic case.

In the current training data, case 2 is rare, so this is noise rather than systematic
bias. RoPE phrase-step positions give the model some structural grounding even in
misaligned windows. No action taken; revisit if training data is expanded significantly
or if generation shows systematic phase-alignment errors.

A potential fix (if needed): at tokenization time, scan each channel's phrase for
CMD_H at step k and zero out steps k+1..15 for that channel (null note/instr/fx).
No model architecture changes required; "dead step" is already representable as
all-null tokens.

### Transpose augmentation distribution
Using `p_transpose` (probability of any transposition) + uniform over non-zero offsets,
rather than a Gaussian over offset magnitude. Rationale: semitone distance is not a
reliable proxy for musical naturalness — C→E (4 semitones) is a more common modulation
than C→C# (1 semitone) — so there is no clean ordinal relationship to exploit.
A Gaussian would impose a false metric. Current default: `p_transpose=0.2`.

### Helix embedding findings and next steps

Qualitative evaluation of v10 (helix, no augmentation, MSE loss) vs v8 showed:
- Main melodies frequently missing or only appearing intermittently
- More ±1 semitone pitch errors than v8, even without transpose augmentation
- Validation loss plateau ~2 nats higher than v8 despite 4× lower training loss

Hypothesis: the helix input geometry (circular chroma, linear octave) conflicts with
the flat CE output head over 157 notes. The backbone is pulled in two directions —
helix structure from the input side, uniform discrete discrimination from the output
side. In a small-data regime this degrades note accuracy.

**Conclusion**: helix is only expected to help when paired with a factorized chroma ×
octave output head (Step 3 of post_v8_plan.md). Helix as a standalone input change
appears to actively hurt note prediction.

**Next**: implement Step 3 (factorized chroma × octave output head + NULL flag head)
before training a new helix model. NULL note handling and valid-octave masking design
discussed in conversation history.

### Robustness to sparsity and prompt dropout (deferred)

Two related problems:

**Sparse cues (sustained chords, silence).** LSDJ represents sustained notes as a
single trigger followed by null steps. The model sees null steps and must infer that
something is still playing via long-range attention. The training signal from null
steps is weak (predicting null is easy, so gradient contribution is small), risking
the model learning to coast through sparse regions rather than maintaining harmonic
state. Possible fix: explicitly propagate the last active note/instrument forward into
null steps during tokenization — significant tokenizer change, deferred.

**Exposure bias / cold-start robustness.** Teacher forcing makes the model brittle to
weak or absent context at inference. Embedding noise (already in place) partially
addresses this. A targeted extension: **prompt dropout** — with some probability, zero
out the first k steps of each crop before feeding to the model, forcing generation from
weak or absent context. Cheap to implement as an augmentation in `make_multi_track_batch`,
no architecture changes. Deferred until Step 3 / helix architecture is settled.
