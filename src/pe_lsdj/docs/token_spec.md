# PE-LSDJ Token Specification
# [Generated upon request by Claude Opus 4.6]

Token layout for the LSDJ ML pipeline. All values use **+1 null offset**
(0 = NULL) unless noted otherwise.

## Overview

The model consumes two kinds of data:

1. **Reusable entities** — instruments, tables, grooves, and softsynths.
   These are defined once and referenced by ID from the song sequence.
   The generation strategy (upfront vs. incremental) is flexible; see
   the Embedding Notes section.
2. **Song sequence** — the main autoregressive sequence, one token vector
   per step across 4 channels.

---

## 1. Reusable Entities

### Grooves (32 grooves)

Each groove has 16 steps. Each step is nibble-split into two tick values.

| Field           | Shape per groove | Vocab | Source          |
|-----------------|-----------------|-------|-----------------|
| even_step_ticks | (16,)           | 17    | high nibble + 1 |
| odd_step_ticks  | (16,)           | 17    | low nibble + 1  |

**Total**: (32, 16, 2) — 32 features per groove.

### Softsynths (16 synths)

WAV-channel waveform parameters. 13 scalar fields per synth.

| Field               | Vocab | Parse          |
|---------------------|-------|----------------|
| Waveform            | 4     | enum [0-2] + 1 |
| Filter type         | 5     | enum [0-3] + 1 |
| Filter resonance    | 257   | byte + 1       |
| Distortion          | 3     | enum [0-1] + 1 |
| Phase type          | 4     | enum [0-2] + 1 |
| Start volume        | 257   | byte + 1       |
| Start filter cutoff | 257   | byte + 1       |
| Start phase amount  | 257   | byte + 1       |
| Start vert. shift   | 257   | byte + 1       |
| End volume          | 257   | byte + 1       |
| End filter cutoff   | 257   | byte + 1       |
| End phase amount    | 257   | byte + 1       |
| End vert. shift     | 257   | byte + 1       |

**Total**: (16, 13).

### Instruments (64 instruments)

30 scalar fields per instrument. Many fields are type-conditional
(zeroed for irrelevant instrument types).

| Field            | Vocab | Used by     | Parse              |
|------------------|-------|-------------|--------------------|
| Type ID          | 5     | All         | enum [0-3] + 1     |
| Table ID         | 33    | All         | 5-bit + 1          |
| Table on/off     | 3     | All         | bit + 1            |
| Table automate   | 3     | All         | bit + 1            |
| Automate 2       | 3     | All         | bit + 1            |
| Pan              | 5     | All         | enum [0-3] + 1     |
| Vibrato type     | 5     | PU,WAV,KIT  | enum [0-3] + 1     |
| Vibrato direction| 3     | PU,WAV,KIT  | bit + 1            |
| Env volume       | 17    | PU,NOI      | high nibble + 1    |
| Env fade         | 17    | PU,NOI      | low nibble + 1     |
| Length           | 65    | PU,NOI      | 6-bit + 1          |
| Length limited   | 3     | PU,NOI      | bit + 1            |
| Sweep            | 257   | PU,NOI      | byte + 1           |
| Volume           | 5     | WAV,KIT     | 2-bit + 1          |
| Phase transpose  | 257   | PU          | byte + 1           |
| Wave             | 3     | PU          | bit + 1            |
| Phase finetune   | 17    | PU          | 4-bit + 1          |
| Softsynth ID     | 17    | WAV         | high nibble + 1    |
| Repeat           | 17    | WAV         | low nibble + 1     |
| Play type        | 5     | WAV         | enum [0-3] + 1     |
| Wave length      | 17    | WAV         | high nibble + 1    |
| Speed            | 17    | WAV         | low nibble + 1     |
| Keep attack 1    | 3     | KIT         | bit + 1            |
| Keep attack 2    | 3     | KIT         | bit + 1            |
| Kit 1 ID         | 65    | KIT         | 6-bit + 1          |
| Kit 2 ID         | 65    | KIT         | 6-bit + 1          |
| Length kit 1     | 257   | KIT         | byte + 1           |
| Length kit 2     | 257   | KIT         | byte + 1           |
| Loop kit 1       | 3     | KIT         | bit + 1            |
| Loop kit 2       | 3     | KIT         | bit + 1            |
| Offset kit 1     | 257   | KIT         | byte + 1           |
| Offset kit 2     | 257   | KIT         | byte + 1           |
| Half-speed       | 3     | KIT         | bit + 1            |
| Pitch            | 257   | KIT         | byte + 1           |
| Distortion type  | 5     | KIT         | enum [0-3] + 1     |

**Total**: (64, 30). Column-stacked in dict-key order above.

Note: `Table ID` and `Softsynth ID` reference reusable entities.
At embedding time, these IDs should resolve to learned representations
rather than raw ID embeddings (see Embedding Notes).

### Tables (32 tables × 16 steps)

Two representations with identical shape and keys:

- **Raw tables**: original data with A/H commands intact. Primary
  representation; used for entity-level embedding.
- **Traces**: A/H commands resolved into execution traces. Used within
  table definitions to avoid recursive references.

| Field           | Shape per table | Vocab | Parse              |
|-----------------|----------------|-------|--------------------|
| Env volume      | (16,)          | 17    | high nibble + 1    |
| Env duration    | (16,)          | 17    | low nibble + 1     |
| Transpose       | (16,)          | 257   | byte + 1           |
| FX command 1    | (16,)          | 19    | enum [0-18], no +1 |
| FX value 1      | (16, 17)       | mixed | see FX Values      |
| FX command 2    | (16,)          | 19    | enum [0-18], no +1 |
| FX value 2      | (16, 17)       | mixed | see FX Values      |

**Total per step**: 2 + 1 + 1 + 17 + 1 + 17 = 39 features.
**Total per table**: (16, 39) = 624 features.

---

## 2. Song Sequence

### Sequence layout

The song is flattened to a single sequence ordered by channel:

```
[PU1 steps] [PU2 steps] [WAV steps] [NOI steps]
```

Each channel contributes `NUM_SONG_CHAINS (256) × PHRASES_PER_CHAIN (16)
× STEPS_PER_PHRASE (16) = 65536` steps.

**Total sequence length**: 4 × 65536 = 262144 steps.

### Per-step feature vector

| Feature          | Dims | Vocab | Source / Notes                       |
|------------------|------|-------|--------------------------------------|
| Note             | 1    | 159   | Pitch [0-157] + 1; invalid → 0      |
| Chain transpose  | 1    | 257   | Per-phrase, broadcast across steps   |
| FX command       | 1    | 19    | Enum [0-18], no +1 offset            |
| FX values        | 17   | mixed | Sparse: only 1 group active per step |
| Instrument       | 30   | mixed | Inlined from instrument palette      |

**Raw feature count per step**: 1 + 1 + 1 + 17 + 30 = **50 scalar tokens**.

### FX Values breakdown (17 features)

Sparse vector — at most one group is non-zero per step, determined by the
FX command. All use +1 null offset. Column order matches `FX_VALUE_KEYS`.

| Col | Key             | Vocab | Active when | Parse           |
|-----|-----------------|-------|-------------|-----------------|
| 0   | Table FX        | 33    | CMD_A       | 5-bit ID + 1    |
| 1   | Groove FX       | 33    | CMD_G       | 5-bit ID + 1    |
| 2   | Hop FX          | 257   | CMD_H       | byte + 1        |
| 3   | Pan FX          | 5     | CMD_O       | enum [0-3] + 1  |
| 4   | Chord FX 1      | 17    | CMD_C       | high nibble + 1 |
| 5   | Chord FX 2      | 17    | CMD_C       | low nibble + 1  |
| 6   | Env FX vol      | 17    | CMD_E       | high nibble + 1 |
| 7   | Env FX fade     | 17    | CMD_E       | low nibble + 1  |
| 8   | Retrig FX fade  | 17    | CMD_R       | high nibble + 1 |
| 9   | Retrig FX rate  | 17    | CMD_R       | low nibble + 1  |
| 10  | Vibrato speed   | 17    | CMD_V       | high nibble + 1 |
| 11  | Vibrato depth   | 17    | CMD_V       | low nibble + 1  |
| 12  | Volume FX       | 257   | CMD_M       | byte + 1        |
| 13  | Wave FX         | 5     | CMD_W       | enum [0-3] + 1  |
| 14  | Random FX L     | 17    | CMD_Z       | high nibble + 1 |
| 15  | Random FX R     | 17    | CMD_Z       | low nibble + 1  |
| 16  | Continuous FX   | 257   | CMD_D/F/K/L/P/S/T | byte + 1 |

Columns 0 and 1 (Table FX, Groove FX) are entity IDs — at embedding
time, these should resolve to learned representations of the referenced
entity rather than raw ID embeddings.

### Embedding Notes

Entity references (instrument IDs, table IDs, groove IDs, softsynth IDs)
appear both in the song sequence and within entity definitions themselves.
At embedding time, each ID should resolve to a learned representation of
the referenced entity — the exact mechanism is architecture-dependent.

Possible strategies include:

- **Upfront generation**: all entities are generated before the song
  sequence, forming an embedding bank ("palette") for lookup during sequence
  generation. In this architecture, the model outputs a query vector 
  in the same space as the palette embeddings. Cosine similarity produces a categorical distribution over palette entries (tables, grooves, instruments).
  
- **Incremental generation**: entities are defined on-the-fly as the
  song is generated, with new entities created when needed (e.g., based
  on cosine similarity with existing ones).

The tokenizer is agnostic to this choice — it produces the same token
arrays regardless. The embedding/generation strategy is an architecture
decision.

```
Entity encoders:
  Grooves  (32)  ──┐
  Softsynths (16) ─┤
  Tables (32)  ────┤──→  Entity representations
  Instruments (64) ┘

Song sequence (per step):
  note ─────────────→ Embedding(159)     ──┐
  chain_transpose ──→ Embedding(257)     ──┤
  fx_command ───────→ Embedding(19)      ──┤
  fx_values ────────→ FXValueEmbedder    ──┤──→ concat → Linear → d_model
  instrument_ID ────→ entity lookup      ──┤
  table_fx_ID ──────→ entity lookup      ──┤
  groove_fx_ID ─────→ entity lookup      ──┘
```


For generation: the model outputs a query vector in the same space as the
      204 -palette embeddings. Cosine similarity produces a categorical distribution
      205 -over palette entries (tables, grooves, instruments).