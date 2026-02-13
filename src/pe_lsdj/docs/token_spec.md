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

35 scalar fields per instrument (before inlining). Many fields are
type-conditional (zeroed for irrelevant instrument types).

| Field            | Dims | Vocab | Used by     | Parse              |
|------------------|------|-------|-------------|--------------------|
| Type ID          | 1    | 5     | All         | enum [0-3] + 1     |
| Table            | 1616 | —     | All         | inlined raw table  |
| Table on/off     | 1    | 3     | All         | bit + 1            |
| Table automate   | 1    | 3     | All         | bit + 1            |
| Automate 2       | 1    | 3     | All         | bit + 1            |
| Pan              | 1    | 5     | All         | enum [0-3] + 1     |
| Vibrato type     | 1    | 5     | PU,WAV,KIT  | enum [0-3] + 1     |
| Vibrato direction| 1    | 3     | PU,WAV,KIT  | bit + 1            |
| Env volume       | 1    | 17    | PU,NOI      | high nibble + 1    |
| Env fade         | 1    | 17    | PU,NOI      | low nibble + 1     |
| Length           | 1    | 65    | PU,NOI      | 6-bit + 1          |
| Length limited   | 1    | 3     | PU,NOI      | bit + 1            |
| Sweep            | 1    | 257   | PU,NOI      | byte + 1           |
| Volume           | 1    | 5     | WAV,KIT     | 2-bit + 1          |
| Phase transpose  | 1    | 257   | PU          | byte + 1           |
| Wave             | 1    | 3     | PU          | bit + 1            |
| Phase finetune   | 1    | 17    | PU          | 4-bit + 1          |
| Softsynth        | 13   | mixed | WAV         | inlined softsynth  |
| Repeat           | 1    | 17    | WAV         | low nibble + 1     |
| Play type        | 1    | 5     | WAV         | enum [0-3] + 1     |
| Wave length      | 1    | 17    | WAV         | high nibble + 1    |
| Speed            | 1    | 17    | WAV         | low nibble + 1     |
| Keep attack 1    | 1    | 3     | KIT         | bit + 1            |
| Keep attack 2    | 1    | 3     | KIT         | bit + 1            |
| Kit 1 ID         | 1    | 65    | KIT         | 6-bit + 1          |
| Kit 2 ID         | 1    | 65    | KIT         | 6-bit + 1          |
| Length kit 1     | 1    | 257   | KIT         | byte + 1           |
| Length kit 2     | 1    | 257   | KIT         | byte + 1           |
| Loop kit 1       | 1    | 3     | KIT         | bit + 1            |
| Loop kit 2       | 1    | 3     | KIT         | bit + 1            |
| Offset kit 1     | 1    | 257   | KIT         | byte + 1           |
| Offset kit 2     | 1    | 257   | KIT         | byte + 1           |
| Half-speed       | 1    | 3     | KIT         | bit + 1            |
| Pitch            | 1    | 257   | KIT         | byte + 1           |
| Distortion type  | 1    | 5     | KIT         | enum [0-3] + 1     |

**Total after inlining**: (64, 1662) = 33 scalar fields + 1616 raw table + 13 softsynth.

The `Table` field (originally a scalar 5-bit ID) is replaced with the
full groove-inlined raw table vector (1616 dims). Raw tables preserve
A-command patterns so the model can learn table composition.

The `Softsynth` field (originally a scalar 4-bit ID) is replaced with
the full softsynth parameter vector (13 fields). WAV-only; zeroed for
other instrument types.

### Tables (32 tables x 16 steps)

Two representations, both fully inlined (no entity IDs remain):

- **Raw tables**: original data with A/H commands intact, but groove
  and nested-table slots expanded in-place. Used for entity-level embedding.
- **Traces**: A/H commands resolved into execution traces, groove slots
  expanded. Used as the lookup target when inlining A-command references.

FX command fields (`TABLE_FX_1`, `TABLE_FX_2`) use a **reduced enum**
(8 values: 0 = non-continuous, 1-7 = CMD_D/F/K/L/P/S/T). This
disambiguates the 7 continuous commands that share a single FX value
column, while non-continuous commands are recoverable from sparse FX
value structure.

#### Trace representation (after groove inlining)

| Field           | Shape per table | Notes                        |
|-----------------|----------------|------------------------------|
| Env volume      | (16,)          | high nibble + 1              |
| Env duration    | (16,)          | low nibble + 1               |
| Transpose       | (16,)          | byte + 1                     |
| Reduced FX cmd 1| (16,)          | reduced enum (8 values)      |
| FX value 1      | (16, 48)       | groove slot expanded (see §) |
| Reduced FX cmd 2| (16,)          | reduced enum (8 values)      |
| FX value 2      | (16, 48)       | groove slot expanded (see §) |

**Total per trace**: 5x16 + 2x(16x48) = 80 + 1536 = **1616 features**.

#### Raw table representation (after groove + trace inlining)

Same fields as traces, but FX value columns are wider: the Table FX
slot (col 0) is replaced with the full flattened trace of the target
table (1616 features), and the Groove FX slot is replaced with the
groove vector (32 features). Non-A/G steps get zeros in those slots.

| Field           | Shape per table | Notes                             |
|-----------------|----------------|-----------------------------------|
| Env volume      | (16,)          | high nibble + 1                   |
| Env duration    | (16,)          | low nibble + 1                    |
| Transpose       | (16,)          | byte + 1                          |
| Reduced FX cmd 1| (16,)          | reduced enum (8 values)           |
| FX value 1      | (16, 1663)     | table + groove slots expanded (§) |
| Reduced FX cmd 2| (16,)          | reduced enum (8 values)           |
| FX value 2      | (16, 1663)     | table + groove slots expanded (§) |

---

## 2. Song Sequence

### Sequence layout

The song is flattened to a single sequence ordered by channel:

```
[PU1 steps] [PU2 steps] [WAV steps] [NOI steps]
```

Each channel contributes `NUM_SONG_CHAINS (256) x PHRASES_PER_CHAIN (16)
x STEPS_PER_PHRASE (16) = 65536` steps.

**Total sequence length**: 4 x 65536 = 262144 steps.

### Per-step feature vector

| Feature          | Dims | Vocab | Source / Notes                         |
|------------------|------|-------|----------------------------------------|
| Note             | 1    | 159   | Pitch [0-157] + 1; invalid -> 0        |
| Instrument       | 1662 | mixed | Inlined; table + softsynth expanded    |
| FX values        | 1663 | mixed | Sparse; table + groove slots inlined § |
| Reduced FX cmd   | 1    | 8     | 0=non-continuous, 1-7=D/F/K/L/P/S/T   |
| Chain transpose  | 1    | 257   | Per-phrase, broadcast across steps     |

**Raw feature count per step**: 1 + 1662 + 1663 + 1 + 1 = **3328 scalar tokens**.

The reduced FX command disambiguates the 7 continuous commands
(D/F/K/L/P/S/T) that share a single FX value column. Non-continuous
commands are recoverable from which FX value group is non-zero.

### FX Values breakdown (1663 features) §

Sparse vector — at most one group is non-zero per step. The active
command is largely implicit (sparse FX values encode it), except for
continuous commands which require the reduced FX cmd field. All scalar
values use +1 null offset. Column order derives from `FX_VALUE_KEYS`
with the Table FX and Groove FX slots expanded.

| Cols       | Key             | Dims | Active when | Notes                     |
|------------|-----------------|------|-------------|---------------------------|
| 0-1615     | Table FX        | 1616 | CMD_A       | full table trace (§§)     |
| 1616-1647  | Groove FX       | 32   | CMD_G       | full groove vector (§§§)  |
| 1648       | Hop FX          | 1    | CMD_H       | byte + 1                  |
| 1649       | Pan FX          | 1    | CMD_O       | enum [0-3] + 1            |
| 1650-1651  | Chord FX        | 2    | CMD_C       | nibble-split + 1          |
| 1652-1653  | Env FX          | 2    | CMD_E       | nibble-split + 1          |
| 1654-1655  | Retrig FX       | 2    | CMD_R       | nibble-split + 1          |
| 1656-1657  | Vibrato FX      | 2    | CMD_V       | nibble-split + 1          |
| 1658       | Volume FX       | 1    | CMD_M       | byte + 1                  |
| 1659       | Wave FX         | 1    | CMD_W       | enum [0-3] + 1            |
| 1660-1661  | Random FX       | 2    | CMD_Z       | nibble-split + 1          |
| 1662       | Continuous FX   | 1    | CMD_D/F/K/L/P/S/T | byte + 1           |

**No scalar entity IDs remain** in the final song tokens. All entity
references are replaced with the full data of the referenced entity.

**§§ Table inlining**: The Table FX slot (originally a scalar table ID)
is replaced with the flattened **trace** of the target table (1616
features). Traces are used rather than raw tables because A/H commands
are already resolved, avoiding recursive expansion. Steps without
CMD_A get zeros.

**§§§ Groove inlining**: The Groove FX slot (originally a scalar groove
ID) is replaced with the full groove vector: `grooves[id]` flattened
from (STEPS_PER_GROOVE, 2) to 32 scalars. Steps without CMD_G get zeros.

### Embedding Notes

All entity references in the song sequence are **inlined** — the
tokenizer replaces every entity ID with the full data of the referenced
entity. The final song tokens contain no scalar entity IDs.

Entity generation strategies (how the model produces the entities that
get inlined) remain architecture-dependent. Possible strategies include:

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
Song sequence (per step, all entities inlined):
  note ──────────────→ Embedding(159)     ──┐
  chain_transpose ───→ Embedding(257)     ──┤
  reduced_fx_cmd ────→ Embedding(8)       ──┤
  fx_values (1663) ──→ FXValueEmbedder    ──┤──→ concat → Linear → d_model
  instrument (1662) ─→ Linear             ──┘
```
