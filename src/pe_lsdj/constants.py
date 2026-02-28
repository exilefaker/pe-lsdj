"""
Constants:

- Shapes / enum sizes
- Memory addresses
- Channel names
- Token dict keys / feature names
- misc.
"""

# Shapes (import from `pylsdj`)
from pylsdj import (
    FRAMES_PER_WAVE,
    NOTES,
    NUM_CHAINS,
    NUM_GROOVES, 
    NUM_INSTRUMENTS,
    NUM_SONG_CHAINS,
    NUM_SYNTHS,
    NUM_TABLES,
    NUM_PHRASES, 
    NUM_WORDS,
    PHRASES_PER_CHAIN,
    STEPS_PER_GROOVE, 
    STEPS_PER_TABLE, 
    STEPS_PER_PHRASE, 
    WAVES_PER_SYNTH,
    WORD_LENGTH,   
)

NUM_NOTES = len(NOTES)

NUM_SONG_PHRASES = NUM_SONG_CHAINS * PHRASES_PER_CHAIN
NUM_SONG_STEPS = NUM_SONG_PHRASES * STEPS_PER_PHRASE

NUM_PHRASE_STEPS = NUM_PHRASES * STEPS_PER_PHRASE

# MEMORY OFFSETS (cf. LSDJ manual v8.3.1+)
# I use LSDJ v3.9.2, which nearly matches the memory map shown here:
# https://littlesounddj.fandom.com/wiki/.sav_structure (v6.7.0)

# Bank 0:
# 4080 0000-0FEF: phrases-> notes
PHRASE_NOTES_ADDR = slice(0x0000,0x0FF0)
# Layout: [ step 0 phrase 0, step 1 phrase 0, ... step M phrase N ]
# where phrases are ordered by hex index (not order of occurrence in song!)
#   64 0FF0-102F: bookmarks
BOOKMARKS_ADDR = slice(0x0FF0,0x1030)
# (TBH I never use this)
#   96 1030-108F: empty
#  512 1090-128F: grooves
GROOVES_ADDR = slice(0x1090,0x1290)
# 1024 1290-168F: song-> chainno (listed PU1,PU2,WAV,NOI for $00 .. $ff in order)
# List of chain indices that make up the song structure. Important!
# Listed as [PU1, PU2, WAV, NOI] per step 00...FF
SONG_CHAINS_ADDR = slice(0x1290,0x1690)
#  512 1690-188F: tables-> envelope
TABLE_ENVELOPES_ADDR = slice(0x1690,0x1890)
# 1344 1890-1DCF: instrument->speech->words ($20*42)
WORDS_ADDR = slice(0x1890,0x1DD0)
#  168 1DD0-1E77: instr->speech->wordnames
WORD_NAMES_ADDR = slice(0x1DD0,0x1E78)
#    2 1E78-1E79: mem initialized flag (set to “rb” on init)
MEM_INIT_FLAG_ADDR = slice(0x1E78,0x1E7A)
#  320 1E7A-1FB9: instr->names
INSTR_NAMES_ADDR = slice(0x1E7a,0x1FBA)
#   70 1FBA-1FFF: empty

# Bank 1:
#   32 2000-201F: empty
#   32 2020-203F: table allocation table (byte per table, 1 if alloced, 0 else)
TABLE_ALLOC_TABLE_ADDR = slice(0x2020,0x2040)
#   64 2040-207F: instr alloc table (byte per instrument, 1 if alloced, 0 else)
INSTR_ALLOC_TABLE_ADDR = slice(0x2040,0x2080)
# 2048 2080-287F: chains-> phraseno
CHAIN_PHRASES_ADDR = slice(0x2080,0x2880)
# 2048 2880-307F: chains-> transposes
CHAIN_TRANSPOSES_ADDR = slice(0x2880,0x3080)
# 1024 3080-347F: instr->param
INSTRUMENTS_ADDR = slice(0x3080,0x3480)
#  512 3480-367F: tables-> transpose
TABLE_TRANSPOSES_ADDR = slice(0x3480,0x3680)
#  512 3680-387F: tables-> fx
TABLE_FX_ADDR = slice(0x3680,0x3880)
#  (FX enum, value) - 1 byte each
#  I believe pylsdj first interleaves the FX and FX-val per command, 
#  in the decompression stage?
#  512 3880-3A7F: tables-> fx val
TABLE_FX_VAL_ADDR = slice(0x3880,0x3A80)
#  512 3A80-3C7F: tables-> fx 2
TABLE_FX_2_ADDR = slice(0x3A80,0x3C80)
#  512 3C80-3E7F: tables-> fx 2 val
TABLE_FX_2_VAL_ADDR = slice(0x3C80,0x3E80)
#    2 3E80-3E81: mem initialized flag (set to “rb” on init)
MEM_INIT_FLAG2_ADDR = slice(0x3E80,0x3E82)
#   32 3E82-3EA1: phrase allocation table (bit per phrase, 1 if alloced, 0 else)
# 255 phrases; technically I think the last bit is padding
PHRASE_ALLOC_TABLE_ADDR = slice(0x3E82,0x3EA2)
#   16 3EA2-3EB1: chain allocation table (bit per chain, 1 if alloced, 0 else)
CHAIN_ALLOC_TABLE_ADDR = slice(0x3EA2,0x3EB2)
#  256 3EB2-3FB1: softsynth params (16 parameter blocks of 16 bytes each)
SOFTSYNTH_PARAMS_ADDR = slice(0x3EB2,0x3FB2)
#    1 3FB2-3FB2: clock, hours
CLOCK_HOURS_LEN = 1 # Ignore for now
#    1 3FB3-3FB3: clock, minutes
CLOCK_MINS_LEN = 1 # Ignore for now
#    1 3FB4-3FB4: tempo
TEMPO_ADDR = slice(0x3FB4,0x3FB5)
# Ignore most of these for now...
#    1 3FB5-3FB5: tune setting
TUNE_SETTING_ADDR = slice(0x3FB5,0x3FB6)
#    1 3FB6-3FB6: total clock, days
#    1 3FB7-3FB7: total clock, hours
#    1 3FB8-3FB8: total clock, minutes
#    1 3FB9-3FB9: total clock, checksum (days+hours+minutes)
#    1 3FBA-3FBA: key delay
KEY_DELAY_ADDR = slice(0x3FBA,0x3FBB)
#    1 3FBB-3FBB: key repeat
KEY_REPEAT_ADDR = slice(0x3FBB,0x3FBC)
#    1 3FBC-3FBC: font
#    1 3FBD-3FBD: sync setting
#    1 3FBE-3FBE: colorset
#    1 3FBF-3FBF: empty
#    1 3FC0-3FC0: clone (0=deep, 1=slim)
#    1 3FC1-3FC1: file changed?
#    1 3FC2-3FC2: power save
#    1 3FC3-3FC3: prelisten
#    2 3FC4-3FC5: wave synth overwrite locks
#   58 3FC6-3FFF: empty

# Summarize these as we mainly want to copy over the defaults
SETTINGS_ADDR = slice(0x3FB5,0x4000)

# Bank 2:
# 4080 4000-4FEF: phrases->fx
PHRASE_FX_ADDR = slice(0x4000,0x4FF0)
# 4080 4FF0-5FDF: phrases->fx val
PHRASE_FX_VAL_ADDR = slice(0x4FF0,0x5FE0)
#   32 5FE0-5FFF: empty

# Bank 3:
# 4096 6000-6FFF: 256 wave frames (each frame has 32 4-bit samples)
WAVE_FRAMES_ADDR = slice(0x6000,0x7000)
# 4080 7000-7FEF: phrases->instr
PHRASE_INSTR_ADDR = slice(0x7000,0x7FF0)
#    2 7FF0-7FF1: mem initialized flag (set to “rb” on init)
MEM_INIT_FLAG3_ADDR = slice(0x7FF0,0x7FF2)
#   13 7FF2-7FFE: empty
#    1 7FFF-7FFF: version byte
VERSION_ADDR = slice(0x7FFF,0x8000)

# Channel and feature names

PU = 1
WAV = 2
KIT = 3
NOI = 4

CMD_NULL = 0
CMD_A = 1
CMD_C = 2
CMD_D = 3
CMD_E = 4
CMD_F = 5
CMD_G = 6
CMD_H = 7
CMD_K = 8
CMD_L = 9
CMD_M = 10
CMD_O = 11
CMD_P = 12
CMD_R = 13
CMD_S = 14
CMD_T = 15
CMD_V = 16
CMD_W = 17
CMD_Z = 18

TABLE_FX = "Table FX"
GROOVE_FX = "Groove FX"
HOP_FX = "Hop FX"
PAN_FX = "Pan FX"
CHORD_FX_1 = "Chord FX 1"
CHORD_FX_2 = "Chord FX 2"
ENV_FX_VOL = "Env FX Vol"
ENV_FX_FADE = "Env FX Fade"
RETRIG_FX_FADE = "Retrig FX Fade"
RETRIG_FX_RATE = "Retrig FX Rate"
VIBRATO_FX_SPEED = "Vibrato FX Speed"
VIBRATO_FX_DEPTH = "Vibrato FX Depth"
VOLUME_FX = "Volume FX"
WAVE_FX = "Wave FX"
RANDOM_FX_L = "Random FX L"
RANDOM_FX_R = "Random FX R"
CONTINUOUS_FX = "Continuous FX"

# Main song fields
PHRASE_NOTES = "Phrase notes"
GROOVES = "Grooves"

# Instrument fields
TYPE_ID = "Type ID"
ENV_VOLUME = "Env volume"
ENV_FADE = "Env fade"
VOLUME = "Volume"
PHASE_TRANSPOSE = "Phase transpose"
SOFTSYNTH_ID = "Softsynth ID"
REPEAT = "Repeat"
KEEP_ATTACK_1 = "Keep attack 1"
HALF_SPEED = "Half-speed"
KIT_1_ID = "Kit 1 ID"
LENGTH = "Length"
LENGTH_LIMITED = "Length limited"
LENGTH_KIT_1 = "Length kit 1"
SWEEP = "Sweep"
TABLE_AUTOMATE = "Table automate"
AUTOMATE_2 = "Automate 2"
VIBRATO_TYPE = "Vibrato type"
VIBRATO_DIRECTION = "Vibrato direction"
LOOP_KIT_1 = "Loop kit 1"
LOOP_KIT_2 = "Loop kit 2"
TABLE = "Table"
TABLE_ON_OFF = "Table on/off"
WAVE = "Wave"
PHASE_FINETUNE = "Phase finetune"
PAN = "Pan"
PITCH = "Pitch"
PLAY_TYPE = "Play type"
KEEP_ATTACK_2 = "Keep attack 2"
KIT_2_ID = "Kit 2 ID"
WAVE_LENGTH = "Wave length"
SPEED = "Speed"
DISTORTION_TYPE = "Distortion type"
LENGTH_KIT_2 = "Length kit 2"
OFFSET_KIT_1 = "Offset kit 1"
OFFSET_KIT_2 = "Offset kit 2"

# Table tokenizer fields
TABLE_ENV_VOLUME = "Table env volume"
TABLE_ENV_DURATION = "Table env duration"
TABLE_TRANSPOSE = "Table transpose"
TABLE_FX_1 = "Table FX 1"
TABLE_FX_VALUE_1 = "Table FX value 1"
TABLE_FX_2 = "Table FX 2"
TABLE_FX_VALUE_2 = "Table FX value 2"

# Softsynth fields
SOFTSYNTH_WAVEFORM = "Softsynth waveform"
SOFTSYNTH_FILTER_TYPE = "Softsynth filter type"
SOFTSYNTH_FILTER_RESONANCE = "Softsynth filter resonance"
SOFTSYNTH_DISTORTION = "Softsynth distortion"
SOFTSYNTH_PHASE_TYPE = "Softsynth phase type"
SOFTSYNTH_START_VOLUME = "Softsynth start volume"
SOFTSYNTH_START_FILTER_CUTOFF = "Softsynth start filter cutoff"
SOFTSYNTH_START_PHASE_AMOUNT = "Softsynth start phase amount"
SOFTSYNTH_START_VERTICAL_SHIFT = "Softsynth start vertical shift"
SOFTSYNTH_END_VOLUME = "Softsynth end volume"
SOFTSYNTH_END_FILTER_CUTOFF = "Softsynth end filter cutoff"
SOFTSYNTH_END_PHASE_AMOUNT = "Softsynth end phase amount"
SOFTSYNTH_END_VERTICAL_SHIFT = "Softsynth end vertical shift"

# Model wave frames along with instruments
WAVEFRAME_ID = "Wave frame ID"
WAVEFRAME_DIM = WAVES_PER_SYNTH * FRAMES_PER_WAVE

# Utilities / misc
INSTRUMENT_SIZE = 0x10
SYNTH_SIZE = 0x10
GROOVE_SIZE = 0x1F
TABLE_SIZE = 0xBF
FX_VALUES_FEATURE_DIM = 17  # len(FX_VALUE_KEYS)

EMPTY = 255

NUM_CHANNELS = 4

# Canonical column order for stacked FX value arrays.
# Must match the dict key order in parse_fx_values.
FX_VALUE_KEYS = (
    TABLE_FX, GROOVE_FX, HOP_FX, PAN_FX,
    CHORD_FX_1, CHORD_FX_2, ENV_FX_VOL, ENV_FX_FADE,
    RETRIG_FX_FADE, RETRIG_FX_RATE, VIBRATO_FX_SPEED, VIBRATO_FX_DEPTH,
    VOLUME_FX, WAVE_FX, RANDOM_FX_L, RANDOM_FX_R,
    CONTINUOUS_FX,
)


# Token dimensions (input to embedding layer)

INSTR_WIDTH = 35 # +1 within embedding, for waveframes
TABLE_STEP_WIDTH = 39
TABLE_WIDTH = STEPS_PER_TABLE * TABLE_STEP_WIDTH
SOFTSYNTH_WIDTH = 13
