from pe_lsdj.constants import *
from pe_lsdj.embedding import (
    ConcatEmbedder, 
    GatedNormedEmbedder, 
    GrooveEntityEmbedder,
    InstrumentEntityEmbedder, 
    FXEmbedder,
    PhraseFXEmbedder,
    SoftsynthEntityEmbedder,
    WaveFrameEntityEmbedder,
    # TableFXEmbedder,
    TableFXValueEmbedder,
    TableEmbedder,
    TableEntityEmbedder,
)
from jaxtyping import Array, Key
import jax.random as jr


"""
Sketch on overall dimensions:

Song step - [Note, Instrument, FX, Transpose]

Instrument - [ misc., Table, Softsynth, Waveframes ]
    Target: 128

    misc: 64
    table: 64
    softsynth: 64
    waveframes: 32

    224 -> linear -> 128

FX - [ misc., Table, Groove ]
    Target: 128

    misc: 128
    table: 64
    groove: 32

    224 -> linear -> 128

    
Target for d_model (song step): 512

Note: 128
Instrument: 128
FX: 128
Transpose: 16

sum: 400 PER CHANNEL * NUM_CHANNELS = 1600
(concat -> proj)? ConcatEmbedder -> 512

"""


class ChannelStepEmbedder(ConcatEmbedder):
    def __init__(
        self, 
        key: Key, 
        out_dim: int,
        note_embedder: GatedNormedEmbedder,
        instrument_embedder: InstrumentEntityEmbedder,
        fx_embedder: FXEmbedder,
        transpose_embedder: GatedNormedEmbedder,
    ):
        
        super().__init__(
            key, 
            {
                "note_embedder": note_embedder,
                "instrument_embedder": instrument_embedder,
                "fx_embedder": fx_embedder,
                "transpose_embedder": transpose_embedder
            },
            out_dim
        )


class SongStepEmbedder(ConcatEmbedder):
    """
    Embedder for one step of an LSDJ track:
    [Pulse 1 | Pulse 2 | Wav | Noise]
        |       
        |__> | Note | Instrument | FX | Traspose | (per channel)
                          |        |
                         ...      ...

    Channels share feature encoders, but each have their own
    learned linear projection into the final step embedding.
    """
    def __init__(
        self, 
        key: Key, 
        instruments: Array,
        softsynths: Array,
        waveframes: Array,
        grooves: Array,
        tables: Array,
        out_dim: int = 512,
        PU1_dim: int = 400,
        PU2_dim: int = 400,
        WAV_dim: int = 400,
        NOI_dim: int = 400,
        note_dim: int = 128,
        instr_dim: int = 128,
        fx_dim: int = 128,
        table_dim: int = 64,
        transpose_dim: int = 16,
        groove_dim: int = 32,
        softsynth_dim: int = 64,
        waveframe_dim: int = 32,
    ):
        keys = jr.split(key, 15)

        note_embedder = GatedNormedEmbedder(
            note_dim,
            keys[0],
            max_value=NUM_NOTES,
        )

        groove_embedder = GrooveEntityEmbedder(
            groove_dim,
            keys[1],
            grooves,
        )

        table_fx_value_embedder = TableFXValueEmbedder(
            64, # What should this be? Make top-level param
            keys[2],
            groove_embedder,
        )

        table_fx_embedder = FXEmbedder(
            keys[3],
            table_fx_value_embedder,
            fx_dim,
        )

        table_embedder = TableEntityEmbedder(
            table_dim,
            keys[4],
            tables,
            table_fx_embedder=table_fx_embedder,
        )

        softsynth_embedder = SoftsynthEntityEmbedder(
            keys[5],
            softsynths,
            softsynth_dim,
        )

        waveframe_embedder = WaveFrameEntityEmbedder(
            keys[6],
            waveframes,
            waveframe_dim,
        )

        instr_embedder = InstrumentEntityEmbedder(
            keys[7], 
            instruments, 
            table_embedder, 
            softsynth_embedder, 
            waveframe_embedder,
            instr_dim,
        )

        phrase_fx_embedder = PhraseFXEmbedder(
            keys[8],
            table_fx_value_embedder,
            table_embedder,
            fx_dim,
        )

        transpose_embedder = GatedNormedEmbedder(
            transpose_dim,
            keys[9],
        )

        PU1_embedder = ChannelStepEmbedder(
            keys[10],
            PU1_dim,
            note_embedder,
            instr_embedder,
            phrase_fx_embedder,
            transpose_embedder,
        )

        PU2_embedder = ChannelStepEmbedder(
            keys[11],
            PU2_dim,
            note_embedder,
            instr_embedder,
            phrase_fx_embedder,
            transpose_embedder,
        )

        WAV_embedder = ChannelStepEmbedder(
            keys[12],
            WAV_dim,
            note_embedder,
            instr_embedder,
            phrase_fx_embedder,
            transpose_embedder,
        )

        NOI_embedder = ChannelStepEmbedder(
            keys[13],
            NOI_dim,
            note_embedder,
            instr_embedder,
            phrase_fx_embedder,
            transpose_embedder,
        )

        embedders = {
            "PU1_embedder": PU1_embedder,
            "PU2_embedder": PU2_embedder,
            "WAV_embedder": WAV_embedder,
            "NOI_embedder": NOI_embedder,
        }

        super().__init__(keys[14], embedders, out_dim)
