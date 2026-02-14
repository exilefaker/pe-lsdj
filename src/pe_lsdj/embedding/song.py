from pe_lsdj.constants import *
from pe_lsdj.embedding.fx import FXEmbedder
from pe_lsdj.embedding 
import jax.numpy as jnp
import equinox as eqx


"""
Begin to think about overall dimensions:

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

# TODO: Overall song encoder