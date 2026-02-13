__title__ = 'pylsdj'
__version__ = '2.3.3'
__author__ = 'Alex Rasmussen'
__license__ = 'MIT'
__copyright__ = 'Copyright 2015 Alex Rasmussen'

from . import bread_spec
from . import chain
from . import clock
from . import consts
from . import filepack
from .instrument import Instrument
from .wave_instrument import WaveInstrument
from .pulse_instrument import PulseInstrument
from .kit_instrument import KitInstrument
from .noise_instrument import NoiseInstrument
from .phrase import Phrase
from .project import load_lsdsng, load_srm, Project
from .savfile import SAVFile
from .song import Song, Sequence, Instruments
from .speech_instrument import Word, SpeechInstrument
from .synth import Synth, SynthSoundParams
from .table import Table, TableFX
from .vibrato import Vibrato
from . import exceptions
from . import utils

# Alex: Add constants
from .bread_spec import (
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
    WORD_LENGTH,    
)