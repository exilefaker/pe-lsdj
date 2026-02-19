import numpy as np
import pytest
from pe_lsdj.tokenizer.songfile import SongFile
from pe_lsdj.constants import *


# --- Song round-trip ---

def test_repack_song_tokens_round_trip(song_file):
    """song_tokens should survive the repack → reload round-trip."""
    raw_bytes = song_file.repack()
    sf2 = SongFile(raw_bytes=raw_bytes, name=song_file.name)
    np.testing.assert_array_equal(
        np.array(song_file.song_tokens),
        np.array(sf2.song_tokens),
    )

def test_repack_song_grooves_round_trip(song_file):
    """Grooves should survive the round-trip."""
    raw_bytes = song_file.repack()
    sf2 = SongFile(raw_bytes=raw_bytes, name=song_file.name)
    np.testing.assert_array_equal(
        np.array(song_file.grooves),
        np.array(sf2.grooves),
    )


def test_repack_song_instruments_round_trip(song_file):
    """Instrument tensors should survive the round-trip."""
    raw_bytes = song_file.repack()
    sf2 = SongFile(raw_bytes=raw_bytes, name=song_file.name)
    for key in song_file.instruments:
        np.testing.assert_array_equal(
            np.array(song_file.instruments[key]),
            np.array(sf2.instruments[key]),
            err_msg=f"Instrument field '{key}' mismatch",
        )


def test_repack_song_softsynths_round_trip(song_file):
    """Softsynth tensors should survive the round-trip."""
    raw_bytes = song_file.repack()
    sf2 = SongFile(raw_bytes=raw_bytes, name=song_file.name)
    for key in song_file.softsynths:
        np.testing.assert_array_equal(
            np.array(song_file.softsynths[key]),
            np.array(sf2.softsynths[key]),
            err_msg=f"Softsynth field '{key}' mismatch",
        )


def test_repack_song_tables_round_trip(song_file):
    """Table tensors should survive the round-trip."""
    raw_bytes = song_file.repack()
    sf2 = SongFile(raw_bytes=raw_bytes, name=song_file.name)
    for key in song_file.tables:
        np.testing.assert_array_equal(
            np.array(song_file.tables[key]),
            np.array(sf2.tables[key]),
            err_msg=f"Table field '{key}' mismatch",
        )


def test_repack_song_tempo_round_trip(song_file):
    """Tempo should survive the round-trip."""
    raw_bytes = song_file.repack()
    sf2 = SongFile(raw_bytes=raw_bytes, name=song_file.name)
    assert int(song_file.tempo) == int(sf2.tempo)


# ---- Array property shortcuts ----

def test_instruments_array(song_file):
    instruments_arr = song_file.instruments_array
    assert instruments_arr.shape == (NUM_INSTRUMENTS, INSTR_WIDTH)

def test_tables_array(song_file):
    tables_arr = song_file.tables_array
    assert tables_arr.shape == (NUM_TABLES, STEPS_PER_TABLE * TABLE_WIDTH)

def test_softsynths_array(song_file):
    softsynths_arr = song_file.softsynths_array
    assert softsynths_arr.shape == (NUM_SYNTHS, SOFTSYNTH_WIDTH)


# ---- Set max phrases for repack ----

def test_repack_song_tokens_custom_max_phrases(song_file):
    raw_bytes = song_file.repack(max_phrases_per_chain=4)
    chain_phrases = (
        np.array(raw_bytes)[CHAIN_PHRASES_ADDR]
    ).reshape((NUM_CHAINS, PHRASES_PER_CHAIN))

    # Get first chain
    chain = chain_phrases[0]
    assert not np.any(chain[:4] == EMPTY)
    assert np.all(chain[4:] == EMPTY)
