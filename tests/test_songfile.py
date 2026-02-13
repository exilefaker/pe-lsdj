import numpy as np
import pytest
from pe_lsdj.detokenizer import repack_song, _recover_fx_commands
from pe_lsdj.songfile import SongFile
from pe_lsdj.constants import *


# --- FX command recovery ---

def test_recover_fx_commands_continuous():
    """Continuous commands should round-trip through reduced → recovered."""
    reduced = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.uint8)
    fx_vals = np.zeros((8, 17), dtype=np.uint8)
    recovered = _recover_fx_commands(reduced, fx_vals)
    expected = np.array([
        0, CMD_D, CMD_F, CMD_K, CMD_L, CMD_P, CMD_S, CMD_T
    ], dtype=np.uint8)
    np.testing.assert_array_equal(recovered, expected)


def test_recover_fx_commands_non_continuous():
    """Non-continuous commands should be inferred from sparse FX values."""
    reduced = np.zeros(5, dtype=np.uint8)
    fx_vals = np.zeros((5, 17), dtype=np.uint8)
    # CMD_A: col 0 (TABLE_FX)
    fx_vals[0, 0] = 5
    # CMD_G: col 1 (GROOVE_FX)
    fx_vals[1, 1] = 3
    # CMD_C: col 4 (CHORD_FX_1)
    fx_vals[2, 4] = 7
    # CMD_E: col 6 (ENV_FX_VOL)
    fx_vals[3, 6] = 10
    # CMD_H: col 2 (HOP_FX)
    fx_vals[4, 2] = 1

    recovered = _recover_fx_commands(reduced, fx_vals)
    expected = np.array([CMD_A, CMD_G, CMD_C, CMD_E, CMD_H], dtype=np.uint8)
    np.testing.assert_array_equal(recovered, expected)


# --- Song round-trip ---

def test_repack_song_tokens_round_trip(song_file):
    """song_tokens should survive the repack → reload round-trip."""
    raw_bytes = repack_song(song_file)
    sf2 = SongFile(raw_bytes=raw_bytes, name=song_file.name)
    np.testing.assert_array_equal(
        np.array(song_file.song_tokens),
        np.array(sf2.song_tokens),
    )


def test_repack_song_grooves_round_trip(song_file):
    """Grooves should survive the round-trip."""
    raw_bytes = repack_song(song_file)
    sf2 = SongFile(raw_bytes=raw_bytes, name=song_file.name)
    np.testing.assert_array_equal(
        np.array(song_file.grooves),
        np.array(sf2.grooves),
    )


def test_repack_song_instruments_round_trip(song_file):
    """Instrument tensors should survive the round-trip."""
    raw_bytes = repack_song(song_file)
    sf2 = SongFile(raw_bytes=raw_bytes, name=song_file.name)
    for key in song_file.instruments:
        np.testing.assert_array_equal(
            np.array(song_file.instruments[key]),
            np.array(sf2.instruments[key]),
            err_msg=f"Instrument field '{key}' mismatch",
        )


def test_repack_song_softsynths_round_trip(song_file):
    """Softsynth tensors should survive the round-trip."""
    raw_bytes = repack_song(song_file)
    sf2 = SongFile(raw_bytes=raw_bytes, name=song_file.name)
    for key in song_file.softsynths:
        np.testing.assert_array_equal(
            np.array(song_file.softsynths[key]),
            np.array(sf2.softsynths[key]),
            err_msg=f"Softsynth field '{key}' mismatch",
        )


def test_repack_song_tables_round_trip(song_file):
    """Table tensors should survive the round-trip."""
    raw_bytes = repack_song(song_file)
    sf2 = SongFile(raw_bytes=raw_bytes, name=song_file.name)
    for key in song_file.tables:
        np.testing.assert_array_equal(
            np.array(song_file.tables[key]),
            np.array(sf2.tables[key]),
            err_msg=f"Table field '{key}' mismatch",
        )


def test_repack_song_tempo_round_trip(song_file):
    """Tempo should survive the round-trip."""
    raw_bytes = repack_song(song_file)
    sf2 = SongFile(raw_bytes=raw_bytes, name=song_file.name)
    assert int(song_file.tempo) == int(sf2.tempo)
