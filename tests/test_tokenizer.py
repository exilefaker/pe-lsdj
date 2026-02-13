from pe_lsdj.tokenizer import parse_instruments, parse_notes, parse_softsynths, parse_fx_commands, parse_fx_values, parse_tables, parse_grooves
from pe_lsdj.detokenizer import repack_instruments, repack_notes, repack_softsynths, repack_fx_values, repack_tables, repack_grooves
from pe_lsdj.constants import (
    INSTRUMENTS_ADDR, PHRASE_NOTES_ADDR, PHRASE_NOTES, SOFTSYNTH_PARAMS_ADDR,
    PHRASE_FX_ADDR, PHRASE_FX_VAL_ADDR, GROOVES_ADDR,
    TABLE_ENVELOPES_ADDR, TABLE_TRANSPOSES_ADDR,
    TABLE_FX_ADDR, TABLE_FX_VAL_ADDR, TABLE_FX_2_ADDR, TABLE_FX_2_VAL_ADDR,
)
import jax.numpy as jnp
import pytest
from pytest_lazyfixture import lazy_fixture


@pytest.mark.parametrize(
    "raw_bytes",
    [
        lazy_fixture("tohou_bytes"),
        lazy_fixture("crshhh_bytes"),
        lazy_fixture("ofd_bytes"),
        lazy_fixture("equus_bytes"),
        lazy_fixture("organelle_bytes"),
    ]
)
def test_instrument_tokenizer_round_trip(raw_bytes):
    raw_bytes_in = raw_bytes[INSTRUMENTS_ADDR]
    print("Parsing raw bytes...")
    tokens_step_1 = parse_instruments(jnp.array(raw_bytes_in))

    print("Compiling tokens back to bytes...")
    repacked_bytes = repack_instruments(tokens_step_1)

    print("Parsing recovered bytes...")
    tokens_step_2 = parse_instruments(jnp.array(repacked_bytes))

    print("Comparing tokens...")
    for key in tokens_step_1:
        diff = tokens_step_1[key] != tokens_step_2[key]
        assert not jnp.any(diff), f"FAIL: {key} mismatch! Indices: {jnp.where(diff)}"
        print(f"PASS: {key}")

@pytest.mark.parametrize(
    "raw_bytes",
    [
        lazy_fixture("tohou_bytes"),
        lazy_fixture("crshhh_bytes"),
        lazy_fixture("ofd_bytes"),
        lazy_fixture("equus_bytes"),
        lazy_fixture("organelle_bytes"),
    ]
)
def test_notes_tokenizer_round_trip(raw_bytes):
    raw_bytes_in = raw_bytes[PHRASE_NOTES_ADDR]
    print("Parsing raw bytes...")
    tokens_step_1 = parse_notes(jnp.array(raw_bytes_in))

    print("Compiling tokens back to bytes...")
    repacked_bytes = repack_notes({PHRASE_NOTES: tokens_step_1})

    print("Parsing recovered bytes...")
    tokens_step_2 = parse_notes(jnp.array(repacked_bytes))

    print("Comparing tokens...")
    diff = tokens_step_1 != tokens_step_2
    assert not jnp.any(diff), f"FAIL on indices: {jnp.where(diff)}"
    print(f"PASS!")


@pytest.mark.parametrize(
    "raw_bytes",
    [
        lazy_fixture("tohou_bytes"),
        lazy_fixture("crshhh_bytes"),
        lazy_fixture("ofd_bytes"),
        lazy_fixture("equus_bytes"),
        lazy_fixture("organelle_bytes"),
    ]
)
def test_softsynth_tokenizer_round_trip(raw_bytes):
    raw_bytes_in = raw_bytes[SOFTSYNTH_PARAMS_ADDR]
    print("Parsing raw bytes...")
    tokens_step_1 = parse_softsynths(jnp.array(raw_bytes_in))

    print("Compiling tokens back to bytes...")
    repacked_bytes = repack_softsynths(tokens_step_1)

    print("Parsing recovered bytes...")
    tokens_step_2 = parse_softsynths(jnp.array(repacked_bytes))

    print("Comparing tokens...")
    for key in tokens_step_1:
        diff = tokens_step_1[key] != tokens_step_2[key]
        assert not jnp.any(diff), f"FAIL: {key} mismatch! Indices: {jnp.where(diff)}"
        print(f"PASS: {key}")


@pytest.mark.parametrize(
    "raw_bytes",
    [
        lazy_fixture("tohou_bytes"),
        lazy_fixture("crshhh_bytes"),
        lazy_fixture("ofd_bytes"),
        lazy_fixture("equus_bytes"),
        lazy_fixture("organelle_bytes"),
    ]
)
def test_fx_values_tokenizer_round_trip(raw_bytes):
    raw_fx_cmd_bytes = raw_bytes[PHRASE_FX_ADDR]
    raw_fx_val_bytes = raw_bytes[PHRASE_FX_VAL_ADDR]

    print("Parsing FX commands...")
    fx_command_IDs = parse_fx_commands(jnp.array(raw_fx_cmd_bytes))

    print("Parsing FX values...")
    tokens_step_1 = parse_fx_values(jnp.array(raw_fx_val_bytes), fx_command_IDs)

    print("Compiling tokens back to bytes...")
    repacked_bytes = repack_fx_values(tokens_step_1, fx_command_IDs)

    print("Parsing recovered bytes...")
    tokens_step_2 = parse_fx_values(jnp.array(repacked_bytes), fx_command_IDs)

    print("Comparing tokens...")
    for key in tokens_step_1:
        diff = tokens_step_1[key] != tokens_step_2[key]
        assert not jnp.any(diff), f"FAIL: {key} mismatch! Indices: {jnp.where(diff)}"
        print(f"PASS: {key}")


@pytest.mark.parametrize(
    "raw_bytes",
    [
        lazy_fixture("tohou_bytes"),
        lazy_fixture("crshhh_bytes"),
        lazy_fixture("ofd_bytes"),
        lazy_fixture("equus_bytes"),
        lazy_fixture("organelle_bytes"),
    ]
)
def test_groove_tokenizer_round_trip(raw_bytes):
    raw_bytes_in = jnp.array(raw_bytes[GROOVES_ADDR], dtype=jnp.uint8)

    print("Parsing grooves...")
    tokens_step_1 = parse_grooves(raw_bytes_in)

    print("Compiling tokens back to bytes...")
    repacked_bytes = repack_grooves(tokens_step_1)

    print("Parsing recovered bytes...")
    tokens_step_2 = parse_grooves(jnp.array(repacked_bytes, dtype=jnp.uint8))

    print("Comparing tokens...")
    diff = tokens_step_1 != tokens_step_2
    assert not jnp.any(diff), f"FAIL on indices: {jnp.where(diff)}"
    print("PASS!")


TABLE_REGION_MAP = {
    "envelopes": TABLE_ENVELOPES_ADDR,
    "transposes": TABLE_TRANSPOSES_ADDR,
    "fx_cmd_1": TABLE_FX_ADDR,
    "fx_val_1": TABLE_FX_VAL_ADDR,
    "fx_cmd_2": TABLE_FX_2_ADDR,
    "fx_val_2": TABLE_FX_2_VAL_ADDR,
}

@pytest.mark.parametrize(
    "raw_bytes",
    [
        lazy_fixture("tohou_bytes"),
        lazy_fixture("crshhh_bytes"),
        lazy_fixture("ofd_bytes"),
        lazy_fixture("equus_bytes"),
        lazy_fixture("organelle_bytes"),
    ]
)
def test_table_tokenizer_round_trip(raw_bytes):
    raw_data = jnp.array(raw_bytes, dtype=jnp.uint8)

    print("Parsing tables...")
    tokens_step_1 = parse_tables(raw_data)

    print("Compiling tokens back to bytes...")
    repacked = repack_tables(tokens_step_1)

    print("Reassembling raw data...")
    raw_data_2 = raw_data
    for name, addr in TABLE_REGION_MAP.items():
        raw_data_2 = raw_data_2.at[addr].set(
            jnp.array(repacked[name], dtype=jnp.uint8)
        )

    print("Parsing recovered bytes...")
    tokens_step_2 = parse_tables(raw_data_2)

    print("Comparing tokens...")
    for key in tokens_step_1:
        diff = tokens_step_1[key] != tokens_step_2[key]
        assert not jnp.any(diff), f"FAIL: {key} mismatch! Indices: {jnp.where(diff)}"
        print(f"PASS: {key}")
