from pe_lsdj.tokenizer import parse_instruments, parse_notes
from pe_lsdj.detokenizer import repack_instruments, repack_notes
from pe_lsdj.constants import INSTRUMENTS_ADDR, PHRASE_NOTES_ADDR, PHRASE_NOTES
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
