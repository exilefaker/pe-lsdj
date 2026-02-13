from pe_lsdj.tokenizer import (
    parse_instruments, parse_notes, parse_softsynths,
    parse_fx_commands, parse_fx_values, parse_tables, parse_grooves,
    get_resolve_maps, get_traces,
)
from pe_lsdj.detokenizer import repack_instruments, repack_notes, repack_softsynths, repack_fx_values, repack_tables, repack_grooves
from pe_lsdj.constants import (
    INSTRUMENTS_ADDR, PHRASE_NOTES_ADDR, PHRASE_NOTES, SOFTSYNTH_PARAMS_ADDR,
    PHRASE_FX_ADDR, PHRASE_FX_VAL_ADDR, GROOVES_ADDR,
    TABLE_ENVELOPES_ADDR, TABLE_TRANSPOSES_ADDR,
    TABLE_FX_ADDR, TABLE_FX_VAL_ADDR, TABLE_FX_2_ADDR, TABLE_FX_2_VAL_ADDR,
    CMD_A, CMD_H, NUM_TABLES, STEPS_PER_TABLE,
    TABLE_ENV_VOLUME, TABLE_ENV_DURATION, TABLE_TRANSPOSE,
    TABLE_FX_1, TABLE_FX_VALUE_1, TABLE_FX_2, TABLE_FX_VALUE_2,
    FX_VALUE_KEYS,
)
import numpy as np
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
    raw_tables_1, traces_1 = parse_tables(raw_data)

    print("Compiling tokens back to bytes...")
    repacked = repack_tables(raw_tables_1)

    print("Reassembling raw data...")
    raw_data_2 = raw_data
    for name, addr in TABLE_REGION_MAP.items():
        raw_data_2 = raw_data_2.at[addr].set(
            jnp.array(repacked[name], dtype=jnp.uint8)
        )

    print("Parsing recovered bytes...")
    raw_tables_2, traces_2 = parse_tables(raw_data_2)

    print("Comparing raw table tokens...")
    for key in raw_tables_1:
        diff = raw_tables_1[key] != raw_tables_2[key]
        assert not jnp.any(diff), f"FAIL: {key} (raw) mismatch! Indices: {jnp.where(diff)}"
        print(f"PASS: {key} (raw)")

    print("Comparing trace tokens...")
    for key in traces_1:
        diff = traces_1[key] != traces_2[key]
        assert not jnp.any(diff), f"FAIL: {key} (trace) mismatch! Indices: {jnp.where(diff)}"
        print(f"PASS: {key} (trace)")


# ===== Synthetic table execution trace edge-case tests =====

FLAT_DIM = NUM_TABLES * STEPS_PER_TABLE  # 512


def _idx(table, step):
    """Flat index for (table, step)."""
    return table * STEPS_PER_TABLE + step


def _marker(table, step):
    """Expected marker value at (table, step)."""
    return _idx(table, step) + 1


def _make_synthetic(cmd1=None, cmd2=None):
    """Synthetic flat table data with unique flat-index markers.

    TRANSPOSE and FX_VALUE_1[:,0] carry L-column markers (flat_idx + 1).
    FX_VALUE_2[:,0] carries R-column markers (flat_idx + 1).
    FX_1/FX_2 reflect the supplied command arrays.
    """
    fx_dim = len(FX_VALUE_KEYS)
    markers = jnp.arange(FLAT_DIM) + 1
    c1 = jnp.array(cmd1) if cmd1 is not None else jnp.zeros(FLAT_DIM)
    c2 = jnp.array(cmd2) if cmd2 is not None else jnp.zeros(FLAT_DIM)
    return {
        TABLE_ENV_VOLUME: jnp.ones(FLAT_DIM),
        TABLE_ENV_DURATION: jnp.ones(FLAT_DIM),
        TABLE_TRANSPOSE: markers,
        TABLE_FX_1: c1,
        TABLE_FX_VALUE_1: jnp.zeros((FLAT_DIM, fx_dim)).at[:, 0].set(markers),
        TABLE_FX_2: c2,
        TABLE_FX_VALUE_2: jnp.zeros((FLAT_DIM, fx_dim)).at[:, 0].set(markers),
    }


def _run_traces(cmd1, val1, cmd2, val2):
    """Build resolve maps and traces from command/value arrays."""
    flat = _make_synthetic(cmd1, cmd2)
    resolve_L, resolve_R = get_resolve_maps(cmd1, val1, cmd2, val2)
    traces = get_traces(resolve_L, resolve_R, flat)
    return traces


def _empty_arrays():
    """Return zeroed cmd/val arrays."""
    cmd1 = np.zeros(FLAT_DIM, dtype=np.uint8)
    cmd2 = np.zeros(FLAT_DIM, dtype=np.uint8)
    val1 = np.zeros(FLAT_DIM, dtype=np.uint8)
    val2 = np.zeros(FLAT_DIM, dtype=np.uint8)
    return cmd1, val1, cmd2, val2


def test_trace_no_commands():
    """No A/H commands: trace equals raw data for both columns."""
    cmd1, val1, cmd2, val2 = _empty_arrays()
    traces = _run_traces(cmd1, val1, cmd2, val2)

    # L column: TRANSPOSE should be sequential markers per table
    for t in range(NUM_TABLES):
        for s in range(STEPS_PER_TABLE):
            assert int(traces[TABLE_TRANSPOSE][t, s]) == _marker(t, s)

    # R column: FX_VALUE_2[:,0] should match
    for t in range(NUM_TABLES):
        for s in range(STEPS_PER_TABLE):
            assert int(traces[TABLE_FX_VALUE_2][t, s, 0]) == _marker(t, s)


def test_trace_A_cmd1_jumps_both():
    """A in CMD1 at Table 0 Step 3 → Table 1: both L and R jump."""
    cmd1, val1, cmd2, val2 = _empty_arrays()
    cmd1[_idx(0, 3)] = CMD_A
    val1[_idx(0, 3)] = 1  # target: Table 1

    traces = _run_traces(cmd1, val1, cmd2, val2)
    tr_L = traces[TABLE_TRANSPOSE]
    tr_R = traces[TABLE_FX_VALUE_2][:, :, 0]

    # Steps 0-2: Table 0 data
    for s in range(3):
        assert int(tr_L[0, s]) == _marker(0, s)
        assert int(tr_R[0, s]) == _marker(0, s)

    # Steps 3+: Table 1 data (starting from step 0)
    for s in range(3, STEPS_PER_TABLE):
        assert int(tr_L[0, s]) == _marker(1, s - 3), f"L step {s}"
        assert int(tr_R[0, s]) == _marker(1, s - 3), f"R step {s}"


def test_trace_A_cmd2_jumps_both():
    """A in CMD2 only at Table 0 Step 3 → Table 1: both columns still jump."""
    cmd1, val1, cmd2, val2 = _empty_arrays()
    cmd2[_idx(0, 3)] = CMD_A
    val2[_idx(0, 3)] = 1

    traces = _run_traces(cmd1, val1, cmd2, val2)
    tr_L = traces[TABLE_TRANSPOSE]

    for s in range(3):
        assert int(tr_L[0, s]) == _marker(0, s)
    for s in range(3, STEPS_PER_TABLE):
        assert int(tr_L[0, s]) == _marker(1, s - 3), f"L should also jump at step {s}"


def test_trace_dual_A_cmd1_priority():
    """A in both CMD1→Table 1 and CMD2→Table 2: CMD1 wins."""
    cmd1, val1, cmd2, val2 = _empty_arrays()
    cmd1[_idx(0, 3)] = CMD_A
    val1[_idx(0, 3)] = 1  # CMD1 → Table 1
    cmd2[_idx(0, 3)] = CMD_A
    val2[_idx(0, 3)] = 2  # CMD2 → Table 2

    traces = _run_traces(cmd1, val1, cmd2, val2)

    # CMD1 takes priority: both columns should land in Table 1
    assert int(traces[TABLE_TRANSPOSE][0, 3]) == _marker(1, 0), "L: CMD1 priority"
    assert int(traces[TABLE_FX_VALUE_2][0, 3, 0]) == _marker(1, 0), "R: CMD1 priority"


def test_trace_H_diverges_columns():
    """H in CMD1 at Step 2 → hop to step 0: L loops, R stays linear."""
    cmd1, val1, cmd2, val2 = _empty_arrays()
    cmd1[_idx(0, 2)] = CMD_H
    val1[_idx(0, 2)] = 0x00  # hop to step 0

    traces = _run_traces(cmd1, val1, cmd2, val2)
    tr_L = traces[TABLE_TRANSPOSE]
    tr_R = traces[TABLE_FX_VALUE_2][:, :, 0]

    # L: iter0=step0, iter1=step1, iter2=hop→step0, iter3=step1, ...
    # Pattern: 0, 1, 0, 1, 0, 1, ... (period-2 from the start)
    for i in range(STEPS_PER_TABLE):
        assert int(tr_L[0, i]) == _marker(0, i % 2), f"L iter {i}"

    # R: no H in CMD2, continues linearly 0..15
    for i in range(STEPS_PER_TABLE):
        assert int(tr_R[0, i]) == _marker(0, i), f"R iter {i}"


def test_trace_H_then_A_syncs():
    """H on CMD1 causes desync, then A on CMD2 re-syncs both columns."""
    cmd1, val1, cmd2, val2 = _empty_arrays()
    cmd1[_idx(0, 2)] = CMD_H
    val1[_idx(0, 2)] = 0x00  # L hops to step 0
    cmd2[_idx(0, 4)] = CMD_A
    val2[_idx(0, 4)] = 1     # R jumps to Table 1

    traces = _run_traces(cmd1, val1, cmd2, val2)
    tr_L = traces[TABLE_TRANSPOSE]
    tr_R = traces[TABLE_FX_VALUE_2][:, :, 0]

    # Walk through:
    # iter 0: L=0,  R=0   (both normal)
    # iter 1: L=1,  R=1
    # iter 2: L→0(H), R=2  (L hops, R continues)
    # iter 3: L=1,  R=3
    # iter 4: L→0(H), R→T1s0(A)  → A detected on R, sync both to T1s0
    # iter 5: L=T1s1, R=T1s1  (synced, continue together)
    assert int(tr_L[0, 0]) == _marker(0, 0)
    assert int(tr_L[0, 1]) == _marker(0, 1)
    assert int(tr_L[0, 2]) == _marker(0, 0)  # H hop
    assert int(tr_L[0, 3]) == _marker(0, 1)
    assert int(tr_L[0, 4]) == _marker(1, 0)  # synced to Table 1
    assert int(tr_L[0, 5]) == _marker(1, 1)

    assert int(tr_R[0, 0]) == _marker(0, 0)
    assert int(tr_R[0, 1]) == _marker(0, 1)
    assert int(tr_R[0, 2]) == _marker(0, 2)  # R didn't hop
    assert int(tr_R[0, 3]) == _marker(0, 3)
    assert int(tr_R[0, 4]) == _marker(1, 0)  # A jump
    assert int(tr_R[0, 5]) == _marker(1, 1)


def test_trace_A_chain():
    """A chain: Table 0 step 0 → Table 1 → Table 2. Resolves to Table 2."""
    cmd1, val1, cmd2, val2 = _empty_arrays()
    cmd1[_idx(0, 0)] = CMD_A
    val1[_idx(0, 0)] = 1  # Table 0 → Table 1
    cmd1[_idx(1, 0)] = CMD_A
    val1[_idx(1, 0)] = 2  # Table 1 → Table 2

    traces = _run_traces(cmd1, val1, cmd2, val2)
    tr_L = traces[TABLE_TRANSPOSE]

    # Chain resolves: Table 0 step 0 → Table 2 step 0
    assert int(tr_L[0, 0]) == _marker(2, 0), "Should resolve through chain"
    assert int(tr_L[0, 1]) == _marker(2, 1), "Should continue in Table 2"

    # Table 1 step 0 also resolves to Table 2
    assert int(tr_L[1, 0]) == _marker(2, 0)


def test_trace_other_tables_unaffected():
    """Commands in Table 0 should not affect other tables' traces."""
    cmd1, val1, cmd2, val2 = _empty_arrays()
    cmd1[_idx(0, 3)] = CMD_A
    val1[_idx(0, 3)] = 1

    traces = _run_traces(cmd1, val1, cmd2, val2)
    tr_L = traces[TABLE_TRANSPOSE]

    # Table 2 should be completely untouched
    for s in range(STEPS_PER_TABLE):
        assert int(tr_L[2, s]) == _marker(2, s)
    # Table 1 also untouched (it's the target, not the source)
    for s in range(STEPS_PER_TABLE):
        assert int(tr_L[1, s]) == _marker(1, s)
