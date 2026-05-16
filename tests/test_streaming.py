"""
Unit tests for atomic SRAM write operations added during streaming entity work.
No ROM or real PyBoy required — MockPyBoy backs memory with a plain dict.
"""
import json
import types

import numpy as np
import pytest

from pe_lsdj.constants import (
    GROOVES_ADDR, INSTR_ALLOC_TABLE_ADDR, INSTRUMENTS_ADDR,
    NUM_GROOVES, NUM_INSTRUMENTS, NUM_TABLES,
    STEPS_PER_GROOVE, STEPS_PER_TABLE,
    TABLE_ALLOC_TABLE_ADDR, TABLE_WIDTH,
    TABLE_ENVELOPES_ADDR, TABLE_TRANSPOSES_ADDR,
    TABLE_FX_ADDR, TABLE_FX_VAL_ADDR,
    TABLE_FX_2_ADDR, TABLE_FX_2_VAL_ADDR,
)
from pe_lsdj.streaming.sram import write_sram, write_sram_range, read_sram, read_sram_range
from pe_lsdj.tokenizer.tokenize import INSTRUMENT_FIELDS

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

class MockMemory:
    def __init__(self):
        self._store = {}

    def __getitem__(self, key):
        return self._store.get(key, 0)

    def __setitem__(self, key, value):
        self._store[key] = value


class MockPyBoy:
    def __init__(self):
        self.memory = MockMemory()


def make_mock_banks(
    instrs_occupied=None,
    grooves_occupied=None,
    tables_occupied=None,
):
    """Return a namespace that looks like SongBanks to _write_new_entities.

    Shapes match the real SongBanks from SongBanks.from_songfile():
      instruments: (N+1, len(INSTRUMENT_FIELDS))
      grooves:     (N+1, STEPS_PER_GROOVE*2)  — session reshapes to (..., 16, 2)
      tables:      (N+1, TABLE_WIDTH)          — flat, _arr_to_tables_dict expects this
    """
    b = types.SimpleNamespace()
    b.instrs_occupied  = np.zeros(NUM_INSTRUMENTS + 1, dtype=bool) if instrs_occupied  is None else instrs_occupied
    b.grooves_occupied = np.zeros(NUM_GROOVES     + 1, dtype=bool) if grooves_occupied is None else grooves_occupied
    b.tables_occupied  = np.zeros(NUM_TABLES      + 1, dtype=bool) if tables_occupied  is None else tables_occupied
    b.instruments = np.zeros((NUM_INSTRUMENTS + 1, len(INSTRUMENT_FIELDS)), dtype=np.uint16)
    b.grooves     = np.zeros((NUM_GROOVES     + 1, STEPS_PER_GROOVE * 2), dtype=np.uint16)
    b.tables      = np.zeros((NUM_TABLES      + 1, TABLE_WIDTH), dtype=np.uint16)
    return b


# ---------------------------------------------------------------------------
# Group 1 — sram.py bank/offset arithmetic
# ---------------------------------------------------------------------------

class TestSRAMHelpers:

    def test_write_read_bank0(self):
        pb = MockPyBoy()
        write_sram(pb, 0x0000, 0xAB)
        assert pb.memory[0, 0xA000] == 0xAB

    def test_write_read_bank1(self):
        pb = MockPyBoy()
        write_sram(pb, 0x2000, 0xCD)
        assert pb.memory[1, 0xA000] == 0xCD

    def test_write_read_bank3(self):
        pb = MockPyBoy()
        write_sram(pb, 0x6000, 0xEF)
        assert pb.memory[3, 0xA000] == 0xEF

    def test_bank_boundary(self):
        pb = MockPyBoy()
        write_sram(pb, 0x1FFF, 0x11)
        write_sram(pb, 0x2000, 0x22)
        assert pb.memory[0, 0xBFFF] == 0x11
        assert pb.memory[1, 0xA000] == 0x22

    def test_write_sram_range_spans_bank(self):
        pb = MockPyBoy()
        write_sram_range(pb, 0x1FFE, [0xAA, 0xBB, 0xCC, 0xDD])
        assert pb.memory[0, 0xBFFE] == 0xAA
        assert pb.memory[0, 0xBFFF] == 0xBB
        assert pb.memory[1, 0xA000] == 0xCC
        assert pb.memory[1, 0xA001] == 0xDD

    def test_read_sram_range(self):
        pb = MockPyBoy()
        for i, v in enumerate([0x01, 0x02, 0x03]):
            write_sram(pb, 0x0010 + i, v)
        assert read_sram_range(pb, 0x0010, 3) == [0x01, 0x02, 0x03]


# ---------------------------------------------------------------------------
# Group 2 — lsdj_replay._apply_entity
# ---------------------------------------------------------------------------

class TestApplyEntity:

    @pytest.fixture(autouse=True)
    def _import(self):
        import importlib.util, pathlib
        spec = importlib.util.spec_from_file_location(
            "lsdj_replay",
            pathlib.Path(__file__).parent.parent / "scripts" / "lsdj_replay.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self._apply_entity = mod._apply_entity

    def test_apply_instr_id1(self):
        pb = MockPyBoy()
        ent = {"type": "instr", "id": 1, "bytes": [0xAB] * 16}
        self._apply_entity(pb, ent)
        row = 0
        for i in range(16):
            assert read_sram(pb, INSTRUMENTS_ADDR.start + row * 16 + i) == 0xAB
        assert read_sram(pb, INSTR_ALLOC_TABLE_ADDR.start + row) == 1

    def test_apply_instr_offset(self):
        pb = MockPyBoy()
        ent = {"type": "instr", "id": 5, "bytes": [0x42] * 16}
        self._apply_entity(pb, ent)
        row = 4
        for i in range(16):
            assert read_sram(pb, INSTRUMENTS_ADDR.start + row * 16 + i) == 0x42

    def test_apply_groove(self):
        pb = MockPyBoy()
        ent = {"type": "groove", "id": 2, "bytes": [0x11] * STEPS_PER_GROOVE}
        self._apply_entity(pb, ent)
        row = 1
        for i in range(STEPS_PER_GROOVE):
            assert read_sram(pb, GROOVES_ADDR.start + row * STEPS_PER_GROOVE + i) == 0x11
        # grooves have no alloc table
        assert read_sram(pb, TABLE_ALLOC_TABLE_ADDR.start + row) == 0

    def test_apply_table(self):
        pb = MockPyBoy()
        ent = {
            "type":       "table",
            "id":         1,
            "envelopes":  [0x01] * STEPS_PER_TABLE,
            "transposes": [0x02] * STEPS_PER_TABLE,
            "fx_cmd_1":   [0x03] * STEPS_PER_TABLE,
            "fx_val_1":   [0x04] * STEPS_PER_TABLE,
            "fx_cmd_2":   [0x05] * STEPS_PER_TABLE,
            "fx_val_2":   [0x06] * STEPS_PER_TABLE,
        }
        self._apply_entity(pb, ent)
        row = 0
        assert read_sram(pb, TABLE_ENVELOPES_ADDR.start  + row * STEPS_PER_TABLE) == 0x01
        assert read_sram(pb, TABLE_TRANSPOSES_ADDR.start + row * STEPS_PER_TABLE) == 0x02
        assert read_sram(pb, TABLE_FX_ADDR.start         + row * STEPS_PER_TABLE) == 0x03
        assert read_sram(pb, TABLE_FX_VAL_ADDR.start     + row * STEPS_PER_TABLE) == 0x04
        assert read_sram(pb, TABLE_FX_2_ADDR.start       + row * STEPS_PER_TABLE) == 0x05
        assert read_sram(pb, TABLE_FX_2_VAL_ADDR.start   + row * STEPS_PER_TABLE) == 0x06
        assert read_sram(pb, TABLE_ALLOC_TABLE_ADDR.start + row) == 1


# ---------------------------------------------------------------------------
# Group 3 — StreamingSession._write_new_entities
# ---------------------------------------------------------------------------

def _make_session_stub(pyboy, banks, record_path=None, step_idx=0):
    """Return a minimal namespace that _write_new_entities can be bound to."""
    from pe_lsdj.streaming.session import StreamingSession

    iocc = np.array(banks.instrs_occupied)
    gocc = np.array(banks.grooves_occupied)
    tocc = np.array(banks.tables_occupied)

    stub = types.SimpleNamespace(
        pyboy             = pyboy,
        _banks            = banks,
        _written_instr_ids  = {k for k in range(1, NUM_INSTRUMENTS + 1) if iocc[k]},
        _written_groove_ids = {k for k in range(1, NUM_GROOVES     + 1) if gocc[k]},
        _written_table_ids  = {k for k in range(1, NUM_TABLES      + 1) if tocc[k]},
        _record_path      = record_path,
        _recorded_entities = [],
        _step_idx         = step_idx,
    )
    stub._write_new_entities = StreamingSession._write_new_entities.__get__(stub)
    return stub


class TestWriteNewEntities:

    def test_new_instr_written(self):
        pb = MockPyBoy()
        banks = make_mock_banks()
        banks.instrs_occupied[3] = True
        banks.instruments[3, :] = 0xAB  # all 16 bytes = 0xAB

        stub = _make_session_stub(pb, banks)
        stub._written_instr_ids = set()  # nothing written yet
        stub._write_new_entities()

        row = 2  # id 3 → row 2
        assert read_sram(pb, INSTR_ALLOC_TABLE_ADDR.start + row) == 1
        assert 3 in stub._written_instr_ids

    def test_existing_instr_not_rewritten(self):
        pb = MockPyBoy()
        banks = make_mock_banks()
        banks.instrs_occupied[3] = True

        stub = _make_session_stub(pb, banks)
        stub._written_instr_ids = {3}  # already written
        stub._write_new_entities()

        # SRAM should remain all zeros — nothing written
        assert pb.memory._store == {}

    def test_new_table_all_regions(self):
        pb = MockPyBoy()
        banks = make_mock_banks()
        banks.tables_occupied[1] = True

        stub = _make_session_stub(pb, banks)
        stub._written_table_ids = set()
        stub._write_new_entities()

        assert 1 in stub._written_table_ids
        # alloc byte set
        assert read_sram(pb, TABLE_ALLOC_TABLE_ADDR.start + 0) == 1

    def test_no_recording_no_log(self):
        pb = MockPyBoy()
        banks = make_mock_banks()
        banks.instrs_occupied[1] = True

        stub = _make_session_stub(pb, banks, record_path=None)
        stub._written_instr_ids = set()
        stub._write_new_entities()

        assert stub._recorded_entities == []

    def test_recording_logs_entity(self):
        pb = MockPyBoy()
        banks = make_mock_banks()
        banks.instrs_occupied[2] = True

        stub = _make_session_stub(pb, banks, record_path="out.npz", step_idx=7)
        stub._written_instr_ids = set()
        stub._write_new_entities()

        assert len(stub._recorded_entities) == 1
        entry = stub._recorded_entities[0]
        assert entry["type"]  == "instr"
        assert entry["id"]    == 2
        assert entry["step"]  == 7
        assert "bytes" in entry


# ---------------------------------------------------------------------------
# Group 4 — entity delta log round-trip
# ---------------------------------------------------------------------------

class TestEntityDeltaLog:

    def test_npz_round_trip(self, tmp_path):
        entities = [
            {"step": 0, "type": "instr", "id": 1, "bytes": [0xAB] * 16},
            {"step": 5, "type": "groove", "id": 2, "bytes": [0x11] * STEPS_PER_GROOVE},
        ]
        path = tmp_path / "test.npz"
        np.savez(str(path), entities=np.bytes_(json.dumps(entities)))

        data = np.load(str(path), allow_pickle=False)
        assert "entities" in data.files
        loaded = json.loads(bytes(data["entities"]).decode())
        assert loaded == entities

    def test_replay_applies_entities_to_sram(self):
        import importlib.util, pathlib
        spec = importlib.util.spec_from_file_location(
            "lsdj_replay",
            pathlib.Path(__file__).parent.parent / "scripts" / "lsdj_replay.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        pb = MockPyBoy()
        entities = [
            {"step": 0, "type": "instr", "id": 1, "bytes": [0x99] * 16},
            {"step": 1, "type": "groove", "id": 1, "bytes": [0x55] * STEPS_PER_GROOVE},
        ]
        for ent in entities:
            mod._apply_entity(pb, ent)

        assert read_sram(pb, INSTRUMENTS_ADDR.start) == 0x99
        assert read_sram(pb, GROOVES_ADDR.start)     == 0x55

    def test_old_recording_no_entities_key(self, tmp_path):
        path = tmp_path / "old.npz"
        np.savez(
            str(path),
            tokens=np.zeros((1, 4, 21), dtype=np.uint16),
            config=np.bytes_(json.dumps({})),
            events=np.bytes_(json.dumps([])),
            # no 'entities' key
        )
        data = np.load(str(path), allow_pickle=False)
        entities = json.loads(bytes(data["entities"]).decode()) if "entities" in data.files else []
        assert entities == []
