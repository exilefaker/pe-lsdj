"""
StreamingBuffer: rolling deque of generated LSDJ song rows.

Generation is step-wise (one model step = one song_token per channel).
Steps accumulate into phrases, phrases into chains, chains into song rows.

Structure (per channel):
  song row → chain → [phrase_0, phrase_1, ..., phrase_{N-1}]
                       ↑ each phrase has STEPS_PER_PHRASE steps

Default num_phrases_per_chain = 4, so each song row holds 4×16 = 64 steps
per channel before advancing to the next row.

Push flow:
  push_step(tokens)  — call once per generated model step
    → writes step to SRAM immediately
    → when all phrases in the current chain are full, wires the chain to
      the song row and commits a RowEntry to the deque
    → if the deque is full, recycles the oldest RowEntry first

Recycle: frees chains + phrases, clears the song row's chain slots.

channel_mask (bool[4]):
  True = frozen channel. Steps for that channel are not written to SRAM.
  Allocation still happens (so song structure is consistent), but phrase
  content is left as whatever LSDJ or the user had there.
"""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from pe_lsdj.constants import (
    SONG_CHAINS_ADDR,
    CHAIN_PHRASES_ADDR,
    CHAIN_TRANSPOSES_ADDR,
    PHRASE_NOTES_ADDR,
    PHRASE_INSTR_ADDR,
    PHRASE_FX_ADDR,
    PHRASE_FX_VAL_ADDR,
    NUM_CHANNELS,
    STEPS_PER_PHRASE,
    FX_VALUES_FEATURE_DIM,
    EMPTY,
)
from pe_lsdj.tokenizer.detokenize import phrase_step_bytes
from .sram import read_sram, write_sram, write_sram_range
from .alloc import AllocationManager

_CHAIN_PHRASE_SLOTS = 16

# ── Song-screen display caches (discovered via find_song_display.py) ──────────
# LSDJ maintains two WRAM caches for the song grid, both updated on screen init:
#
#   0xC2EC  raw chain-ID cache   — layout: base + song_row*4 + ch
#   0xD800  tile-index cache     — layout: base + ((2+song_row)%32)*32 + (3+ch*3)
#                                  (mirrors the BG tile map at 0x9800)
#
# The rendering loop copies the tile-index cache → VRAM (0x9800) every frame,
# so VRAM writes alone last only one frame.  We must write all three.
_WRAM_CHAIN_CACHE = 0xC2EC   # raw chain IDs; stride = NUM_CHANNELS
_WRAM_TILE_CACHE  = 0xD800   # tile indices;  mirrors BG map layout
_BG_MAP_BASE      = 0x9800   # VRAM BG tile map
_FONT_BASE        = 0x04     # tile 0x04='0', 0x05='1', ... 0x13='F'
_GRID_HEADER_ROWS = 2
_GRID_CHAN_COL0   = 3
_GRID_CHAN_STRIDE = 3


def _grid_tile_offset(song_row: int, ch: int) -> int:
    """BG-map / tile-cache offset for the first hex digit of (song_row, ch)."""
    tile_row = (_GRID_HEADER_ROWS + song_row) % 32
    return tile_row * 32 + _GRID_CHAN_COL0 + ch * _GRID_CHAN_STRIDE


def _chain_tile_pair(chain_id: int, empty_tile: int) -> tuple[int, int]:
    if chain_id == EMPTY:
        return empty_tile, empty_tile
    return _FONT_BASE + ((chain_id >> 4) & 0xF), _FONT_BASE + (chain_id & 0xF)


@dataclass
class RowEntry:
    song_row:   int
    chain_ids:  list[int]        # [ch]
    phrase_ids: list[list[int]]  # [ch][phrase_slot]


@dataclass
class _BuildingChain:
    """A chain being filled step by step before it's committed to the song row."""
    song_row:    int
    chain_ids:   list[int]        # [ch]
    phrase_ids:  list[list[int]]  # [ch][phrase_slot]
    step_cursor: int = 0          # 0 .. num_phrases_per_chain*STEPS_PER_PHRASE - 1

    @property
    def phrase_slot(self) -> int:
        return self.step_cursor // STEPS_PER_PHRASE

    @property
    def step_in_phrase(self) -> int:
        return self.step_cursor % STEPS_PER_PHRASE


class StreamingBuffer:
    """
    Rolling buffer of generated LSDJ song rows, filled step by step.

    Args:
        pyboy:                 PyBoy instance (LSDJ running)
        alloc:                 AllocationManager (already loaded)
        next_song_row:         First song row index to write (0–255)
        max_rows:              Max rows to hold before recycling (default 64)
        num_phrases_per_chain: Phrases per chain (default 4; × 16 steps = 64 per row)
        channel_mask:          bool[4]; True = freeze writes for that channel
    """

    def __init__(
        self,
        pyboy,
        alloc: AllocationManager,
        next_song_row: int = 0,
        max_rows: int = 64,
        num_phrases_per_chain: int = 4,
        channel_mask: list[bool] | None = None,
    ):
        self.pyboy                 = pyboy
        self.alloc                 = alloc
        self.max_rows              = max_rows
        self.num_phrases_per_chain = num_phrases_per_chain
        self.channel_mask          = channel_mask or [False] * NUM_CHANNELS
        self._next_row             = next_song_row % 256
        self._buf: deque[RowEntry] = deque()
        self._building: Optional[_BuildingChain] = None
        self._total_phrases_committed: int = 0
        self._empty_tile: int = self._read_empty_tile()

    # ── public interface ──────────────────────────────────────────────────────

    @property
    def steps_per_row(self) -> int:
        return self.num_phrases_per_chain * STEPS_PER_PHRASE

    @property
    def committed_rows(self) -> int:
        return len(self._buf)

    @property
    def phrases_committed(self) -> int:
        """Monotonically-increasing count of phrases fully written to SRAM.
        Unaffected by recycling — safe to diff against phrases_consumed."""
        return self._total_phrases_committed

    @property
    def next_song_row(self) -> int:
        return self._next_row

    @property
    def building(self) -> Optional[_BuildingChain]:
        return self._building

    def push_step(self, tokens: np.ndarray) -> Optional[RowEntry]:
        """
        Consume one generated model step.

        Args:
            tokens: (NUM_CHANNELS, token_dim) uint16

        Returns a committed RowEntry when a chain is complete, else None.
        """
        if self._building is None:
            b = self._start_chain()
            if b is None:
                return None   # allocation failure
            self._building = b

        b = self._building
        self._write_step(b, tokens)
        b.step_cursor += 1

        if b.step_cursor % STEPS_PER_PHRASE == 0:
            self._total_phrases_committed += 1

        if b.step_cursor == self.steps_per_row:
            return self._commit()
        return None

    def rows_ahead_of(self, playheads: list[int]) -> int:
        """
        How many committed rows sit ahead of the slowest channel's playhead.
        Used by the main loop to decide whether to keep generating.
        """
        if not self._buf:
            return 0
        last_committed = self._buf[-1].song_row
        slowest = min(playheads)
        return (last_committed - slowest) % 256

    # ── internal ─────────────────────────────────────────────────────────────

    def _read_empty_tile(self) -> int:
        """Read the tile LSDJ uses for '--' (empty chain slot) from VRAM.

        next_song_row is always an SRAM-empty row at init time (lsdj_stream.py
        calls find_first_empty_row before constructing the buffer), so reading
        its BG tile position gives the '--' tile directly.  Falls back to
        scanning other SRAM-empty slots if needed.
        """
        for row in range(256):
            for ch in range(NUM_CHANNELS):
                val = read_sram(self.pyboy,
                                SONG_CHAINS_ADDR.start + row * NUM_CHANNELS + ch)
                if val == EMPTY:
                    offset = _grid_tile_offset(row, ch)
                    tile = self.pyboy.memory[_BG_MAP_BASE + offset]
                    if not (_FONT_BASE <= tile <= _FONT_BASE + 15):
                        return tile
        return 0x01  # safe fallback

    def _write_chain_display(self, song_row: int, ch: int, chain_id: int) -> None:
        """Keep all three display caches in sync with an SRAM chain write."""
        hi, lo  = _chain_tile_pair(chain_id, self._empty_tile)
        offset  = _grid_tile_offset(song_row, ch)
        # Raw chain-ID cache (LSDJ UI logic)
        self.pyboy.memory[_WRAM_CHAIN_CACHE + song_row * NUM_CHANNELS + ch] = chain_id
        # Tile-index cache (LSDJ renders from this to VRAM every frame)
        self.pyboy.memory[_WRAM_TILE_CACHE + offset]     = hi
        self.pyboy.memory[_WRAM_TILE_CACHE + offset + 1] = lo
        # VRAM directly (gets the current frame right, before the next render tick)
        self.pyboy.memory[_BG_MAP_BASE + offset]     = hi
        self.pyboy.memory[_BG_MAP_BASE + offset + 1] = lo

    def _start_chain(self) -> Optional[_BuildingChain]:
        """Allocate chains + phrases for the next row; recycle if buffer full."""
        if len(self._buf) >= self.max_rows:
            self._recycle_oldest()

        chain_ids  = []
        phrase_ids = []

        for ch in range(NUM_CHANNELS):
            cid = self.alloc.alloc_chain()
            if cid is None:
                for c in chain_ids: self.alloc.free_chain(c)
                return None
            chain_ids.append(cid)

            ch_phrases = []
            for _ in range(self.num_phrases_per_chain):
                pid = self.alloc.alloc_phrase()
                if pid is None:
                    for c in chain_ids:  self.alloc.free_chain(c)
                    for p in ch_phrases: self.alloc.free_phrase(p)
                    for plist in phrase_ids:
                        for p in plist: self.alloc.free_phrase(p)
                    return None
                ch_phrases.append(pid)

                # Initialise phrase to EMPTY so LSDJ sees valid data
                # even if the playhead reaches it before we finish filling
                self._init_phrase(pid)

            phrase_ids.append(ch_phrases)

            # Wire chain → phrases and chain → song row immediately
            self._wire_chain(cid, ch_phrases)

        song_row = self._next_row
        self._next_row = (self._next_row + 1) % 256

        # Wire song row → chains (SRAM + VRAM for immediate display refresh)
        for ch in range(NUM_CHANNELS):
            write_sram(self.pyboy,
                       SONG_CHAINS_ADDR.start + song_row * 4 + ch,
                       chain_ids[ch])
            self._write_chain_display(song_row, ch, chain_ids[ch])
        return _BuildingChain(song_row, chain_ids, phrase_ids)

    def _commit(self) -> RowEntry:
        b = self._building
        entry = RowEntry(b.song_row, b.chain_ids, b.phrase_ids)
        self._buf.append(entry)
        self._building = None
        return entry

    def _write_step(self, b: _BuildingChain, tokens: np.ndarray) -> None:
        pslot = b.phrase_slot
        step  = b.step_in_phrase
        for ch in range(NUM_CHANNELS):
            if self.channel_mask[ch]:
                continue
            pid = b.phrase_ids[ch][pslot]
            self._write_phrase_step(pid, step, tokens[ch])

    def _write_phrase_step(
        self, pid: int, step: int, ch_tokens: np.ndarray
    ) -> None:
        note, instr, fx_cmd, fxval = phrase_step_bytes(ch_tokens)
        write_sram(self.pyboy, PHRASE_NOTES_ADDR.start  + pid * STEPS_PER_PHRASE + step, note)
        write_sram(self.pyboy, PHRASE_INSTR_ADDR.start  + pid * STEPS_PER_PHRASE + step, instr)
        write_sram(self.pyboy, PHRASE_FX_ADDR.start     + pid * STEPS_PER_PHRASE + step, fx_cmd)
        write_sram(self.pyboy, PHRASE_FX_VAL_ADDR.start + pid * STEPS_PER_PHRASE + step, fxval)

    def _init_phrase(self, pid: int) -> None:
        """Fill a phrase with EMPTY / NULL so it's valid before steps arrive."""
        base_n = PHRASE_NOTES_ADDR.start  + pid * STEPS_PER_PHRASE
        base_i = PHRASE_INSTR_ADDR.start  + pid * STEPS_PER_PHRASE
        base_f = PHRASE_FX_ADDR.start     + pid * STEPS_PER_PHRASE
        base_v = PHRASE_FX_VAL_ADDR.start + pid * STEPS_PER_PHRASE
        write_sram_range(self.pyboy, base_n, [EMPTY] * STEPS_PER_PHRASE)
        write_sram_range(self.pyboy, base_i, [EMPTY] * STEPS_PER_PHRASE)
        write_sram_range(self.pyboy, base_f, [0]     * STEPS_PER_PHRASE)
        write_sram_range(self.pyboy, base_v, [0]     * STEPS_PER_PHRASE)

    def _wire_chain(self, cid: int, phrase_ids: list[int]) -> None:
        """Write chain → phrase mapping and zero transposes."""
        chain_base = CHAIN_PHRASES_ADDR.start + cid * _CHAIN_PHRASE_SLOTS
        trans_base = CHAIN_TRANSPOSES_ADDR.start + cid * _CHAIN_PHRASE_SLOTS

        phrase_row = phrase_ids + [EMPTY] * (_CHAIN_PHRASE_SLOTS - len(phrase_ids))
        write_sram_range(self.pyboy, chain_base, phrase_row)
        write_sram_range(self.pyboy, trans_base, [0] * _CHAIN_PHRASE_SLOTS)

    def _recycle_oldest(self) -> None:
        entry = self._buf.popleft()
        for ch in range(NUM_CHANNELS):
            self.alloc.free_chain(entry.chain_ids[ch])
            for pid in entry.phrase_ids[ch]:
                self.alloc.free_phrase(pid)
            write_sram(self.pyboy,
                       SONG_CHAINS_ADDR.start + entry.song_row * 4 + ch,
                       EMPTY)
            self._write_chain_display(entry.song_row, ch, EMPTY)
