"""
AllocationManager: mirrors LSDJ's in-SRAM phrase and chain allocation tables.

LSDJ uses two bit-packed tables:
  PHRASE_ALLOC_TABLE (32 bytes): one bit per phrase, 255 phrases total (bit 255 unused)
  CHAIN_ALLOC_TABLE  (16 bytes): one bit per chain,  128 chains total

Bit ordering: LSB-first within each byte.
  phrase i → byte (i // 8), bit (i % 8)

AllocationManager reads both tables at construction, maintains a mirror in
Python, and writes back to SRAM on every alloc/free so LSDJ stays in sync
(it uses the alloc tables to decide which slots are available for user edits).

NOTE: bit ordering (LSB vs MSB within each byte) needs empirical verification
against a real .sav file. See verify() for a diagnostic.
"""

from __future__ import annotations
from pe_lsdj.constants import (
    PHRASE_ALLOC_TABLE_ADDR,
    CHAIN_ALLOC_TABLE_ADDR,
    NUM_PHRASES,
    NUM_CHAINS,
)
from .sram import read_sram, write_sram

_PHRASE_BYTES = 32   # 256 bits; only 255 used
_CHAIN_BYTES  = 16   # 128 bits


def _read_bitset(pyboy, base_addr: int, n_bytes: int, max_id: int) -> set[int]:
    """Read a bit-packed alloc table; return set of allocated IDs."""
    allocated = set()
    for byte_i in range(n_bytes):
        byte_val = read_sram(pyboy, base_addr + byte_i)
        for bit in range(8):
            item_id = byte_i * 8 + bit
            if item_id < max_id and (byte_val >> bit) & 1:
                allocated.add(item_id)
    return allocated


def _set_bit(pyboy, base_addr: int, item_id: int, value: int) -> None:
    """Set or clear a single bit in a SRAM bit-packed table."""
    byte_idx = item_id // 8
    bit       = item_id %  8
    flat_addr = base_addr + byte_idx
    current   = read_sram(pyboy, flat_addr)
    new_val   = (current | (1 << bit)) if value else (current & ~(1 << bit))
    write_sram(pyboy, flat_addr, new_val & 0xFF)


class AllocationManager:
    """
    Mirrors LSDJ's phrase and chain alloc tables.

    Usage:
        alloc = AllocationManager(pyboy)
        pid   = alloc.alloc_phrase()    # lowest free phrase; None if full
        cid   = alloc.alloc_chain()     # lowest free chain;  None if full
        alloc.free_phrase(pid)
        alloc.free_chain(cid)
    """

    def __init__(self, pyboy):
        self.pyboy = pyboy
        self._reload()

    def _reload(self):
        """Re-read both alloc tables from SRAM (call if LSDJ may have changed them)."""
        allocated_phrases = _read_bitset(
            self.pyboy, PHRASE_ALLOC_TABLE_ADDR.start, _PHRASE_BYTES, NUM_PHRASES
        )
        allocated_chains = _read_bitset(
            self.pyboy, CHAIN_ALLOC_TABLE_ADDR.start, _CHAIN_BYTES, NUM_CHAINS
        )
        self._free_phrases: set[int] = set(range(NUM_PHRASES)) - allocated_phrases
        self._free_chains:  set[int] = set(range(NUM_CHAINS))  - allocated_chains

    # ── allocation ────────────────────────────────────────────────────────────

    def alloc_phrase(self) -> int | None:
        """Allocate the lowest-numbered free phrase. Returns None if exhausted."""
        if not self._free_phrases:
            return None
        pid = min(self._free_phrases)
        self._free_phrases.discard(pid)
        _set_bit(self.pyboy, PHRASE_ALLOC_TABLE_ADDR.start, pid, 1)
        return pid

    def alloc_chain(self) -> int | None:
        """Allocate the lowest-numbered free chain. Returns None if exhausted."""
        if not self._free_chains:
            return None
        cid = min(self._free_chains)
        self._free_chains.discard(cid)
        _set_bit(self.pyboy, CHAIN_ALLOC_TABLE_ADDR.start, cid, 1)
        return cid

    def free_phrase(self, pid: int) -> None:
        self._free_phrases.add(pid)
        _set_bit(self.pyboy, PHRASE_ALLOC_TABLE_ADDR.start, pid, 0)

    def free_chain(self, cid: int) -> None:
        self._free_chains.add(cid)
        _set_bit(self.pyboy, CHAIN_ALLOC_TABLE_ADDR.start, cid, 0)

    # ── status ────────────────────────────────────────────────────────────────

    @property
    def free_phrase_count(self) -> int:
        return len(self._free_phrases)

    @property
    def free_chain_count(self) -> int:
        return len(self._free_chains)

    @property
    def allocated_phrase_count(self) -> int:
        return NUM_PHRASES - len(self._free_phrases)

    @property
    def allocated_chain_count(self) -> int:
        return NUM_CHAINS - len(self._free_chains)

    # ── diagnostics ───────────────────────────────────────────────────────────

    def verify(self) -> None:
        """
        Print alloc state for manual cross-checking against the LSDJ screen.
        Useful for confirming LSB-first bit ordering is correct.
        """
        print(f"Phrases: {self.allocated_phrase_count}/{NUM_PHRASES} allocated, "
              f"{self.free_phrase_count} free")
        allocated = sorted(set(range(NUM_PHRASES)) - self._free_phrases)
        print(f"  Allocated phrase IDs: {[hex(p)[2:].upper() for p in allocated[:32]]}"
              + (" ..." if len(allocated) > 32 else ""))

        print(f"Chains:  {self.allocated_chain_count}/{NUM_CHAINS} allocated, "
              f"{self.free_chain_count} free")
        allocated = sorted(set(range(NUM_CHAINS)) - self._free_chains)
        print(f"  Allocated chain IDs:  {[hex(c)[2:].upper() for c in allocated[:32]]}"
              + (" ..." if len(allocated) > 32 else ""))
