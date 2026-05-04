"""
StreamingSession: orchestrates the generation thread and PyBoy tick loop.

Generator thread owns all JAX state; produces tokens into a queue.
Main thread ticks PyBoy at steady 60 fps and drains tokens into StreamingBuffer.
JAX releases the Python GIL during C++ computation, so both threads make
progress concurrently without audio stalls.
"""

from __future__ import annotations

import queue as stdlib_queue
import threading
import time
from typing import Optional

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from pe_lsdj.constants import EMPTY, NUM_CHANNELS, SONG_CHAINS_ADDR
from pe_lsdj.embedding import SongBanks
from pe_lsdj.generation import generate_step_cached

from .alloc import AllocationManager
from .buffer import StreamingBuffer
from .sram import read_sram

# Per-channel WRAM playhead addresses (0=PU1, 1=PU2, 2=WAV, 3=NOI)
_PLAYHEAD_ADDRS = [0xC39F, 0xC3A0, 0xC3A1, 0xC3A2]
_FRAME_DURATION = 1.0 / 60.0  # null-mode rate cap

_jit_step = eqx.filter_jit(generate_step_cached)


def find_first_empty_row(pyboy) -> int:
    """Return the first song row where all 4 channel slots are EMPTY (0xFF)."""
    for row in range(256):
        if all(
            read_sram(pyboy, SONG_CHAINS_ADDR.start + row * NUM_CHANNELS + ch) == EMPTY
            for ch in range(NUM_CHANNELS)
        ):
            return row
    return 0  # fully packed; wrap and overwrite from row 0


def read_playheads(pyboy) -> list[int]:
    """Return current song-row playhead for each channel (0–255)."""
    return [pyboy.memory[addr] for addr in _PLAYHEAD_ADDRS]


class StreamingSession:
    """
    Real-time LSDJ generation session.

    Args:
        pyboy:        Running PyBoy instance (LSDJ booted, not yet playing).
        model:        LSDJTransformer in inference mode.
        alloc:        AllocationManager (already loaded).
        buf:          StreamingBuffer (already constructed).
        last_hidden:  Final hidden state from KV-cache prefill.
        banks:        SongBanks from the prompt song.
        k_cache, v_cache: Prefilled KV caches.
        W:            Number of prompt steps used in the prefill.
        song_length:  Song-length hint passed to the model.
        write_ahead:  Target rows ahead of the playhead to maintain.
        seed:         RNG seed for generation.
        window:       True if running SDL2 (skips null-mode rate cap).
        instr/table/groove/softsynth_threshold: Entity prediction thresholds.
        temp:         Sampling temperature.
    """

    def __init__(
        self,
        pyboy,
        model,
        alloc: AllocationManager,
        buf: StreamingBuffer,
        last_hidden,
        banks: SongBanks,
        k_cache,
        v_cache,
        W: int,
        song_length: int,
        write_ahead: int,
        seed: int = 43,
        window: bool = False,
        instr_threshold: float = 0.5,
        table_threshold: float = 0.5,
        groove_threshold: float = 0.1,
        softsynth_threshold: float = 0.5,
        temp: float = 0.9,
    ):
        self.pyboy       = pyboy
        self.model       = model
        self.alloc       = alloc
        self.buf         = buf
        self.write_ahead = write_ahead
        self.window      = window

        self._W           = W
        self._song_length = song_length
        self._thresholds  = (instr_threshold, table_threshold,
                             groove_threshold, softsynth_threshold)
        self._temp        = temp

        # Mutable JAX state. Owned exclusively by the generator thread after
        # run() is called; mutated directly by the main thread during pre-generation.
        self._gen_key     = jr.PRNGKey(seed)
        self._last_hidden = last_hidden
        self._banks       = banks
        self._k_cache     = k_cache
        self._v_cache     = v_cache
        self._step_idx    = 0

        self._step_queue: Optional[stdlib_queue.Queue] = None
        self._stop_gen:   Optional[threading.Event]    = None
        self._gen_thread: Optional[threading.Thread]   = None

    # ── public ────────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Pre-generate write_ahead rows, start the generator thread, then
        tick PyBoy until Ctrl-C."""
        self._pregen()
        self._start_generator_thread()

        print("\nStarting LSDJ playback — Ctrl-C to stop.\n")
        self.pyboy.button("start")
        self.pyboy.tick(render=self.window)

        rows_committed = self.buf.committed_rows
        log_every      = max(1, self.write_ahead // 2)

        try:
            while True:
                frame_start = time.perf_counter()

                # No JAX here — main thread ticks PyBoy at steady 60 fps.
                self.pyboy.tick(render=self.window)

                playheads = read_playheads(self.pyboy)
                ahead     = self.buf.rows_ahead_of(playheads)

                # Drain pre-generated tokens from the queue into SRAM.
                # SRAM writes are µs-level; draining a full row costs <1 ms.
                while ahead < self.write_ahead:
                    try:
                        next_token = self._step_queue.get_nowait()
                    except stdlib_queue.Empty:
                        break  # generator hasn't caught up; retry next frame
                    committed = self.buf.push_step(next_token)
                    if committed is not None:
                        rows_committed += 1
                        if rows_committed % log_every == 0:
                            ph_str = " ".join(f"{p:02X}" for p in playheads)
                            print(
                                f"row 0x{committed.song_row:02X} committed  "
                                f"total={rows_committed}  "
                                f"playhead=[{ph_str}]  "
                                f"ahead={self.buf.rows_ahead_of(playheads)}  "
                                f"free={self.alloc.free_phrase_count}ph/"
                                f"{self.alloc.free_chain_count}ch"
                            )
                    ahead = self.buf.rows_ahead_of(playheads)

                if not self.window:
                    elapsed   = time.perf_counter() - frame_start
                    remainder = _FRAME_DURATION - elapsed
                    if remainder > 0:
                        time.sleep(remainder)

        except KeyboardInterrupt:
            print()
        finally:
            self._stop_gen.set()
            self._gen_thread.join(timeout=2.0)
            print(f"Stopped after {rows_committed} rows ({self._step_idx} steps).")
            self.pyboy.stop(save=False)

    # ── internal ─────────────────────────────────────────────────────────────

    def _gen_step(self) -> np.ndarray:
        """Run one model step, update internal state, return a numpy token array."""
        self._gen_key, step_key = jr.split(self._gen_key)
        carry = (self._last_hidden, self._banks, self._k_cache, self._v_cache)
        instr_t, table_t, groove_t, softsynth_t = self._thresholds
        carry, next_token = _jit_step(
            carry, (step_key, jnp.int32(self._step_idx)),
            self.model, self._W, self._song_length,
            instr_t, table_t, groove_t, softsynth_t,
            self._temp,
        )
        self._last_hidden, self._banks, self._k_cache, self._v_cache = carry
        self._step_idx += 1
        return np.array(next_token)

    def _pregen(self) -> None:
        """Fill write_ahead rows synchronously before playback starts."""
        print(
            f"Pre-generating {self.write_ahead} rows "
            f"({self.write_ahead * self.buf.steps_per_row} steps) ..."
        )
        while self.buf.committed_rows < self.write_ahead:
            token     = self._gen_step()
            committed = self.buf.push_step(token)
            if committed is not None:
                print(f"  pre-gen row 0x{committed.song_row:02X}")

    def _generator_loop(self) -> None:
        """Generator thread body: produce tokens into _step_queue indefinitely."""
        while not self._stop_gen.is_set():
            token_np = self._gen_step()
            # Block until queue has space (natural backpressure).
            while not self._stop_gen.is_set():
                try:
                    self._step_queue.put(token_np, timeout=0.05)
                    break
                except stdlib_queue.Full:
                    continue

    def _start_generator_thread(self) -> None:
        queue_depth      = self.write_ahead * self.buf.steps_per_row * 2
        self._step_queue = stdlib_queue.Queue(maxsize=queue_depth)
        self._stop_gen   = threading.Event()
        self._gen_thread = threading.Thread(
            target=self._generator_loop, daemon=True, name="pe-lsdj-gen"
        )
        self._gen_thread.start()
        print("Generator thread started.")
