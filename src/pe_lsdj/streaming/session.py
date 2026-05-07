"""
StreamingSession: orchestrates the generation thread and PyBoy tick loop.

Generator thread owns all JAX state; produces tokens into a queue.
Main thread ticks PyBoy at steady 60 fps and drains tokens into StreamingBuffer.
JAX releases the Python GIL during C++ computation, so both threads make
progress concurrently without audio stalls.

Write-ahead is measured in *phrases* using a self-calibrating conversion of the
raw 0xC74B counter.  We do not assume a fixed relationship between 0xC74B ticks
and musical phrases; instead we measure empirically how many ticks elapse each
time the slowest playhead advances by one song row, then divide by
num_phrases_per_chain to get c74b_per_phrase.  Intra-row phrase progress is
interpolated from ticks elapsed since the last row-advance event.
"""

from __future__ import annotations

import queue as stdlib_queue
import select
import sys
import termios
import threading
import time
import tty
from typing import Optional

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from pe_lsdj.constants import EMPTY, NUM_CHANNELS, SONG_CHAINS_ADDR, STEPS_PER_PHRASE
from pe_lsdj.embedding import SongBanks
from pe_lsdj.generation import generate_step_cached

from .alloc import AllocationManager
from .buffer import StreamingBuffer
from .sram import read_sram

# Per-channel WRAM song-row playhead addresses (0=PU1, 1=PU2, 2=WAV, 3=NOI)
_PLAYHEAD_ADDRS = [0xC39F, 0xC3A0, 0xC3A1, 0xC3A2]

# Address used as a monotonic clock for phrase-progress calibration.
# Its exact semantics (step? phrase? frame?) are BPM-dependent; we treat it
# as an opaque ticker and calibrate its rate against the playhead at runtime.
# Found via scripts/find_phrase_cursor.py + probe_phrase_cursor.py.
_PHRASE_CURSOR_ADDR = 0xC74B

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
        write_ahead_phrases: Target phrases of content to keep ahead of the
                      global step-clock (default 2, ≈1–2 s depending on BPM).
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
        write_ahead_phrases: int = 2,
        seed: int = 43,
        window: bool = False,
        loop_progress: float | None = None,
        instr_threshold: float = 0.5,
        table_threshold: float = 0.5,
        groove_threshold: float = 0.1,
        softsynth_threshold: float = 0.5,
        temp: float = 0.9,
    ):
        self.pyboy               = pyboy
        self.model               = model
        self.alloc               = alloc
        self.buf                 = buf
        self.write_ahead_phrases = write_ahead_phrases
        self.window              = window

        self._W           = W
        self._song_length = song_length
        self._thresholds  = (instr_threshold, table_threshold,
                             groove_threshold, softsynth_threshold)
        self._temp          = temp
        self._loop_progress = loop_progress

        # Mutable JAX state. Owned exclusively by the generator thread after
        # run() is called; mutated directly by the main thread during pre-generation.
        self._gen_key     = jr.PRNGKey(seed)
        self._last_hidden = last_hidden
        self._banks       = banks
        self._k_cache     = k_cache
        self._v_cache     = v_cache
        self._step_idx    = 0

        self._step_queue: Optional[stdlib_queue.Queue] = None
        self._ctrl_queue: Optional[stdlib_queue.Queue] = None
        self._stop_gen:   Optional[threading.Event]    = None
        self._gen_thread: Optional[threading.Thread]   = None
        self._ctrl_thread: Optional[threading.Thread]  = None

    # ── public ────────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Pre-generate one full row, start the generator thread,
        then tick PyBoy until Ctrl-C."""
        self._pregen()
        self._start_generator_thread()

        self._start_control_thread()
        print("\nStarting LSDJ playback — Ctrl-C to stop.")
        mode_str = f"locked @ {self._loop_progress:.0%}" if self._loop_progress is not None else "advancing"
        print(f"Progress mode : {mode_str}  (p = toggle,  < / > = nudge ±5%)")
        print(f"Temperature   : {self._temp:.2f}  ([ / ] = ±0.25)")
        print("(focus terminal window to use controls)\n")
        self.pyboy.button("start")
        self.pyboy.tick(render=self.window)

        # ── 0xC74B calibration state ──────────────────────────────────────────
        # We accumulate raw ticks and measure how many occur per song-row advance
        # of the slowest playhead.  After the first row advance we have a
        # c74b_per_row conversion and switch to phrase-level interpolation.
        prev_cursor       = self.pyboy.memory[_PHRASE_CURSOR_ADDR]
        cursor_total      = 0          # cumulative ticks since playback start

        playheads         = read_playheads(self.pyboy)
        initial_min_ph    = min(playheads)
        prev_min_ph       = initial_min_ph
        cursor_at_row_adv = 0          # cursor_total at the last row-advance

        _c74b_samples: list[float] = []
        _c74b_per_row: float | None = None

        rows_committed = self.buf.committed_rows

        try:
            while True:
                frame_start = time.perf_counter()

                # No JAX here — main thread ticks PyBoy at steady 60 fps.
                self.pyboy.tick(render=self.window)

                # Accumulate raw 0xC74B ticks.
                curr  = self.pyboy.memory[_PHRASE_CURSOR_ADDR]
                delta = (curr - prev_cursor) % 256
                if delta:
                    cursor_total += delta
                    prev_cursor   = curr

                # Detect playhead advance; update calibration.
                playheads  = read_playheads(self.pyboy)
                cur_min_ph = min(playheads)
                row_delta  = (cur_min_ph - prev_min_ph) % 256
                if row_delta > 0:
                    ticks_this = cursor_total - cursor_at_row_adv
                    if ticks_this > 0:
                        _c74b_samples.append(ticks_this / row_delta)
                        # Rolling average over the last 8 samples.
                        _c74b_per_row = sum(_c74b_samples[-8:]) / len(_c74b_samples[-8:])
                    cursor_at_row_adv = cursor_total
                    prev_min_ph       = cur_min_ph
                    c74b_str = f"{_c74b_per_row:.1f}" if _c74b_per_row is not None else "?"
                    if self._loop_progress is not None:
                        prog_str = f"{self._loop_progress:.0%} (locked)"
                    else:
                        frac = min(1.0, (self._W + self._step_idx) / self._song_length)
                        prog_str = f"{frac:.0%} (advancing)"
                    print(
                        f"[row+{row_delta}] ph={cur_min_ph:#04x}  "
                        f"c74b/row={c74b_str}  "
                        f"rows_in_buf={self.buf.committed_rows}  "
                        f"progress={prog_str}"
                    )

                # Compute phrases_consumed: coarse row count + intra-row interpolation.
                rows_advanced = (cur_min_ph - initial_min_ph) % 256
                if _c74b_per_row:
                    intra_frac = min(
                        (cursor_total - cursor_at_row_adv) / _c74b_per_row, 1.0
                    )
                else:
                    intra_frac = 0.0
                phrases_consumed = (rows_advanced + intra_frac) * self.buf.num_phrases_per_chain

                phrases_ahead = self.buf.phrases_committed - phrases_consumed

                # Apply live controls from terminal stdin.
                while not self._ctrl_queue.empty():
                    key = self._ctrl_queue.get_nowait()
                    if   key == ']': self._nudge_temp(+0.25)
                    elif key == '[': self._nudge_temp(-0.25)
                    elif key == '>': self._nudge_progress(+0.05)
                    elif key == '<': self._nudge_progress(-0.05)
                    elif key == 'p': self._toggle_progress_lock()

                # Drain pre-generated tokens from the queue into SRAM.
                # SRAM writes are µs-level; the inner loop runs until we're
                # sufficiently ahead or the queue runs dry.
                while phrases_ahead < self.write_ahead_phrases:
                    try:
                        next_token = self._step_queue.get_nowait()
                    except stdlib_queue.Empty:
                        break  # generator hasn't caught up; retry next frame
                    committed = self.buf.push_step(next_token)
                    if committed is not None:
                        rows_committed += 1
                        ph_str = " ".join(f"{p:02X}" for p in playheads)
                        print(
                            f"row 0x{committed.song_row:02X} committed  "
                            f"total={rows_committed}  "
                            f"playhead=[{ph_str}]  "
                            f"rows_ahead={self.buf.rows_ahead_of(playheads)}  "
                            f"phrases_ahead={phrases_ahead:.1f}  "
                            f"free={self.alloc.free_phrase_count}ph/"
                            f"{self.alloc.free_chain_count}ch"
                        )
                    phrases_ahead = self.buf.phrases_committed - phrases_consumed

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
            if self._ctrl_thread is not None:
                self._ctrl_thread.join(timeout=1.0)
            print(f"Stopped after {rows_committed} rows ({self._step_idx} steps).")
            self.pyboy.stop(save=False)

    # ── internal ─────────────────────────────────────────────────────────────

    def _gen_step(self) -> np.ndarray:
        """Run one model step, update internal state, return a numpy token array."""
        self._gen_key, step_key = jr.split(self._gen_key)
        carry = (self._last_hidden, self._banks, self._k_cache, self._v_cache)
        instr_t, table_t, groove_t, softsynth_t = self._thresholds
        if self._loop_progress is not None:
            position    = self._W + self._step_idx
            effective_sl = max(position + 1, round(position / self._loop_progress))
        else:
            effective_sl = self._song_length
        carry, next_token = _jit_step(
            carry, (step_key, jnp.int32(self._step_idx)),
            self.model, self._W, jnp.int32(effective_sl),
            instr_t, table_t, groove_t, softsynth_t,
            jnp.float32(self._temp),   # dynamic array — changes never retrace
        )
        self._last_hidden, self._banks, self._k_cache, self._v_cache = carry
        self._step_idx += 1
        return np.array(next_token)

    def _pregen(self) -> None:
        """Fill rows synchronously before playback starts.

        We need one row for the bootstrap period (the first row plays before the
        first row-advance event gives us calibration data) plus enough additional
        rows to satisfy write_ahead_phrases.  Formula:
            write_ahead_rows = 1 + ceil(write_ahead_phrases / num_phrases_per_chain)
        For the default (write_ahead_phrases=2, num_phrases_per_chain=4) this is 2 rows.
        """
        npp = self.buf.num_phrases_per_chain
        write_ahead_rows = 1 + (self.write_ahead_phrases + npp - 1) // npp
        steps_needed     = write_ahead_rows * self.buf.steps_per_row
        print(
            f"Pre-generating {write_ahead_rows} row(s) "
            f"({steps_needed} steps) ..."
        )
        for _ in range(steps_needed):
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

    def _nudge_temp(self, delta: float) -> None:
        self._temp = max(0.1, min(2.0, round(self._temp + delta, 2)))
        print(f"[temp → {self._temp:.2f}]", flush=True)

    def _nudge_progress(self, delta: float) -> None:
        if self._loop_progress is None:
            # Snap to current fraction before nudging
            position = self._W + self._step_idx
            self._loop_progress = round(
                min(0.95, position / self._song_length), 2
            )
        self._loop_progress = max(0.05, min(0.95, round(self._loop_progress + delta, 2)))
        print(f"[progress locked @ {self._loop_progress:.0%}]", flush=True)

    def _toggle_progress_lock(self) -> None:
        if self._loop_progress is not None:
            self._loop_progress = None
            print("[progress → advancing]", flush=True)
        else:
            position = self._W + self._step_idx
            self._loop_progress = round(
                min(0.95, position / self._song_length), 2
            )
            print(f"[progress locked @ {self._loop_progress:.0%}]", flush=True)

    def _control_loop(self) -> None:
        """Control thread: read keypresses from stdin and post to _ctrl_queue.

        Uses cbreak mode so each keypress is delivered immediately without
        waiting for Enter.  Terminal settings are always restored on exit.
        Silently exits if stdin is not a TTY (headless / piped environments).
        """
        if not sys.stdin.isatty():
            return
        fd  = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not self._stop_gen.is_set():
                # Short poll so we check _stop_gen regularly.
                if not select.select([sys.stdin], [], [], 0.05)[0]:
                    continue
                ch = sys.stdin.read(1)
                # Arrow keys arrive as 3-byte escape sequences: ESC [ <letter>.
                # Consume the rest of the sequence with a tight timeout so we
                # don't mis-interpret a bare ESC as the start of a sequence.
                if ch == '\x1b' and select.select([sys.stdin], [], [], 0.02)[0]:
                    ch += sys.stdin.read(2)
                self._ctrl_queue.put_nowait(ch)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def _start_control_thread(self) -> None:
        self._ctrl_queue  = stdlib_queue.Queue()
        self._ctrl_thread = threading.Thread(
            target=self._control_loop, daemon=True, name="pe-lsdj-ctrl"
        )
        self._ctrl_thread.start()

    def _start_generator_thread(self) -> None:
        queue_depth      = self.write_ahead_phrases * STEPS_PER_PHRASE * 2
        self._step_queue = stdlib_queue.Queue(maxsize=queue_depth)
        self._stop_gen   = threading.Event()
        self._gen_thread = threading.Thread(
            target=self._generator_loop, daemon=True, name="pe-lsdj-gen"
        )
        self._gen_thread.start()
        print("Generator thread started.")
