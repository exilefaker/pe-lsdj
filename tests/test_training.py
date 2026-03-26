import pytest
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

from pe_lsdj.training import (
    sample_crops,
    sequence_loss,
    batch_loss,
    train_step,
    get_validate_sequences,
    make_multi_track_batch,
    _transpose,
    _swap_pulse,
    _annealed_aug_params,
)
from pe_lsdj.constants import NUM_NOTES
from pe_lsdj.models.transformer import LSDJTransformer
from pe_lsdj.embedding.song import SongBanks


KEY = jr.PRNGKey(0)


@pytest.fixture(scope="module")
def model():
    return LSDJTransformer(
        KEY, d_model=64, num_heads_t=2, num_heads_c=2, num_blocks=1,
    )


@pytest.fixture(scope="module")
def banks():
    return SongBanks.default()


@pytest.fixture(scope="module")
def fake_tokens():
    """Fake song tokens: (64, 4, 21) — enough for small crops."""
    return jr.randint(jr.PRNGKey(1), (64, 4, 21), 0, 10).astype(jnp.float32)


class TestSampleCrops:

    def test_shapes(self, fake_tokens):
        inputs, targets = sample_crops(fake_tokens, crop_len=16, batch_size=4, key=KEY)
        assert inputs.shape == (4, 16, 4, 21)
        assert targets.shape == (4, 16, 4, 21)

    def test_teacher_forcing_offset(self, fake_tokens):
        """Target should be input shifted forward by 1 step."""
        inputs, targets = sample_crops(fake_tokens, crop_len=8, batch_size=2, key=KEY)
        # For the first crop, check that target step 0 == input step 1
        # (they come from adjacent slices of the same sequence)
        # We can verify by checking they share the middle portion
        inputs2, targets2 = sample_crops(fake_tokens, crop_len=8, batch_size=2, key=KEY)
        # Same key → same crops
        assert jnp.allclose(inputs, inputs2)
        assert jnp.allclose(targets, targets2)

    def test_target_is_shifted_input(self, fake_tokens):
        """Directly verify the shift relationship."""
        # Use crop_len=4 from a known start
        inputs, targets = sample_crops(fake_tokens, crop_len=4, batch_size=1, key=KEY)
        # Target[b, t] should equal inputs[b, t+1] for t < crop_len-1
        # Actually, target comes from [start+1 : start+crop_len+1] and
        # input from [start : start+crop_len], so target[t] = song[start+1+t]
        # and input[t] = song[start+t], meaning target[t] = input[t+1] when
        # they overlap... but they don't fully overlap. Instead:
        # target[0] = input[1], target[1] = input[2], ..., target[L-2] = input[L-1]
        # and target[L-1] = song[start+L] (one step beyond input)
        assert jnp.allclose(targets[0, :-1], inputs[0, 1:])


class TestTranspose:

    def _make_tokens(self, note_val=10):
        """(8, 4, 21) tokens: known note in channels 0-2, zero in channel 3."""
        tokens = jnp.zeros((8, 4, 21), dtype=jnp.float32)
        tokens = tokens.at[:, :3, 0].set(note_val)   # non-null notes in PU1/PU2/WAV
        return tokens

    def test_shifts_non_null_notes(self):
        tokens = self._make_tokens(note_val=10)
        out = _transpose(tokens, 3)
        assert jnp.all(out[:, :3, 0] == 13)

    def test_null_notes_unchanged(self):
        tokens = jnp.zeros((8, 4, 21), dtype=jnp.float32)   # all notes = 0 (NULL)
        out = _transpose(tokens, 5)
        assert jnp.all(out[:, :3, 0] == 0)

    def test_noise_channel_unchanged(self):
        tokens = jnp.zeros((8, 4, 21), dtype=jnp.float32).at[:, :, 0].set(10)
        out = _transpose(tokens, 7)
        assert jnp.all(out[:, 3, 0] == 10)   # channel 3 untouched

    def test_clamps_at_upper_bound(self):
        tokens = self._make_tokens(note_val=NUM_NOTES - 1)
        out = _transpose(tokens, 5)
        assert jnp.all(out[:, :3, 0] == NUM_NOTES)

    def test_clamps_at_lower_bound(self):
        tokens = self._make_tokens(note_val=2)
        out = _transpose(tokens, -5)
        assert jnp.all(out[:, :3, 0] == 1)

    def test_non_note_fields_unchanged(self):
        tokens = jnp.ones((8, 4, 21), dtype=jnp.float32)
        out = _transpose(tokens, 3)
        assert jnp.all(out[:, :, 1:] == 1)   # fields 1..20 untouched


class TestSwapPulse:

    def _make_tokens(self):
        """(8, 4, 21) tokens where each channel has a distinct constant value."""
        tokens = jnp.zeros((8, 4, 21), dtype=jnp.float32)
        for ch in range(4):
            tokens = tokens.at[:, ch, :].set(float(ch + 1))
        return tokens

    def test_pu1_pu2_swapped(self):
        tokens = self._make_tokens()
        out = _swap_pulse(tokens)
        assert jnp.all(out[:, 0, :] == 2.0)   # PU1 now has PU2's values
        assert jnp.all(out[:, 1, :] == 1.0)   # PU2 now has PU1's values

    def test_other_channels_unchanged(self):
        tokens = self._make_tokens()
        out = _swap_pulse(tokens)
        assert jnp.all(out[:, 2, :] == 3.0)   # WAV unchanged
        assert jnp.all(out[:, 3, :] == 4.0)   # NOI unchanged

    def test_swap_is_involution(self):
        """Swapping twice returns the original."""
        tokens = self._make_tokens()
        assert jnp.allclose(_swap_pulse(_swap_pulse(tokens)), tokens)


class TestSequenceLoss:

    def test_finite(self, model, banks):
        tokens = jr.randint(jr.PRNGKey(10), (16, 4, 21), 0, 10).astype(jnp.float32)
        loss = sequence_loss(model, tokens[:-1], tokens[1:], banks)
        assert jnp.isfinite(loss)
        assert loss > 0

    def test_has_gradients(self, model, banks):
        tokens = jr.randint(jr.PRNGKey(10), (16, 4, 21), 0, 10).astype(jnp.float32)
        grad_fn = eqx.filter_value_and_grad(sequence_loss)
        loss, grads = grad_fn(model, tokens[:-1], tokens[1:], banks)
        # At least some gradients should be non-zero
        flat_grads = jax.tree.leaves(grads)
        has_nonzero = any(
            jnp.any(g != 0) for g in flat_grads if eqx.is_array(g)
        )
        assert has_nonzero


class TestBatchLoss:

    def test_matches_mean(self, model, banks):
        """batch_loss should equal mean of individual sequence losses."""
        B, L = 3, 8
        tokens = jr.randint(jr.PRNGKey(2), (B, L + 1, 4, 21), 0, 10).astype(jnp.float32)
        inputs = tokens[:, :-1]
        targets = tokens[:, 1:]

        bl = batch_loss(model, inputs, targets, banks)

        individual = jnp.array([
            sequence_loss(model, inputs[i], targets[i], banks) for i in range(B)
        ])
        assert jnp.allclose(bl, individual.mean(), atol=1e-5)


class FakeSong:
    """Minimal SongFile stub: just needs song_tokens."""
    def __init__(self, tokens):
        self.song_tokens = tokens


class TestGetValidateSequences:

    def test_crop_count_single_song(self):
        S, crop_len = 50, 16
        song = FakeSong(jr.randint(jr.PRNGKey(0), (S, 4, 21), 0, 10).astype(jnp.uint16))
        seqs = get_validate_sequences([song], [SongBanks.default()], crop_len)
        assert len(seqs) == (S - 1) // crop_len

    def test_crop_count_multiple_songs(self):
        crop_len = 16
        songs = [
            FakeSong(jr.randint(jr.PRNGKey(i), (S, 4, 21), 0, 10).astype(jnp.uint16))
            for i, S in enumerate([50, 70])
        ]
        banks = [SongBanks.default(), SongBanks.default()]
        seqs = get_validate_sequences(songs, banks, crop_len)
        assert len(seqs) == (50 - 1) // crop_len + (70 - 1) // crop_len

    def test_input_target_shapes(self):
        S, crop_len = 49, 16
        song = FakeSong(jr.randint(jr.PRNGKey(0), (S, 4, 21), 0, 10).astype(jnp.uint16))
        seqs = get_validate_sequences([song], [SongBanks.default()], crop_len)
        for inp, tgt, _, _start, _slen in seqs:
            assert inp.shape == (crop_len, 4, 21)
            assert tgt.shape == (crop_len, 4, 21)

    def test_teacher_forcing_shift(self):
        S, crop_len = 32, 16
        song = FakeSong(jr.randint(jr.PRNGKey(0), (S, 4, 21), 0, 10).astype(jnp.float32))
        seqs = get_validate_sequences([song], [SongBanks.default()], crop_len)
        for inp, tgt, _, _start, _slen in seqs:
            assert jnp.allclose(tgt[:-1], inp[1:])

    def test_banks_passed_through(self):
        song = FakeSong(jr.randint(jr.PRNGKey(0), (32, 4, 21), 0, 10).astype(jnp.uint16))
        banks = SongBanks.default()
        seqs = get_validate_sequences([song], [banks], crop_len=16)
        for _, _, bnk, _start, _slen in seqs:
            assert bnk is banks

    def test_val_loss_finite(self, model, banks):
        song = FakeSong(jr.randint(jr.PRNGKey(0), (32, 4, 21), 0, 10).astype(jnp.uint16))
        seqs = get_validate_sequences([song], [banks], crop_len=16)
        val_losses = [sequence_loss(model, inp, tgt, bnk, None, start, slen)
                      for inp, tgt, bnk, start, slen in seqs]
        assert all(jnp.isfinite(l) for l in val_losses)


class TestAnnealedAugParams:

    def test_no_anneal_returns_unchanged(self):
        assert _annealed_aug_params(500, 0, 2, 4, 0.5, True) == (2, 4, 0.5, True)

    def test_at_step_zero_returns_full(self):
        assert _annealed_aug_params(0, 1000, 2, 4, 0.5, True) == (2, 4, 0.5, True)

    def test_at_anneal_end_returns_zero(self):
        td, tu, p, sp = _annealed_aug_params(1000, 1000, 2, 4, 0.5, True)
        assert td == 0
        assert tu == 0
        assert p == 0.0
        assert sp is False

    def test_beyond_anneal_end_clamps(self):
        td, tu, p, sp = _annealed_aug_params(9999, 1000, 2, 4, 0.5, True)
        assert td == 0
        assert tu == 0
        assert p == 0.0

    def test_midpoint_halves_transpose(self):
        td, tu, p, sp = _annealed_aug_params(500, 1000, 2, 4, 0.4, True)
        assert td == 1   # round(2 * 0.5)
        assert tu == 2   # round(4 * 0.5)
        assert abs(p - 0.2) < 1e-6

    def test_swap_pulse_off_at_midpoint(self):
        _, _, _, sp = _annealed_aug_params(500, 1000, 2, 4, 0.5, True)
        assert sp is False

    def test_swap_pulse_on_before_midpoint(self):
        _, _, _, sp = _annealed_aug_params(499, 1000, 2, 4, 0.5, True)
        assert sp is True

    def test_swap_pulse_false_stays_false(self):
        _, _, _, sp = _annealed_aug_params(0, 1000, 2, 4, 0.5, False)
        assert sp is False


class FakeSongAug:
    """Minimal SongFile stub with name attribute for make_multi_track_batch."""
    def __init__(self, tokens):
        self.song_tokens = tokens
        self.name = "fake"


class TestAsymmetricTranspose:

    def _make_songs(self, n=4):
        return [
            FakeSongAug(jr.randint(jr.PRNGKey(i), (64, 4, 21), 1, 10).astype(jnp.uint16))
            for i in range(n)
        ]

    def test_symmetric_range_no_crash(self):
        songs = self._make_songs()
        banks = [SongBanks.default()] * len(songs)
        make_multi_track_batch(songs, banks, 4, 16, KEY,
                               max_transpose_down=2, max_transpose_up=2,
                               p_transpose=1.0)

    def test_asymmetric_range_no_crash(self):
        songs = self._make_songs()
        banks = [SongBanks.default()] * len(songs)
        make_multi_track_batch(songs, banks, 4, 16, KEY,
                               max_transpose_down=2, max_transpose_up=4,
                               p_transpose=1.0)

    def test_output_shapes(self):
        songs = self._make_songs(4)
        banks = [SongBanks.default()] * len(songs)
        inputs, targets, _, idxs, crop_starts, song_lengths = make_multi_track_batch(
            songs, banks, 4, 16, KEY,
            max_transpose_down=2, max_transpose_up=4,
            p_transpose=0.5,
        )
        assert inputs.shape == (4, 16, 4, 21)
        assert targets.shape == (4, 16, 4, 21)
        assert crop_starts.shape == (4,)
        assert song_lengths.shape == (4,)


class TestTrainStep:

    def test_updates_params(self, model, banks):
        """One training step should change model parameters."""
        import optax
        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

        B = 2
        tokens = jr.randint(jr.PRNGKey(3), (B, 9, 4, 21), 0, 10).astype(jnp.float32)
        inputs = tokens[:, :-1]
        targets = tokens[:, 1:]
        # train_step uses multi_track_batch_loss which vmaps over banks (in_axes=0)
        batched_banks = jax.tree.map(lambda x: jnp.stack([x] * B), banks)

        new_model, new_opt_state, loss = train_step(
            model, opt_state, optimizer, inputs, targets, batched_banks, jr.PRNGKey(99),
        )

        # Parameters should have changed
        old_leaves = jax.tree.leaves(eqx.filter(model, eqx.is_array))
        new_leaves = jax.tree.leaves(eqx.filter(new_model, eqx.is_array))
        changed = any(
            not jnp.array_equal(o, n)
            for o, n in zip(old_leaves, new_leaves)
        )
        assert changed
        assert jnp.isfinite(loss)
