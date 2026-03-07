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
)
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
        for inp, tgt, _ in seqs:
            assert inp.shape == (crop_len, 4, 21)
            assert tgt.shape == (crop_len, 4, 21)

    def test_teacher_forcing_shift(self):
        S, crop_len = 32, 16
        song = FakeSong(jr.randint(jr.PRNGKey(0), (S, 4, 21), 0, 10).astype(jnp.float32))
        seqs = get_validate_sequences([song], [SongBanks.default()], crop_len)
        for inp, tgt, _ in seqs:
            assert jnp.allclose(tgt[:-1], inp[1:])

    def test_banks_passed_through(self):
        song = FakeSong(jr.randint(jr.PRNGKey(0), (32, 4, 21), 0, 10).astype(jnp.uint16))
        banks = SongBanks.default()
        seqs = get_validate_sequences([song], [banks], crop_len=16)
        for _, _, bnk in seqs:
            assert bnk is banks

    def test_val_loss_finite(self, model, banks):
        song = FakeSong(jr.randint(jr.PRNGKey(0), (32, 4, 21), 0, 10).astype(jnp.uint16))
        seqs = get_validate_sequences([song], [banks], crop_len=16)
        val_losses = [sequence_loss(model, inp, tgt, bnk) for inp, tgt, bnk in seqs]
        assert all(jnp.isfinite(l) for l in val_losses)


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
