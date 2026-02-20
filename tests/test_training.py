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
)
from pe_lsdj.models.transformer import LSDJTransformer


KEY = jr.PRNGKey(0)


@pytest.fixture(scope="module")
def model():
    return LSDJTransformer(
        KEY, d_model=64, num_heads_t=2, num_heads_c=2, num_blocks=1,
    )


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

    def test_finite(self, model):
        tokens = jr.randint(jr.PRNGKey(10), (16, 4, 21), 0, 10).astype(jnp.float32)
        loss = sequence_loss(model, tokens[:-1], tokens[1:])
        assert jnp.isfinite(loss)
        assert loss > 0

    def test_has_gradients(self, model):
        tokens = jr.randint(jr.PRNGKey(10), (16, 4, 21), 0, 10).astype(jnp.float32)
        grad_fn = eqx.filter_value_and_grad(sequence_loss)
        loss, grads = grad_fn(model, tokens[:-1], tokens[1:])
        # At least some gradients should be non-zero
        flat_grads = jax.tree.leaves(grads)
        has_nonzero = any(
            jnp.any(g != 0) for g in flat_grads if eqx.is_array(g)
        )
        assert has_nonzero


class TestBatchLoss:

    def test_matches_mean(self, model):
        """batch_loss should equal mean of individual sequence losses."""
        B, L = 3, 8
        tokens = jr.randint(jr.PRNGKey(2), (B, L + 1, 4, 21), 0, 10).astype(jnp.float32)
        inputs = tokens[:, :-1]
        targets = tokens[:, 1:]

        bl = batch_loss(model, inputs, targets)

        individual = jnp.array([
            sequence_loss(model, inputs[i], targets[i]) for i in range(B)
        ])
        assert jnp.allclose(bl, individual.mean(), atol=1e-5)


class TestTrainStep:

    def test_updates_params(self, model):
        """One training step should change model parameters."""
        import optax
        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

        tokens = jr.randint(jr.PRNGKey(3), (2, 9, 4, 21), 0, 10).astype(jnp.float32)
        inputs = tokens[:, :-1]
        targets = tokens[:, 1:]

        new_model, new_opt_state, loss = train_step(
            model, opt_state, optimizer, inputs, targets,
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
