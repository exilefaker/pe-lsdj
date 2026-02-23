import pytest
import jax
import jax.numpy as jnp
import jax.random as jr

from pe_lsdj.models.transformer import (
    AxialTransformerBlock,
    LSDJTransformer,
    OutputHeads,
    TOKEN_HEADS,
    LOGIT_GROUPS,
    ENTITY_HEADS,
    hard_targets,
    token_loss,
)
from pe_lsdj.constants import NUM_NOTES


KEY = jr.PRNGKey(0)


class TestAxialTransformerBlock:

    @pytest.fixture(scope="class")
    def block(self):
        return AxialTransformerBlock(d_model=64, num_heads_t=2, num_heads_c=2, key=KEY)

    def test_shape(self, block):
        x = jnp.ones((16, 4, 64))
        mask = jnp.tril(jnp.ones((16, 16), dtype=bool))
        out = block(x, mask)
        assert out.shape == (16, 4, 64)

    def test_causal(self, block):
        """Changing a future timestep should not affect past outputs."""
        S, d = 8, 64
        mask = jnp.tril(jnp.ones((S, S), dtype=bool))

        x1 = jr.normal(jr.PRNGKey(1), (S, 4, d))
        x2 = x1.at[5].set(jr.normal(jr.PRNGKey(2), (4, d)))  # modify step 5

        out1 = block(x1, mask)
        out2 = block(x2, mask)

        # Steps 0-4 should be identical (they can't see step 5)
        assert jnp.allclose(out1[:5], out2[:5], atol=1e-5)
        # Step 5+ should differ
        assert not jnp.allclose(out1[5:], out2[5:])


class TestOutputHeads:

    @pytest.fixture(scope="class")
    def heads(self):
        return OutputHeads(64, KEY)

    def test_output_keys(self, heads):
        """All 21 head names should be present in the output."""
        x = jnp.ones(64)
        out = heads(x)
        assert set(out.keys()) == set(TOKEN_HEADS.keys())

    def test_output_shapes(self, heads):
        """Each head's output should match its vocab size."""
        x = jnp.ones(64)
        out = heads(x)
        for name, (pos, vocab) in TOKEN_HEADS.items():
            assert out[name].shape == (vocab,), (
                f"{name}: expected ({vocab},), got {out[name].shape}"
            )

    def test_log_probs_sum_to_one(self, heads):
        """log_probs should be valid log-probabilities (exp sums to ~1)."""
        x = jr.normal(jr.PRNGKey(1), (64,))
        lp = heads.log_probs(x)
        for name, logp in lp.items():
            assert jnp.allclose(jnp.exp(logp).sum(), 1.0, atol=1e-5), name

    def test_logit_group_weight_shapes(self, heads):
        """Each logit group weight matrix should be (N, vocab, d_model)."""
        for group_name, members in LOGIT_GROUPS.items():
            n = len(members)
            vocab = members[0][2]
            assert heads.weights[group_name].shape == (n, vocab, 64), group_name

    def test_entity_projection_shapes(self, heads):
        """Each entity projection should be (query_dim, d_model)."""
        default_dims = {'instr_id': 128, 'table_id': 64, 'groove_id': 64}
        for name, q_dim in default_dims.items():
            assert heads.entity_projections[name].shape == (q_dim, 64), name

    def test_entity_bank_emb_shapes(self, heads):
        """Each entity bank embedding should be (vocab, query_dim)."""
        default_dims = {'instr_id': 128, 'table_id': 64, 'groove_id': 64}
        for name, (pos, vocab) in ENTITY_HEADS.items():
            q_dim = default_dims[name]
            assert heads.entity_bank_embs[name].shape == (vocab, q_dim), name


class TestHardTargetsAndLoss:

    def test_hard_targets_shapes(self):
        tokens = jnp.zeros(21, dtype=jnp.int32)
        targets = hard_targets(tokens)
        for name, (pos, vocab) in TOKEN_HEADS.items():
            assert targets[name].shape == (vocab,), name

    def test_hard_targets_one_hot(self):
        tokens = jnp.array([5] + [0] * 20, dtype=jnp.int32)
        targets = hard_targets(tokens)
        assert targets['note'][5] == 1.0
        assert targets['note'].sum() == 1.0

    def test_token_loss_finite(self):
        heads = OutputHeads(64, KEY)
        x = jr.normal(jr.PRNGKey(1), (64,))
        logits = heads(x)
        tokens = jnp.zeros(21, dtype=jnp.int32)
        targets = hard_targets(tokens)
        loss = token_loss(logits, targets)
        assert jnp.isfinite(loss)
        assert loss > 0

    def test_soft_targets_lower_loss(self):
        """Soft targets that match the logits should produce lower loss."""
        heads = OutputHeads(64, KEY)
        x = jr.normal(jr.PRNGKey(1), (64,))
        logits = heads(x)

        # Soft targets from the model's own predictions (should be low loss)
        soft_targets = {
            name: jax.nn.softmax(logits[name])
            for name in logits
        }
        loss_soft = token_loss(logits, soft_targets)

        # Hard targets at index 0 (arbitrary, likely high loss)
        loss_hard = token_loss(logits, hard_targets(jnp.zeros(21, dtype=jnp.int32)))

        assert loss_soft < loss_hard


class TestLSDJTransformer:

    @pytest.fixture(scope="class")
    def model(self):
        return LSDJTransformer(
            KEY,
            d_model=64,
            num_heads_t=2,
            num_heads_c=2,
            num_blocks=2,
        )

    def test_output_is_logits_dict(self, model):
        tokens = jnp.zeros((8, 4, 21))
        out = model(tokens)
        assert isinstance(out, dict)
        assert set(out.keys()) == set(TOKEN_HEADS.keys())

    def test_output_shapes(self, model):
        S = 8
        tokens = jnp.zeros((S, 4, 21))
        out = model(tokens)
        for name, (pos, vocab) in TOKEN_HEADS.items():
            assert out[name].shape == (S, 4, vocab), (
                f"{name}: expected ({S}, 4, {vocab}), got {out[name].shape}"
            )

    def test_default_banks(self):
        """Model constructable with just a key."""
        model = LSDJTransformer(
            jr.PRNGKey(42),
            d_model=64,
            num_heads_t=2,
            num_heads_c=1,
            num_blocks=1,
        )
        out = model(jnp.zeros((4, 4, 21)))
        assert 'note' in out
        assert out['note'].shape == (4, 4, NUM_NOTES)

    def test_entity_heads_present(self, model):
        """Entity reference heads should appear in output."""
        out = model(jnp.zeros((4, 4, 21)))
        for name, (pos, vocab) in ENTITY_HEADS.items():
            assert name in out
            assert out[name].shape == (4, 4, vocab), name
