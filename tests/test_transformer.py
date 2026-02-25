import pytest
import jax
import jax.numpy as jnp
import jax.random as jr

from pe_lsdj.models.transformer import (
    AxialTransformerBlock,
    EntityDecoder,
    LSDJTransformer,
    OutputHeads,
    TOKEN_HEADS,
    LOGIT_GROUPS,
    ENTITY_HEADS,
    ENTITY_HEAD_SPECS,
    ENTITY_HEAD_TOTAL_VOCAB,
    hard_targets,
    token_loss,
    entity_loss,
)
from pe_lsdj.embedding.song import SongBanks
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
        x2 = x1.at[5].set(jr.normal(jr.PRNGKey(2), (4, d)))

        out1 = block(x1, mask)
        out2 = block(x2, mask)

        assert jnp.allclose(out1[:5], out2[:5], atol=1e-5)
        assert not jnp.allclose(out1[5:], out2[5:])


class TestOutputHeads:

    D_MODEL   = 64
    ENTITY_DIM = 32

    @pytest.fixture(scope="class")
    def heads(self):
        return OutputHeads(self.D_MODEL, self.ENTITY_DIM, KEY)

    def test_output_keys(self, heads):
        """All logit-group and entity head names should appear in output."""
        x = jnp.ones(64)
        out = heads(x)
        assert set(out.keys()) == set(TOKEN_HEADS.keys()) | set(ENTITY_HEADS.keys())

    def test_logit_group_shapes(self, heads):
        """Each logit-group head should have its declared vocab size."""
        x = jnp.ones(64)
        out = heads(x)
        for name, (pos, vocab) in TOKEN_HEADS.items():
            assert out[name].shape == (vocab,), (
                f"{name}: expected ({vocab},), got {out[name].shape}"
            )

    def test_entity_head_shapes(self, heads):
        """Each entity head should output total_field_vocab logits."""
        x = jnp.ones(64)
        out = heads(x)
        for name, total_vocab in ENTITY_HEAD_TOTAL_VOCAB.items():
            assert out[name].shape == (total_vocab,), (
                f"{name}: expected ({total_vocab},), got {out[name].shape}"
            )

    def test_log_probs_logit_groups(self, heads):
        """log_probs should return valid log-probabilities for logit-group heads."""
        x = jr.normal(jr.PRNGKey(1), (64,))
        lp = heads.log_probs(x)
        assert set(lp.keys()) == set(TOKEN_HEADS.keys())
        for name, logp in lp.items():
            assert jnp.allclose(jnp.exp(logp).sum(), 1.0, atol=1e-5), name

    def test_logit_group_weight_shapes(self, heads):
        """Each logit-group weight matrix should be (N, vocab, d_model)."""
        for group_name, members in LOGIT_GROUPS.items():
            n = len(members)
            vocab = members[0][2]
            assert heads.weights[group_name].shape == (n, vocab, 64), group_name

    def test_entity_decoder_shapes(self, heads):
        """Each entity decoder should have correct layer shapes."""
        for name, total_vocab in ENTITY_HEAD_TOTAL_VOCAB.items():
            dec = heads.entity_decoders[name]
            assert dec.linear_in.weight.shape  == (self.ENTITY_DIM, self.D_MODEL), name
            assert dec.linear_out.weight.shape == (total_vocab, self.ENTITY_DIM),  name


class TestHardTargetsAndLoss:

    def test_hard_targets_shapes(self):
        tokens = jnp.zeros(21, dtype=jnp.int32)
        targets = hard_targets(tokens)
        assert set(targets.keys()) == set(TOKEN_HEADS.keys())
        for name, (pos, vocab) in TOKEN_HEADS.items():
            assert targets[name].shape == (vocab,), name

    def test_hard_targets_one_hot(self):
        tokens = jnp.array([5] + [0] * 20, dtype=jnp.int32)
        targets = hard_targets(tokens)
        assert targets['note'][5] == 1.0
        assert targets['note'].sum() == 1.0

    def test_token_loss_finite(self):
        heads = OutputHeads(64, 32, KEY)
        x = jr.normal(jr.PRNGKey(1), (64,))
        logits = heads(x)
        tokens = jnp.zeros(21, dtype=jnp.int32)
        targets = hard_targets(tokens)
        loss = token_loss(logits, targets)
        assert jnp.isfinite(loss)
        assert loss > 0

    def test_soft_targets_lower_loss(self):
        """Soft targets matching the model's own logit-group predictions → lower loss."""
        heads = OutputHeads(64, 32, KEY)
        x = jr.normal(jr.PRNGKey(1), (64,))
        logits = heads(x)

        soft_targets = {
            name: jax.nn.softmax(logits[name])
            for name in TOKEN_HEADS
        }
        loss_soft = token_loss(logits, soft_targets)
        loss_hard = token_loss(logits, hard_targets(jnp.zeros(21, dtype=jnp.int32)))
        assert loss_soft < loss_hard

    def test_entity_loss_finite(self):
        """entity_loss should return a finite scalar."""
        heads = OutputHeads(64, 32, KEY)
        x = jr.normal(jr.PRNGKey(1), (64,))
        out = heads(x)
        entity_logits = {name: out[name] for name in ENTITY_HEAD_SPECS}
        banks = SongBanks.default()
        tokens = jnp.zeros(21, dtype=jnp.int32)
        loss = entity_loss(entity_logits, banks, tokens)
        assert jnp.isfinite(loss)

    def test_entity_loss_null_tokens(self):
        """All-zero tokens (NULL entity IDs) should still produce finite loss."""
        heads = OutputHeads(64, 32, KEY)
        x = jr.normal(jr.PRNGKey(2), (64,))
        out = heads(x)
        entity_logits = {name: out[name] for name in ENTITY_HEAD_SPECS}
        banks = SongBanks.default()
        tokens = jnp.zeros(21, dtype=jnp.int32)
        loss = entity_loss(entity_logits, banks, tokens)
        assert jnp.isfinite(loss)


class TestLSDJTransformer:

    @pytest.fixture(scope="class")
    def model(self):
        return LSDJTransformer(
            KEY,
            d_model=64,
            entity_dim=32,
            num_heads_t=2,
            num_heads_c=2,
            num_blocks=2,
        )

    def test_output_keys(self, model):
        tokens = jnp.zeros((8, 4, 21))
        out = model(tokens)
        assert isinstance(out, dict)
        assert set(out.keys()) == set(TOKEN_HEADS.keys()) | set(ENTITY_HEADS.keys())

    def test_logit_group_output_shapes(self, model):
        S = 8
        tokens = jnp.zeros((S, 4, 21))
        out = model(tokens)
        for name, (pos, vocab) in TOKEN_HEADS.items():
            assert out[name].shape == (S, 4, vocab), (
                f"{name}: expected ({S}, 4, {vocab}), got {out[name].shape}"
            )

    def test_entity_head_output_shapes(self, model):
        S = 8
        tokens = jnp.zeros((S, 4, 21))
        out = model(tokens)
        for name, total_vocab in ENTITY_HEAD_TOTAL_VOCAB.items():
            assert out[name].shape == (S, 4, total_vocab), (
                f"{name}: expected ({S}, 4, {total_vocab}), got {out[name].shape}"
            )

    def test_default_banks(self):
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

    def test_with_banks_no_crash(self, model):
        """with_banks should return a valid model that still produces output."""
        banks = SongBanks.default()
        new_model = model.with_banks(banks)
        out = new_model(jnp.zeros((4, 4, 21)))
        assert out['note'].shape == (4, 4, NUM_NOTES)
