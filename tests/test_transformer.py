import pytest
import jax
import jax.numpy as jnp
import jax.random as jr

from pe_lsdj.models.transformer import (
    AxialTransformerBlock,
    EntityDecoder,
    GrooveDecoder,
    TableDecoder,
    SoftSynthDecoder,
    InstrumentDecoder,
    LSDJTransformer,
    OutputHeads,
    TOKEN_HEADS,
    LOGIT_GROUPS,
    ENTITY_HEADS,
    WAVEFRAME_DIM,
    GROOVE_CONT_N,
    TABLE_SCALAR_CAT_TOTAL_VOCAB,
    TABLE_SCALAR_CONT_N,
    INSTR_SCALAR_CAT_TOTAL_VOCAB,
    INSTR_SCALAR_CONT_N,
    SOFTSYNTH_CAT_TOTAL_VOCAB,
    SOFTSYNTH_CONT_N,
    N_TABLE_SLOTS,
    N_GROOVE_SLOTS,
    hard_targets,
    token_loss,
    entity_loss,
)
from pe_lsdj.embedding.song import SongBanks
from pe_lsdj.constants import NUM_NOTES


KEY = jr.PRNGKey(0)
D_MODEL    = 64
ENTITY_DIM = 32


class TestAxialTransformerBlock:

    @pytest.fixture(scope="class")
    def block(self):
        return AxialTransformerBlock(d_model=D_MODEL, num_heads_t=2, num_heads_c=2, key=KEY)

    def test_shape(self, block):
        x = jnp.ones((16, 4, D_MODEL))
        mask = jnp.tril(jnp.ones((16, 16), dtype=bool))
        out = block(x, mask)
        assert out.shape == (16, 4, D_MODEL)

    def test_causal(self, block):
        S, d = 8, D_MODEL
        mask = jnp.tril(jnp.ones((S, S), dtype=bool))
        x1 = jr.normal(jr.PRNGKey(1), (S, 4, d))
        x2 = x1.at[5].set(jr.normal(jr.PRNGKey(2), (4, d)))
        out1 = block(x1, mask)
        out2 = block(x2, mask)
        assert jnp.allclose(out1[:5], out2[:5], atol=1e-5)
        assert not jnp.allclose(out1[5:], out2[5:])


class TestGrooveDecoder:

    @pytest.fixture(scope="class")
    def dec(self):
        return GrooveDecoder(ENTITY_DIM, KEY)

    def test_single_slot_shape(self, dec):
        ctx = jnp.ones(ENTITY_DIM)
        out = dec(ctx, 0)
        assert out.shape == (GROOVE_CONT_N,)

    def test_all_slots_shape(self, dec):
        ctx = jnp.ones(ENTITY_DIM)
        out = dec.all_slots(ctx)
        assert out.shape == (N_GROOVE_SLOTS, GROOVE_CONT_N)

    def test_encode_shape(self, dec):
        ctx = jnp.ones(ENTITY_DIM)
        latent = dec.encode(ctx, 0)
        assert latent.shape == (ENTITY_DIM,)

    def test_slots_differ(self, dec):
        ctx = jr.normal(KEY, (ENTITY_DIM,))
        out = dec.all_slots(ctx)
        # Different slots should give different outputs (slot embeddings distinguish them)
        assert not jnp.allclose(out[0], out[1])


class TestTableDecoder:

    @pytest.fixture(scope="class")
    def groove_dec(self):
        return GrooveDecoder(ENTITY_DIM, KEY)

    @pytest.fixture(scope="class")
    def trace_dec(self, groove_dec):
        return TableDecoder(ENTITY_DIM, is_trace=True, groove_decoder=groove_dec,
                            key=jr.PRNGKey(1))

    @pytest.fixture(scope="class")
    def table_dec(self, groove_dec, trace_dec):
        return TableDecoder(ENTITY_DIM, is_trace=False, groove_decoder=groove_dec,
                            key=jr.PRNGKey(2), sub_table_decoder=trace_dec,
                            _slot_embeds=trace_dec.slot_embeds,
                            _linear_in=trace_dec.linear_in,
                            _cat_out=trace_dec.cat_out,
                            _cont_out=trace_dec.cont_out)

    def test_trace_output_keys(self, trace_dec):
        out = trace_dec(jnp.ones(ENTITY_DIM))
        assert set(out.keys()) == {'cat', 'cont', 'grooves'}

    def test_table_output_keys(self, table_dec):
        out = table_dec(jnp.ones(ENTITY_DIM))
        assert set(out.keys()) == {'cat', 'cont', 'grooves', 'traces'}

    def test_trace_cat_shape(self, trace_dec):
        out = trace_dec(jnp.ones(ENTITY_DIM))
        assert out['cat'].shape == (TABLE_SCALAR_CAT_TOTAL_VOCAB,)

    def test_trace_cont_shape(self, trace_dec):
        out = trace_dec(jnp.ones(ENTITY_DIM))
        assert out['cont'].shape == (TABLE_SCALAR_CONT_N,)

    def test_trace_grooves_shape(self, trace_dec):
        out = trace_dec(jnp.ones(ENTITY_DIM))
        assert out['grooves'].shape == (N_GROOVE_SLOTS, GROOVE_CONT_N)

    def test_table_traces_cat_shape(self, table_dec):
        out = table_dec(jnp.ones(ENTITY_DIM))
        assert out['traces']['cat'].shape == (N_TABLE_SLOTS, TABLE_SCALAR_CAT_TOTAL_VOCAB)

    def test_table_traces_cont_shape(self, table_dec):
        out = table_dec(jnp.ones(ENTITY_DIM))
        assert out['traces']['cont'].shape == (N_TABLE_SLOTS, TABLE_SCALAR_CONT_N)

    def test_table_traces_grooves_shape(self, table_dec):
        out = table_dec(jnp.ones(ENTITY_DIM))
        assert out['traces']['grooves'].shape == (N_TABLE_SLOTS, N_GROOVE_SLOTS, GROOVE_CONT_N)

    def test_trace_masks_invalid_cmds(self, trace_dec):
        """A and H command logits should be -inf in trace cat output."""
        from pe_lsdj.models.transformer import _TABLE_SCALAR_CAT_GROUPS
        from pe_lsdj.constants import CMD_A, CMD_H
        out = trace_dec(jr.normal(KEY, (ENTITY_DIM,)))
        for vocab, starts, cols in _TABLE_SCALAR_CAT_GROUPS:
            if vocab == 19:
                for cmd in (CMD_A, CMD_H):
                    assert jnp.all(out['cat'][starts + cmd] == -jnp.inf), \
                        f"cmd {cmd} should be -inf in trace cat"

    def test_table_does_not_mask_cmds(self, table_dec):
        """A and H should NOT be -inf in top-level table cat output."""
        from pe_lsdj.models.transformer import _TABLE_SCALAR_CAT_GROUPS
        from pe_lsdj.constants import CMD_A, CMD_H
        out = table_dec(jr.normal(KEY, (ENTITY_DIM,)))
        for vocab, starts, cols in _TABLE_SCALAR_CAT_GROUPS:
            if vocab == 19:
                for cmd in (CMD_A, CMD_H):
                    assert jnp.all(jnp.isfinite(out['cat'][starts + cmd])), \
                        f"cmd {cmd} should be finite in table cat"

    def test_shared_weights(self, table_dec, trace_dec):
        """table_dec and trace_dec must share linear_in, cat_out, cont_out."""
        assert table_dec.linear_in is trace_dec.linear_in
        assert table_dec.cat_out   is trace_dec.cat_out
        assert table_dec.cont_out  is trace_dec.cont_out


class TestOutputHeads:

    @pytest.fixture(scope="class")
    def heads(self):
        return OutputHeads(D_MODEL, ENTITY_DIM, KEY)

    def test_token_head_keys(self, heads):
        x = jnp.ones(D_MODEL)
        out = heads(x)
        for name in TOKEN_HEADS:
            assert name in out

    def test_token_head_shapes(self, heads):
        x = jnp.ones(D_MODEL)
        out = heads(x)
        for name, (pos, vocab) in TOKEN_HEADS.items():
            assert out[name].shape == (vocab,), f"{name}: expected ({vocab},)"

    def test_entity_keys(self, heads):
        x = jnp.ones(D_MODEL)
        out = heads(x)
        assert 'instr' in out
        assert 'table' in out
        assert 'groove' in out

    def test_instr_keys(self, heads):
        x = jnp.ones(D_MODEL)
        out = heads(x)
        assert set(out['instr'].keys()) >= {'cat', 'cont', 'softsynth', 'table'}

    def test_instr_cat_shape(self, heads):
        out = heads(jnp.ones(D_MODEL))
        assert out['instr']['cat'].shape == (INSTR_SCALAR_CAT_TOTAL_VOCAB,)

    def test_instr_cont_shape(self, heads):
        out = heads(jnp.ones(D_MODEL))
        assert out['instr']['cont'].shape == (INSTR_SCALAR_CONT_N,)

    def test_instr_table_shape(self, heads):
        out = heads(jnp.ones(D_MODEL))
        t = out['instr']['table']
        assert t['cat'].shape    == (TABLE_SCALAR_CAT_TOTAL_VOCAB,)
        assert t['cont'].shape   == (TABLE_SCALAR_CONT_N,)
        assert t['grooves'].shape == (N_GROOVE_SLOTS, GROOVE_CONT_N)
        assert t['traces']['cat'].shape    == (N_TABLE_SLOTS, TABLE_SCALAR_CAT_TOTAL_VOCAB)
        assert t['traces']['cont'].shape   == (N_TABLE_SLOTS, TABLE_SCALAR_CONT_N)
        assert t['traces']['grooves'].shape == (N_TABLE_SLOTS, N_GROOVE_SLOTS, GROOVE_CONT_N)

    def test_softsynth_shapes(self, heads):
        out = heads(jnp.ones(D_MODEL))
        ss = out['instr']['softsynth']
        assert ss['cat'].shape       == (SOFTSYNTH_CAT_TOTAL_VOCAB,)
        assert ss['cont'].shape      == (SOFTSYNTH_CONT_N,)
        assert ss['waveframes'].shape == (WAVEFRAME_DIM,)

    def test_phrase_table_shape(self, heads):
        out = heads(jnp.ones(D_MODEL))
        t = out['table']
        assert t['cat'].shape    == (TABLE_SCALAR_CAT_TOTAL_VOCAB,)
        assert t['cont'].shape   == (TABLE_SCALAR_CONT_N,)
        assert t['grooves'].shape == (N_GROOVE_SLOTS, GROOVE_CONT_N)
        assert 'traces' in t

    def test_phrase_groove_shape(self, heads):
        out = heads(jnp.ones(D_MODEL))
        assert out['groove'].shape == (GROOVE_CONT_N,)

    def test_log_probs(self, heads):
        x = jr.normal(jr.PRNGKey(1), (D_MODEL,))
        lp = heads.log_probs(x)
        assert set(lp.keys()) == set(TOKEN_HEADS.keys())
        for name, logp in lp.items():
            assert jnp.allclose(jnp.exp(logp).sum(), 1.0, atol=1e-5), name

    def test_logit_group_weight_shapes(self, heads):
        for group_name, members in LOGIT_GROUPS.items():
            n    = len(members)
            vocab = members[0][2]
            assert heads.weights[group_name].shape == (n, vocab, D_MODEL), group_name


class TestHardTargetsAndLoss:

    def test_hard_targets_shapes(self):
        tokens = jnp.zeros(21, dtype=jnp.int32)
        targets = hard_targets(tokens)
        assert set(targets.keys()) == set(TOKEN_HEADS.keys())

    def test_hard_targets_one_hot(self):
        tokens = jnp.array([5] + [0] * 20, dtype=jnp.int32)
        targets = hard_targets(tokens)
        assert targets['note'][5] == 1.0
        assert targets['note'].sum() == 1.0

    def test_token_loss_finite(self):
        heads = OutputHeads(D_MODEL, ENTITY_DIM, KEY)
        x = jr.normal(jr.PRNGKey(1), (D_MODEL,))
        out = heads(x)
        tokens = jnp.zeros(21, dtype=jnp.int32)
        loss = token_loss(out, hard_targets(tokens))
        assert jnp.isfinite(loss)
        assert loss > 0

    def test_entity_loss_finite(self):
        heads  = OutputHeads(D_MODEL, ENTITY_DIM, KEY)
        x      = jr.normal(jr.PRNGKey(1), (D_MODEL,))
        out    = heads(x)
        entity_preds = {k: out[k] for k in ('instr', 'table', 'groove')}
        loss   = entity_loss(entity_preds, SongBanks.default(), jnp.zeros(21, dtype=jnp.int32))
        assert jnp.isfinite(loss)

    def test_entity_loss_null_tokens(self):
        heads  = OutputHeads(D_MODEL, ENTITY_DIM, KEY)
        x      = jr.normal(jr.PRNGKey(2), (D_MODEL,))
        out    = heads(x)
        entity_preds = {k: out[k] for k in ('instr', 'table', 'groove')}
        loss   = entity_loss(entity_preds, SongBanks.default(), jnp.zeros(21, dtype=jnp.int32))
        assert jnp.isfinite(loss)


class TestLSDJTransformer:

    @pytest.fixture(scope="class")
    def model(self):
        return LSDJTransformer(
            KEY, d_model=D_MODEL, entity_dim=ENTITY_DIM,
            num_heads_t=2, num_heads_c=2, num_blocks=2,
        )

    def test_output_token_keys(self, model):
        out = model(jnp.zeros((8, 4, 21)))
        for name in TOKEN_HEADS:
            assert name in out

    def test_output_entity_keys(self, model):
        out = model(jnp.zeros((8, 4, 21)))
        assert 'instr' in out
        assert 'table' in out
        assert 'groove' in out

    def test_token_head_shapes(self, model):
        S = 8
        out = model(jnp.zeros((S, 4, 21)))
        for name, (pos, vocab) in TOKEN_HEADS.items():
            assert out[name].shape == (S, 4, vocab), f"{name}: expected ({S},4,{vocab})"

    def test_instr_cat_shape(self, model):
        S = 8
        out = model(jnp.zeros((S, 4, 21)))
        assert out['instr']['cat'].shape == (S, 4, INSTR_SCALAR_CAT_TOTAL_VOCAB)

    def test_instr_table_grooves_shape(self, model):
        S = 8
        out = model(jnp.zeros((S, 4, 21)))
        assert out['instr']['table']['grooves'].shape == (S, 4, N_GROOVE_SLOTS, GROOVE_CONT_N)

    def test_instr_table_traces_shape(self, model):
        S = 8
        out = model(jnp.zeros((S, 4, 21)))
        traces = out['instr']['table']['traces']
        assert traces['cat'].shape    == (S, 4, N_TABLE_SLOTS, TABLE_SCALAR_CAT_TOTAL_VOCAB)
        assert traces['grooves'].shape == (S, 4, N_TABLE_SLOTS, N_GROOVE_SLOTS, GROOVE_CONT_N)

    def test_phrase_groove_shape(self, model):
        S = 8
        out = model(jnp.zeros((S, 4, 21)))
        assert out['groove'].shape == (S, 4, GROOVE_CONT_N)

    def test_waveframe_shape(self, model):
        S = 8
        out = model(jnp.zeros((S, 4, 21)))
        assert out['instr']['softsynth']['waveframes'].shape == (S, 4, WAVEFRAME_DIM)

    def test_default_banks(self):
        model = LSDJTransformer(jr.PRNGKey(42), d_model=D_MODEL, num_heads_t=2,
                                num_heads_c=1, num_blocks=1)
        out = model(jnp.zeros((4, 4, 21)))
        assert out['note'].shape == (4, 4, NUM_NOTES)

    def test_with_banks_no_crash(self, model):
        new_model = model.with_banks(SongBanks.default())
        out = new_model(jnp.zeros((4, 4, 21)))
        assert out['note'].shape == (4, 4, NUM_NOTES)
