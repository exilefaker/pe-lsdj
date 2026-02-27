import pytest
import jax
import jax.numpy as jnp
import jax.random as jr

from pe_lsdj.models.transformer import (
    AxialTransformerBlock,
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
    cond_entity_scan_loss,
)
from pe_lsdj.models.decoders import GrooveDecoder
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

    def test_encode_shape(self, dec):
        ctx = jnp.ones(ENTITY_DIM)
        latent = dec.encode(ctx, 0)
        assert latent.shape == (ENTITY_DIM,)

    def test_phrase_slot_shape(self, dec):
        """Slot index N_GROOVE_SLOTS is the phrase-level slot."""
        ctx = jnp.ones(ENTITY_DIM)
        out = dec(ctx, N_GROOVE_SLOTS)
        assert out.shape == (GROOVE_CONT_N,)

    def test_phrase_slot_differs_from_table_slots(self, dec):
        ctx = jr.normal(KEY, (ENTITY_DIM,))
        phrase_out = dec(ctx, N_GROOVE_SLOTS)
        slot0_out  = dec(ctx, 0)
        assert not jnp.allclose(phrase_out, slot0_out)


class TestTableDecoder:

    @pytest.fixture(scope="class")
    def table_dec(self):
        return TableDecoder(ENTITY_DIM, key=jr.PRNGKey(1))

    def test_output_keys(self, table_dec):
        out = table_dec(jnp.ones(ENTITY_DIM))
        assert set(out.keys()) == {'cat', 'cont'}

    def test_cat_shape(self, table_dec):
        out = table_dec(jnp.ones(ENTITY_DIM))
        assert out['cat'].shape == (TABLE_SCALAR_CAT_TOTAL_VOCAB,)

    def test_cont_shape(self, table_dec):
        out = table_dec(jnp.ones(ENTITY_DIM))
        assert out['cont'].shape == (TABLE_SCALAR_CONT_N,)

    def test_slot_embeds_shape(self, table_dec):
        assert table_dec.slot_embeds.shape == (N_TABLE_SLOTS, ENTITY_DIM)


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
        assert set(t.keys()) == {'cat', 'cont'}
        assert t['cat'].shape  == (TABLE_SCALAR_CAT_TOTAL_VOCAB,)
        assert t['cont'].shape == (TABLE_SCALAR_CONT_N,)

    def test_softsynth_shapes(self, heads):
        out = heads(jnp.ones(D_MODEL))
        ss = out['instr']['softsynth']
        assert ss['cat'].shape       == (SOFTSYNTH_CAT_TOTAL_VOCAB,)
        assert ss['cont'].shape      == (SOFTSYNTH_CONT_N,)
        assert ss['waveframes'].shape == (WAVEFRAME_DIM,)

    def test_phrase_table_shape(self, heads):
        out = heads(jnp.ones(D_MODEL))
        t = out['table']
        assert set(t.keys()) == {'cat', 'cont'}
        assert t['cat'].shape  == (TABLE_SCALAR_CAT_TOTAL_VOCAB,)
        assert t['cont'].shape == (TABLE_SCALAR_CONT_N,)

    def test_phrase_groove_shape(self, heads):
        out = heads(jnp.ones(D_MODEL))
        assert out['groove'].shape == (GROOVE_CONT_N,)

    def test_phrase_groove_uses_shared_decoder(self, heads):
        """Phrase groove must route through the shared GrooveDecoder (phrase slot)."""
        assert hasattr(heads, 'phrase_groove_proj')
        assert heads.groove_decoder.slot_embeds.shape == (N_GROOVE_SLOTS + 1, ENTITY_DIM)

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

    def test_soft_targets_lower_loss(self):
        heads = OutputHeads(D_MODEL, ENTITY_DIM, KEY)
        x = jr.normal(jr.PRNGKey(1), (D_MODEL,))
        out = heads(x)
        soft = {name: jax.nn.softmax(out[name]) for name in TOKEN_HEADS}
        assert token_loss(out, soft) < token_loss(out, hard_targets(jnp.zeros(21, dtype=jnp.int32)))

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

    def test_cond_entity_scan_loss_finite(self):
        """cond_entity_scan_loss should run and return a finite scalar."""
        model  = LSDJTransformer(KEY, d_model=D_MODEL, entity_dim=ENTITY_DIM,
                                 num_heads_t=2, num_heads_c=2, num_blocks=1)
        L = 4
        tokens  = jnp.zeros((L, 4, 21), dtype=jnp.float32)
        hiddens = model.encode(tokens)                  # (L, 4, D_MODEL)
        loss    = cond_entity_scan_loss(
            model.output_heads, hiddens, tokens, SongBanks.default()
        )
        assert jnp.isfinite(loss)

    def test_cond_entity_scan_loss_zero_for_null_banks(self):
        """With all-null tokens, all groove/trace ids are 0 — cond loss should be 0."""
        model  = LSDJTransformer(KEY, d_model=D_MODEL, entity_dim=ENTITY_DIM,
                                 num_heads_t=2, num_heads_c=2, num_blocks=1)
        L = 4
        tokens  = jnp.zeros((L, 4, 21), dtype=jnp.float32)
        hiddens = model.encode(tokens)
        loss    = cond_entity_scan_loss(
            model.output_heads, hiddens, tokens, SongBanks.default()
        )
        # Default banks have null (all-zero) rows; groove/trace ids in null rows are 0
        assert loss == 0.0


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

    def test_instr_table_shape(self, model):
        S = 8
        out = model(jnp.zeros((S, 4, 21)))
        t = out['instr']['table']
        assert set(t.keys()) == {'cat', 'cont'}
        assert t['cat'].shape  == (S, 4, TABLE_SCALAR_CAT_TOTAL_VOCAB)
        assert t['cont'].shape == (S, 4, TABLE_SCALAR_CONT_N)

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

    def test_encode_shape(self, model):
        hiddens = model.encode(jnp.zeros((8, 4, 21)))
        assert hiddens.shape == (8, 4, D_MODEL)
