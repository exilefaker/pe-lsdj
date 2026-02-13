import torch
import numpy as np
import torch.nn as nn

from pylsdj.bread_spec import (
    STEPS_PER_PHRASE, 
    PHRASES_PER_CHAIN, 
    NUM_SONG_CHAINS, 
    NUM_INSTRUMENTS,
    NOTES,
    FX_COMMANDS
)

D_IN = 128
D_OUT_KQ = 128
D_OUT_V = 128
NUM_STEPS = 16
NUM_CHANNELS = 4

SEQ_LEN = NUM_SONG_CHAINS * PHRASES_PER_CHAIN


def get_positional_encoding(seq_len, d, n=10000):
    P = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d/2)):
            denominator = np.power(n, 2*i/d)
            P[k, 2*i] = np.sin(k/denominator)
            P[k, 2*i+1] = np.cos(k/denominator)
    return torch.Tensor(P)


def get_channel_weights(channel_width, bias):
    print()
    t = torch.Tensor(
        np.kron(np.eye(channel_width), np.ones((channel_width,channel_width))*bias)
    )
    return torch.Tensor(
        np.kron(np.eye(channel_width), np.ones((channel_width,channel_width))*bias)
    )
 
 
class MultinomialLogisticRegression(nn.Module): 
    def __init__(self, input_size, num_classes): 
        super(MultinomialLogisticRegression, self).__init__() 
        self.linear = nn.Linear(input_size, num_classes) 
  
    def forward(self, x):
        # print("x shape", x.shape)
        out = self.linear(x)
        # print("out.shape", out.shape)
        out = nn.functional.softmax(out, dim=0) 
        return out 


class WeightedTemporalSelfAttention(nn.Module):
    def __init__(
        self, 
        d_in: int, 
        d_out_kq: int, 
        d_out_v: int,
        d_batch: int,
        same_channel_bias: float,
        channel_width: int,
        num_channels: int,
        T: int,
        num_steps: int,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_out_kq = d_out_kq
        self.d_out_v = d_out_v
        self.d_batch = d_batch
        self.W_query = nn.Parameter(torch.rand(d_in, d_out_kq))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out_kq))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out_v))
        self.channel_width = channel_width
        self.num_channels = num_channels
        self.same_channel_bias = same_channel_bias
        self.T = T
        self.t = 0
        self.num_steps = num_steps
        self.step = 0

        self.b_channel = nn.Parameter(torch.rand(d_batch, d_out_v))
        self.b_t = nn.Parameter(torch.rand(num_steps, d_batch, d_out_v))

        self.pos = get_positional_encoding(T, d_out_v, d_batch)

    def forward(self, x=None):
        # print("input shape", x.shape)
        if x is not None:
            # Generate K, Q, V
            keys = x @ self.W_key
            queries = x @ self.W_query
            values = x @ self.W_value

            # print("keys shape", keys.shape)
            # print("queries shape", queries.shape)
            # print("values shape", values.shape)
        
            attn_scores = queries @ keys.T  # Unnormalized attention weights
            attn_scores += get_channel_weights(self.channel_width, self.same_channel_bias) # Add same-channel bias

            # Normalize
            attn_weights = torch.softmax(
                attn_scores / self.d_out_kq**0.5, dim=-1
            )

            # print("attention weights shape", attn_weights.shape)

            # Compute values
            context_vec = attn_weights @ values
            # print("cv shape", context_vec.shape)

        else:
            # if no data, just use biases
            context_vec = torch.zeros(self.d_batch, self.d_out_v)

        # print("b_channel shape", self.b_channel.shape)
        # print("b_step shape", self.b_t[self.step].shape)
        pos = get_positional_encoding(self.d_in, self.d_out_v)
        # print("pos shape", pos[self.t].shape)

        context_vec += self.b_channel + self.b_t[self.step] + self.pos[self.t]

        # Increment time on step + global scales
        self.step += 1
        self.t += 1

        # Reset step counter
        if self.step == self.num_steps:
            self.step = 0

        if self.t > self.T:
            raise ValueError("Maximum global sequence length exceeded")
        
        return context_vec


class LSDJNet(nn.Module):
    # TODO Add chain-level transposes
    def __init__(
        self,
        embedding_dim=D_IN,
        kq_dim=D_OUT_KQ,
        same_channel_bias=2.0,
    ):
        super(LSDJNet, self).__init__()

        self.CHANNELS = ["P1", "P2", "W", "N"]
        NUM_CHANNELS = len(self.CHANNELS)
        CHANNEL_WIDTH = 4
        NUM_STEPS = 16

        self.attention = WeightedTemporalSelfAttention(
            d_in=embedding_dim, 
            d_out_kq=kq_dim, 
            d_out_v=embedding_dim, 
            d_batch=NUM_CHANNELS * CHANNEL_WIDTH,
            same_channel_bias=same_channel_bias,  
            channel_width=CHANNEL_WIDTH, 
            num_channels=NUM_CHANNELS, 
            T=NUM_SONG_CHAINS * PHRASES_PER_CHAIN,
            num_steps=NUM_STEPS
        )
        
        self.mnlrs = {
            c: {
                "notes": MultinomialLogisticRegression(
                    embedding_dim,
                    len(NOTES) # TODO: Compute the value and just set as a constant in the model
                ),
                "instruments": MultinomialLogisticRegression(
                    embedding_dim,
                    NUM_INSTRUMENTS
                ),
                "commands": MultinomialLogisticRegression(
                    embedding_dim,
                    len(FX_COMMANDS)
                ),
            }
            for c in self.CHANNELS
        }
        self.mnlrs_list = []
        #TODO: Maybe a more elegant way exists to do this
        for c in self.CHANNELS:
            mnlrs = self.mnlrs[c]
            self.mnlrs_list += [mnlrs["notes"], mnlrs["instruments"], mnlrs["commands"]]
                
        # TODO: Handle instruments intelligently
        # TODO: Handle command values at all
    
    def forward(self, x=None):
        contextual_embeddings = self.attention(x)
        self.embedding = contextual_embeddings
        print("CE shape", contextual_embeddings.shape)

        out = [m(contextual_embeddings[idx]) for idx, m in enumerate(self.mnlrs_list)]
        print("step", self.attention.step)
        print("global time", self.attention.t)
        return out





if __name__ == "__main__":

    # BATCH_SIZE=16

    EMBEDDING_DIM=4
    D_KQ=4
    SAME_CHANNEL_BIAS=2.0

    # CHANNEL_WIDTH
    

    # attn = WeightedTemporalSelfAttention(
    #     d_in=4,
    #     d_out_kq=4,
    #     d_out_v=EMBEDDING_DIM,
    #     d_batch=BATCH_SIZE,
    #     same_channel_bias=2.0,
    #     channel_width=4,
    #     num_channels=4,
    #     T=SEQ_LEN,
    #     num_steps=16,
    # )

    # emb = torch.Tensor(np.random.rand(BATCH_SIZE, EMBEDDING_DIM))

    # r = attn(emb)
    # print("Result", r)

    # base_embedding = attn()
    # print("Base embedding", base_embedding)

    # # TODO: Define full model / integrate MNLR

    # # TODO: Import data

    lsdj_net = LSDJNet(
        embedding_dim=EMBEDDING_DIM,
        kq_dim=D_KQ,
        same_channel_bias=SAME_CHANNEL_BIAS,
    )

    first_output = lsdj_net()
    print("Initial output", first_output)

    initial_embedding = lsdj_net.embedding

    second_output = lsdj_net(initial_embedding)
    print("Next output", second_output)

    # Define cross-entropy loss (MNLR output is categorical)
    loss = nn.CrossEntropyLoss()  

    lr = 1e-2
    optimizer = torch.optim.Adam(lsdj_net.parameters(), lr=lr)

