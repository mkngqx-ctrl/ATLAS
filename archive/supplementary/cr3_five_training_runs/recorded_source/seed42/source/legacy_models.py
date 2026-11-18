"""Existing Lite/DRSA/Weibull/Glazier architectures generalized to 3 inputs.
No old checkpoints are compatible with the new input or state dimensions.
Losses/training/converters live in model_training.py; class defaults remain
those of the prior benchmark unless overridden explicitly.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.shape[1], :]

class SurvTransformerCRModel(nn.Module):

    def __init__(self, d_model=32, nhead=4, num_layers=2, dim_feedforward=64, L=30, dropout=0.1):
        super().__init__()
        self.L = L
        self.input_proj = nn.Linear(4, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=L + 1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True, activation='relu')
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 2 * L + 1)

    def forward(self, x_std):
        batch = x_std.shape[0]
        device = x_std.device
        l_idx = torch.arange(1, self.L + 1, device=device, dtype=torch.float32) / self.L
        x_rep = x_std[:, None, :].expand(batch, self.L, 3)
        l_rep = l_idx.view(1, self.L, 1).expand(batch, self.L, 1)
        inp = torch.cat([x_rep, l_rep], dim=-1)
        h = self.input_proj(inp)
        h = self.pos_enc(h)
        out = self.encoder(h)
        pooled = out.mean(dim=1)
        logits = self.head(pooled)
        pmf = torch.log_softmax(logits, dim=-1)
        return pmf

class DynamicDeepHitCRModel(nn.Module):

    def __init__(self, hidden_size=32, num_layers=1, L=30, attn_size=16):
        super().__init__()
        self.L = L
        self.gru = nn.GRU(input_size=4, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.attn_W = nn.Linear(hidden_size, attn_size)
        self.attn_v = nn.Linear(attn_size, 1, bias=False)
        self.head = nn.Linear(hidden_size * 2, 2 * L + 1)

    def forward(self, x_std):
        batch = x_std.shape[0]
        device = x_std.device
        l_idx = torch.arange(1, self.L + 1, device=device, dtype=torch.float32) / self.L
        x_rep = x_std[:, None, :].expand(batch, self.L, 3)
        l_rep = l_idx.view(1, self.L, 1).expand(batch, self.L, 1)
        inp = torch.cat([x_rep, l_rep], dim=-1)
        (out, h_n) = self.gru(inp)
        last_hidden = h_n[-1]
        scores = self.attn_v(torch.tanh(self.attn_W(out)))
        weights = torch.softmax(scores, dim=1)
        context = (weights * out).sum(dim=1)
        combined = torch.cat([context, last_hidden], dim=-1)
        logits = self.head(combined)
        pmf = torch.log_softmax(logits, dim=-1)
        return (pmf, weights.squeeze(-1))

class DRSACRModel(nn.Module):

    def __init__(self, hidden_size=32, num_layers=1, L=30):
        super().__init__()
        self.L = L
        self.lstm = nn.LSTM(input_size=4, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, 3)

    def forward(self, x_std):
        batch = x_std.shape[0]
        device = x_std.device
        l_idx = torch.arange(1, self.L + 1, device=device, dtype=torch.float32) / self.L
        x_rep = x_std[:, None, :].expand(batch, self.L, 3)
        l_rep = l_idx.view(1, self.L, 1).expand(batch, self.L, 1)
        inp = torch.cat([x_rep, l_rep], dim=-1)
        (out, _) = self.lstm(inp)
        probs = torch.log_softmax(self.head(out), dim=-1)
        return probs

class WTTEModelCR(nn.Module):

    def __init__(self, hidden_size=32, init_alpha=15.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, hidden_size), nn.Tanh(), nn.Linear(hidden_size, hidden_size), nn.Tanh())
        self.head = nn.Linear(hidden_size, 4)
        with torch.no_grad():
            init_alpha = max(float(init_alpha), 2.0)
            log_init = np.log(np.exp(init_alpha) - 1)
            self.head.bias[0] = log_init
            self.head.bias[1] = 0.0
            self.head.bias[2] = log_init
            self.head.bias[3] = 0.0

    def forward(self, x_std):
        h = self.net(x_std)
        raw = self.head(h)
        eps = 0.0001
        alpha_d = torch.nn.functional.softplus(raw[:, 0]) + eps
        beta_d = torch.nn.functional.softplus(raw[:, 1]) + eps
        alpha_o = torch.nn.functional.softplus(raw[:, 2]) + eps
        beta_o = torch.nn.functional.softplus(raw[:, 3]) + eps
        return (alpha_d, beta_d, alpha_o, beta_o)

class GlazierModel(nn.Module):

    def __init__(self, hidden_size=32, num_layers=2, embed_size=8, n_states=10):
        super().__init__()
        self.n_states = n_states
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.START_TOKEN = n_states
        self.embed = nn.Embedding(n_states + 1, embed_size)
        self.gru = nn.GRU(input_size=embed_size + 3, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, n_states)

    def forward_teacher_forced(self, input_states, x_std):
        (batch, T) = input_states.shape
        emb = self.embed(input_states)
        x_rep = x_std[:, None, :].expand(batch, T, 3)
        inp = torch.cat([emb, x_rep], dim=-1)
        (out, _) = self.gru(inp)
        return self.head(out)

    def step(self, prev_state, x_std, hidden):
        emb = self.embed(prev_state).unsqueeze(1)
        x_rep = x_std[:, None, :]
        inp = torch.cat([emb, x_rep], dim=-1)
        (out, hidden) = self.gru(inp, hidden)
        logits = self.head(out.squeeze(1))
        return (logits, hidden)

class CausalSelfAttention(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0, 'd_model must be divisible by nhead'
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.d_model = d_model
        self.dropout_p = dropout
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        (batch, T, _) = x.shape
        qkv = self.qkv_proj(x)
        (q, k, v) = qkv.chunk(3, dim=-1)
        q = q.view(batch, T, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(batch, T, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(batch, T, self.nhead, self.head_dim).transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_p if self.training else 0.0, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, T, self.d_model)
        return self.out_proj(attn_out)

class CausalTransformerBlock(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, nhead, dropout=dropout)
        self.drop1 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, dim_feedforward), nn.ReLU(), nn.Linear(dim_feedforward, d_model))
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop1(self.attn(self.ln1(x)))
        x = x + self.drop2(self.mlp(self.ln2(x)))
        return x

class GlazierTransformerModel(nn.Module):

    def __init__(self, d_model=32, nhead=4, num_layers=2, dim_feedforward=64, n_states=10, max_len=31, dropout=0.1):
        super().__init__()
        self.n_states = n_states
        self.d_model = d_model
        self.START_TOKEN = n_states
        self.embed = nn.Embedding(n_states + 1, d_model)
        self.covariate_proj = nn.Linear(3, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)
        self.blocks = nn.ModuleList([CausalTransformerBlock(d_model, nhead, dim_feedforward, dropout=dropout) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_states)

    def _encode(self, input_states, x_std):
        (batch, T) = input_states.shape
        emb = self.embed(input_states)
        x_rep = self.covariate_proj(x_std[:, None, :]).expand(batch, T, self.d_model)
        h = emb + x_rep
        h = self.pos_enc(h)
        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)
        return h

    def forward_teacher_forced(self, input_states, x_std):
        out = self._encode(input_states, x_std)
        return self.head(out)

    def next_state_logits(self, generated_so_far, x_std):
        out = self._encode(generated_so_far, x_std)
        return self.head(out[:, -1, :])
