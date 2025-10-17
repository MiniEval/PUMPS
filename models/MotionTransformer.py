import copy
import random

import torch
import torch.nn.functional as F
from torch import nn

from models.modules import RMSNorm, PositionalEmbedding


class RotaryAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0, max_length=256):
        """
        Initialize the RoPE Attention module.

        Args:
            embed_dim (int): The dimensionality of the input features.
            num_heads (int): Number of attention heads.
        """
        super(RotaryAttention, self).__init__()
        assert d_model % n_heads == 0, "embed_dim must be divisible by num_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout

        # Define the query, key, and value projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        theta = torch.arange(self.head_dim // 2, dtype=torch.float32)
        theta = 1.0 / (10000 ** (2.0 * theta / self.head_dim))
        seq_range = torch.arange(max_length, dtype=torch.float32)
        position = seq_range[:, None, None, None] * theta # [sequence, 1, 1, head_dim // 2]

        sin_pos = torch.sin(position)
        self.register_buffer('sin_pos', sin_pos)
        cos_pos = torch.cos(position)
        self.register_buffer('cos_pos', cos_pos)

    def apply_rope(self, tensor):
        """
        Apply Rotary Position Embeddings (RoPE) to a tensor.

        Args:
            tensor (torch.Tensor): Input tensor of shape [sequence, batch, num_heads, head_dim].

        Returns:
            torch.Tensor: Tensor after applying RoPE.
        """

        seq_len = tensor.shape[0]
        sin = self.sin_pos[:seq_len].clone()
        cos = self.cos_pos[:seq_len].clone()
        q1 = tensor[..., :self.head_dim // 2]
        q2 = tensor[..., self.head_dim // 2:]
        t1 = q1 * cos - q2 * sin
        t2 = q1 * sin - q2 * cos
        return torch.cat([t1, t2], dim=-1)

    def forward(self, query, key, value, attn_mask=None, need_weights=False):
        """
        Forward pass for RoPE Attention.

        Args:
            x (torch.Tensor): Input tensor of shape [sequence, batch, features].

        Returns:
            torch.Tensor: Output tensor of shape [sequence, batch, features].
        """
        q_len, batches, _ = query.size()
        kv_len = key.shape[0]

        # Project inputs to queries, keys, and values
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        q = torch.reshape(q, (q_len, batches, self.n_heads, self.head_dim))
        k = torch.reshape(k, (kv_len, batches, self.n_heads, self.head_dim))
        v = torch.reshape(v, (kv_len, batches, self.n_heads, self.head_dim))

        # Apply RoPE to queries and keys
        q = self.apply_rope(q)
        k = self.apply_rope(k)
        q = q.permute(1, 2, 0, 3)
        k = k.permute(1, 2, 0, 3)
        v = v.permute(1, 2, 0, 3)

        # Compute scaled dot-product attention
        attn = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout)
        attn = attn.permute(2, 0, 1, 3)
        attn = torch.reshape(attn, (q_len, batches, -1))
        attn = self.out_proj(attn)

        return attn, None


class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1, activation=None, norm=None, rotary=False):
        super(SelfAttentionLayer, self).__init__()
        if d_ff is None:
            d_ff = d_model * 4

        if rotary:
            self.attn = RotaryAttention(d_model, n_heads, dropout=dropout)
        else:
            self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)

        if norm is None:
            self.norm = nn.LayerNorm(d_model)
        else:
            self.norm = norm

        self.ff_in = nn.Linear(d_model, d_ff)
        if activation is None:
            self.activation = nn.ReLU()
        else:
            self.activation = activation
        self.ff_norm = copy.deepcopy(self.norm)
        self.ff_out = nn.Linear(d_ff, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, attn_mask=None):
        x = q
        x_, _ = self.attn(q, q, q, attn_mask=attn_mask, need_weights=False)
        x = x + x_
        x = self.norm(x)

        x_ = self.ff_in(x)
        x_ = self.activation(x_)
        x_ = self.ff_out(x_)
        x_ = self.dropout(x_)
        x = x + x_
        x = self.ff_norm(x)

        return x


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1, activation=None, norm=None, use_self_attn=True, rotary=False):
        super(CrossAttentionLayer, self).__init__()
        self.use_self_attn = use_self_attn
        if d_ff is None:
            d_ff = d_model * 4

        if rotary:
            self.cross_attn = RotaryAttention(d_model, n_heads, dropout=dropout)
        else:
            self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)

        if norm is None:
            self.cross_norm = nn.LayerNorm(d_model)
        else:
            self.cross_norm = norm

        if use_self_attn:
            if rotary:
                self.self_attn = RotaryAttention(d_model, n_heads, dropout=dropout)
            else:
                self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
            self.self_norm = copy.deepcopy(self.cross_norm)

        self.ff_in = nn.Linear(d_model, d_ff)
        if activation is None:
            self.activation = nn.ReLU()
        else:
            self.activation = activation
        self.ff_norm = copy.deepcopy(self.cross_norm)
        self.ff_out = nn.Linear(d_ff, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, self_attn_mask=None, cross_attn_mask=None):
        x = q
        if self.use_self_attn:
            x_, _ = self.self_attn(q, q, q, attn_mask=self_attn_mask, need_weights=False)
            x = x + x_
            x = self.self_norm(x)

        x_, _ = self.cross_attn(q, k, v, attn_mask=cross_attn_mask, need_weights=False)
        x = x + x_
        x = self.cross_norm(x)

        x_ = self.ff_in(x)
        x_ = self.activation(x_)
        x_ = self.ff_out(x_)
        x_ = self.dropout(x_)
        x = x + x_
        x = self.ff_norm(x)

        return x


class MotionTransformer(nn.Module):
    def __init__(self,
                 d_input,
                 d_model,
                 d_output,
                 n_heads=8,
                 n_layers=8,
                 max_length=256):
        super(MotionTransformer, self).__init__()
        self.d_input = d_input
        self.d_model = d_model
        self.d_output = d_output
        self.n_heads = n_heads

        self.embed_in = nn.Linear(d_input, d_model)

        self.encoder = nn.ModuleList([SelfAttentionLayer(d_model, n_heads, rotary=True,
                                                         dropout=0, activation=nn.GELU(), norm=RMSNorm(d_model))
                                      for _ in range(n_layers)])
        self.interm = nn.ModuleList([CrossAttentionLayer(d_model, n_heads, rotary=True, use_self_attn=False,
                                                         dropout=0, activation=nn.GELU(), norm=RMSNorm(d_model))
                                     for _ in range(n_layers)])
        self.interm_init = nn.Parameter(torch.zeros(d_model))
        nn.init.uniform_(self.interm_init)

        self.key_ff_in = nn.Linear(d_model, d_model * 4)
        self.key_ff_activation = nn.GELU()
        self.key_ff_out = nn.Linear(d_model * 4, d_model)

        self.decoder = nn.ModuleList([CrossAttentionLayer(d_model, n_heads, rotary=True,
                                                          dropout=0, activation=nn.GELU(), norm=RMSNorm(d_model))
                                      for _ in range(n_layers)])
        self.embed_out = nn.Linear(d_model, d_output)

    def forward(self, embeds, offsets, key_mask=None):
        # embeds: [batch, frames, data]
        # offsets: [batch, frames, 3]
        # key_mask: [batch, frames], keyframe = True. If not provided, assume encoding-only forward pass
        batches = embeds.shape[0]
        frames = embeds.shape[1]

        use_interm = (key_mask is not None)

        if key_mask is None:
            key_mask = torch.ones(batches, frames, dtype=torch.bool, device=embeds.shape)
        
        self_attn_mask = torch.repeat_interleave(key_mask[:, None, None, :], frames, -2)
        embed_x = self.embed_in(torch.cat([embeds, offsets], dim=-1))
        embed_x = embed_x * key_mask.unsqueeze(-1) # prevent data leaks
        embed_x = embed_x.transpose(0, 1)
        for block in self.encoder:
            embed_x = block(embed_x, attn_mask=~self_attn_mask)

        key_x = self.key_ff_in(embed_x)
        key_x = self.key_ff_activation(key_x)
        key_x = self.key_ff_out(key_x)

        if use_interm:
            interm_x = torch.zeros((frames, batches, self.d_model), device=embed_x.device)
            interm_x = interm_x + self.interm_init

            for block in self.interm:
                interm_x = block(interm_x, embed_x, embed_x, self_attn_mask=self_attn_mask,
                                 cross_attn_mask=~self_attn_mask)
            decoder_x = torch.where(key_mask.unsqueeze(-1).transpose(0, 1).expand(-1, -1, self.d_model), key_x, interm_x)
        else:
            decoder_x = key_x

        for block in self.decoder:
            decoder_x = block(decoder_x, embed_x, embed_x, cross_attn_mask=~self_attn_mask)

        out = self.embed_out(decoder_x).transpose(0, 1)
        out_embeds = out[..., :-3]
        out_offset = out[..., -3:]
        
        return out_embeds, out_offset
