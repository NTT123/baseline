from dataclasses import dataclass
from functools import partial
from typing import Callable

import torch
from torch import Tensor
from torch.nn import Embedding, Linear, Module, ModuleList
from torch.nn.functional import scaled_dot_product_attention


@dataclass
class ModelConfig:
    """Model hyper-parameters"""

    num_layers: int = 8
    hidden_dim: int = 1024
    intermediate_dim: int = 4096
    num_attention_heads: int = 16
    vocab_size: int = 256
    rms_norm_eps: float = 1.0e-6
    rope_theta: float = 10_000.0
    seq_length: int = 128

    @property
    def attention_head_dim(self):
        """Dimension per attention head"""
        assert self.hidden_dim % self.num_attention_heads == 0
        return self.hidden_dim // self.num_attention_heads


class RMSNorm(Module):
    """Normalizing by RMS norm"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.eps = config.rms_norm_eps

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def rotate_half(x):
    """shuffling for faster 2d rotation implementation"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, /, rotations, unsqueeze_dim=2):
    """rotate q, k vectors"""
    cos, sin = rotations.chunk(2, dim=0)
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotated_q = (q * cos) + (rotate_half(q) * sin)
    rotated_k = (k * cos) + (rotate_half(k) * sin)
    return rotated_q, rotated_k


class SelfAttention(Module):
    """Modeling interaction between tokens"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.q_proj = Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.k_proj = Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.v_proj = Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.o_proj = Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.num_heads = config.num_attention_heads
        self.head_dim = config.attention_head_dim
        self.hidden_dim = config.hidden_dim

    def forward(self, x: Tensor, rope: Callable) -> Tensor:
        N, L, _ = x.shape
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        q = q.view(N, L, self.num_heads, self.head_dim)
        k = k.view(N, L, self.num_heads, self.head_dim)
        v = v.view(N, L, self.num_heads, self.head_dim)
        q, k = rope(q, k, unsqueeze_dim=2)
        o = scaled_dot_product_attention(
            query=q.transpose(1, 2),
            key=k.transpose(1, 2),
            value=v.transpose(1, 2),
            is_causal=True,
            enable_gqa=True,
        ).transpose(1, 2)
        o = o.reshape(N, L, self.hidden_dim)
        return self.o_proj(o)


class MLP(Module):
    """Modeling computation at token-level"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.up_proj = Linear(config.hidden_dim, config.intermediate_dim, bias=False)
        self.down_proj = Linear(config.intermediate_dim, config.hidden_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.up_proj(x)
        x = x.relu().square()
        x = self.down_proj(x)
        return x


class Block(Module):
    """An unit of our transformer network which include a self-attention layer and a mlp layer"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config)
        self.attn_layer = SelfAttention(config)
        self.mlp_norm = RMSNorm(config)
        self.mlp_layer = MLP(config)

    def forward(self, x: Tensor, rope: Callable) -> Tensor:
        x = x + self.attn_layer(self.attn_norm(x), rope=rope)
        x = x + self.mlp_layer(self.mlp_norm(x))
        return x


@torch.no_grad()
def compute_rope_params(config: ModelConfig):
    "precompute RoPE parameters at each position"
    dim = config.attention_head_dim
    hidden_dim_index = torch.arange(0, dim, 2, dtype=torch.float64)
    angular_freqs = torch.pow(config.rope_theta, -hidden_dim_index / dim)
    position = torch.arange(0, config.seq_length, dtype=torch.float64)
    angles = position[:, None] * angular_freqs[None, :]
    angles2 = torch.cat((angles, angles), dim=-1)
    cos = angles2.cos()
    sin = angles2.sin()
    return torch.stack((cos, sin), dim=0).to(dtype=torch.float32)


class GPT(Module):
    """Decoder-only transformer network"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed = Embedding(config.vocab_size, config.hidden_dim)
        self.layers = ModuleList(Block(config) for _ in range(config.num_layers))
        self.unembed_norm = RMSNorm(config)
        self.unembed = Linear(config.hidden_dim, config.vocab_size, bias=False)
        rope_params = compute_rope_params(config)
        self.register_buffer("rope_params", rope_params, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        N, L = x.shape
        del N
        x = self.embed(x)
        rotations = self.rope_params[:, :L].to(dtype=x.dtype)
        rope = partial(apply_rotary_pos_emb, rotations=rotations)
        for block in self.layers:
            x = block(x, rope=rope)
        x = self.unembed_norm(x)
        logits = self.unembed(x)
        return logits

    def num_parameters(self):
        "Return number of trainable parameters, excluding embed and unembed layers"
        total = (
            sum([t.numel() for t in self.parameters()])
            - self.embed.weight.numel()
            - self.unembed.weight.numel()
        )
        return total


if __name__ == "__main__":
    _config = ModelConfig()
    gpt = GPT(_config)
    print(gpt)
    _x = torch.randint(0, 255, size=(3, 32))
    _y = gpt(_x)
    print(_y.shape)
