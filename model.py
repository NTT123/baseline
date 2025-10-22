import torch
from torch import Tensor
from torch.nn import Module, Parameter, Linear, Embedding, ModuleList
from dataclasses import dataclass
from torch.nn.functional import scaled_dot_product_attention

@dataclass
class ModelConfig:
    num_layers: int = 8
    hidden_dim: int = 1024
    intermediate_dim: int = 4096
    num_attention_heads: int = 16
    vocab_size: int = 256
    rms_norm_eps: float = 1.0e-6
    rope_theta: float = 10000.0
    seq_length: int = 1024


class RMSNorm(Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.eps = config.rms_norm_eps
        self.weight = Parameter(torch.zeros(config.hidden_dim))

    def _norm(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        return self._norm(x) * self.weight


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, pos_cos_sin, unsqueeze_dim=2):
    cos, sin = pos_cos_sin.chunk(2, dim=0)
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SelfAttention(Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.qkv = Linear(config.hidden_dim, config.hidden_dim*3, bias=False)
        self.o = Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.num_heads = config.num_attention_heads
        assert config.hidden_dim % self.num_heads == 0
        self.head_dim = config.hidden_dim // self.num_heads
        self.hidden_dim = config.hidden_dim

    def forward(self, x: Tensor, pos: Tensor) -> Tensor:
        N, L, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(N, L, self.num_heads, self.head_dim)
        k = k.view(N, L, self.num_heads, self.head_dim)
        v = v.view(N, L, self.num_heads, self.head_dim)
        q, k = apply_rotary_pos_emb(q, k, pos, unsqueeze_dim=2)
        o = scaled_dot_product_attention(
            query=q.transpose(1, 2),
            key=k.transpose(1, 2),
            value=v.transpose(1, 2),
            is_causal=True,
            enable_gqa=True,
        ).transpose(1, 2)
        o = o.reshape(N, L, self.hidden_dim)
        return self.o(o)
        


class MLP(Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.fc1 = Linear(config.hidden_dim, config.intermediate_dim, bias=False)
        self.fc2 = Linear(config.intermediate_dim, config.hidden_dim, bias=False)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = x.relu().square()
        x = self.fc2(x)
        return x

class Block(Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config)
        self.attn_layer = SelfAttention(config)
        self.mlp_norm = RMSNorm(config)
        self.mlp_layer = MLP(config)

    def forward(self, x: Tensor, pos: Tensor) -> Tensor:
        x = x + self.attn_layer(self.attn_norm(x), pos=pos)
        x = x + self.mlp_layer(self.mlp_norm(x))
        return x

@torch.no_grad()
def compute_pos_sin_cos(config: ModelConfig):
    dim = config.hidden_dim // config.num_attention_heads
    inv_freq = torch.pow(config.rope_theta, -torch.arange(0, dim, 2, dtype=torch.float64) / dim)
    inv_freq = inv_freq[None, :, None]
    inv_freq = inv_freq.expand(1, -1, 1)
    pos = torch.arange(0, config.seq_length, dtype=torch.float64)[None, None, :]
    freqs = (inv_freq @ pos).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()
    sin = emb.sin()
    return torch.concat((cos, sin), dim=0)



class GPT(Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed = Embedding(config.vocab_size, config.hidden_dim)
        self.layers = ModuleList(Block(config) for _ in range(config.num_layers))
        self.unembed = Linear(config.hidden_dim, config.vocab_size, bias=False)
        self.register_buffer("pos", compute_pos_sin_cos(config), persistent=False)
        print(self.pos.shape)


    def forward(self, x: Tensor) -> Tensor:
        N, L = x.shape
        x = self.embed(x)
        pos = self.pos[:, :L].to(dtype=x.dtype)
        for block in self.layers:
            x = block(x, pos)
        logits = self.unembed(x)
        return logits
    

if __name__ == "__main__":
    config = ModelConfig()
    gpt = GPT(config)
    print(gpt)
    x = torch.randint(0, 255, size=(3, 32))
    y = gpt(x)
    print(y.shape)