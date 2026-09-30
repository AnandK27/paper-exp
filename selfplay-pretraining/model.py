"""
nanoGPT's model.py, converted to the Llama-style decoder the paper uses for both learner and generator:
RMSNorm, rotary embeddings, SwiGLU MLP, no biases, untied input/output embeddings, byte vocab (256).

Attention uses F.scaled_dot_product_attention by default. Forward-mode AD (used for the generator
reward, eq. 2) is not supported by every SDPA backend, so `attn_impl="manual"` switches to an explicit
softmax(QK^T)V path that torch.func.jvp can always differentiate.
"""
import inspect
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    block_size: int = 4096
    vocab_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    mlp_hidden: int = 0  # 0 -> ~8/3 * n_embd rounded up to a multiple of 64
    rope_theta: float = 10000.0


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x) * self.weight


def rope_cache(T, head_dim, theta, device):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    freqs = torch.outer(torch.arange(T, device=device).float(), inv)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    # x: (B, H, T, D); rotate interleaved pairs
    x1, x2 = x[..., 0::2].float(), x[..., 1::2].float()
    cos, sin = cos[None, None], sin[None, None]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2).type_as(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.attn_impl = "sdpa"

    def forward(self, x, cos, sin):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if self.attn_impl == "sdpa":
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
            att = att.masked_fill(~mask, float("-inf"))
            y = F.softmax(att.float(), dim=-1).type_as(v) @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        h = config.mlp_hidden or 64 * math.ceil(8 * config.n_embd / 3 / 64)
        self.w1 = nn.Linear(config.n_embd, h, bias=False)
        self.w3 = nn.Linear(config.n_embd, h, bias=False)
        self.c_proj = nn.Linear(h, config.n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln_1(x), cos, sin)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=RMSNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
        self._rope = {}

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def set_attn_impl(self, impl):
        for blk in self.transformer.h:
            blk.attn.attn_impl = impl

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def flops_per_token(self, T, backward=True):
        """nanoGPT's estimate: 6N + 12*L*d*T per token for fwd+bwd (N excludes the embedding lookup);
        a third of that for a forward pass."""
        N = self.get_num_params() - self.transformer.wte.weight.numel()
        fb = 6 * N + 12 * self.config.n_layer * self.config.n_embd * T
        return fb if backward else fb / 3

    def rope(self, T, device):
        key = (T, str(device))
        if key not in self._rope:
            self._rope[key] = rope_cache(T, self.config.n_embd // self.config.n_head, self.config.rope_theta, device)
        return self._rope[key]

    def forward(self, idx):
        """Returns logits at every position, (B, T, vocab)."""
        B, T = idx.size()
        assert T <= self.config.block_size, f"sequence of length {T} > block size {self.config.block_size}"
        cos, sin = self.rope(T, idx.device)
        x = self.transformer.wte(idx)
        for block in self.transformer.h:
            x = block(x, cos, sin)
        return self.lm_head(self.transformer.ln_f(x))

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        params = [p for p in self.parameters() if p.requires_grad]
        decay = [p for p in params if p.dim() >= 2]
        nodecay = [p for p in params if p.dim() < 2]
        groups = [{"params": decay, "weight_decay": weight_decay}, {"params": nodecay, "weight_decay": 0.0}]
        fused = "fused" in inspect.signature(torch.optim.AdamW).parameters and device_type == "cuda"
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas, **({"fused": True} if fused else {}))
