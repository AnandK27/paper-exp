"""
Use karpathy/nanochat's GPT (the maintained successor of nanoGPT) as the self-play learner.

- The model is built exactly as nanochat's scripts/base_train.py builds it for a given --depth
  (meta device -> to_empty -> init_weights), so its state_dict loads straight into base_train.
- nanochat uses a byte-level BPE tokenizer, which has one token per raw byte. Machine outputs (bytes) are
  fed as those single-byte tokens, and every sequence starts with <|bos|>, as nanochat's data does.
- FLOPs are counted with nanochat's own convention (6 * matmul params + 12*h*q*effective_seq per token,
  with sliding windows), at the actual (trimmed) sequence length.
- nanochat attention goes through FlashAttention-3 / SDPA, which has no forward-mode AD. set_attn_impl("manual")
  swaps in an explicit causal + sliding-window softmax attention for the JVP reward pass.
- Self-play trains this learner with AdamW (the paper's reward uses the AdamW second moment); nanochat's own
  Muon recipe takes over afterwards in base_train.
"""
import math
import os
import sys
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ManualAttn:
    @staticmethod
    def flash_attn_func(q, k, v, causal=True, window_size=(-1, -1)):
        B, T, H, D = q.shape
        if k.size(2) != H:  # GQA
            r = H // k.size(2)
            k, v = k.repeat_interleave(r, 2), v.repeat_interleave(r, 2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(D))
        i = torch.arange(T, device=q.device)
        mask = i[None, :] <= i[:, None]
        if window_size[0] is not None and window_size[0] >= 0:
            mask = mask & (i[None, :] >= i[:, None] - window_size[0])
        att = att.masked_fill(~mask, float("-inf"))
        y = F.softmax(att.float(), dim=-1).type_as(v) @ v
        return y.transpose(1, 2)


class NanochatLearner(nn.Module):
    def __init__(self, nanochat_dir, depth, seq_len, device, aspect_ratio=64, head_dim=128, window_pattern="SSSL"):
        super().__init__()
        sys.path.insert(0, os.path.expanduser(nanochat_dir))
        import nanochat.gpt as ng
        from nanochat.tokenizer import get_tokenizer

        self._ng, self._orig_fa = ng, ng.flash_attn
        tok = get_tokenizer()
        # identical to base_train.build_model_meta
        base_dim = depth * aspect_ratio
        model_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
        nh = model_dim // head_dim
        cfg = ng.GPTConfig(sequence_len=seq_len, vocab_size=tok.get_vocab_size(), n_layer=depth, n_head=nh,
                           n_kv_head=nh, n_embd=model_dim, window_pattern=window_pattern)
        with torch.device("meta"):
            m = ng.GPT(cfg)
        m.to_empty(device=device)
        m.init_weights()
        self.model, self.nano_config = m, cfg
        self.n_embd, self.n_layer = model_dim, depth
        self.bos_id = tok.get_bos_token_id()
        b2t = [tok.enc.encode_single_token(bytes([b])) for b in range(256)]
        assert len(set(b2t)) == 256, "tokenizer lacks single-byte tokens"
        self.register_buffer("byte_to_token", torch.tensor(b2t, dtype=torch.long, device=device), persistent=False)

    def config_dict(self):
        return asdict(self.nano_config)

    def set_attn_impl(self, impl):
        self._ng.flash_attn = _ManualAttn if impl == "manual" else self._orig_fa

    def forward(self, idx):
        return self.model(idx)  # (B, T, vocab) float32, soft-capped logits

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def flops_per_token(self, T, backward=True):
        m = self.model
        h = m.config.n_head
        q = m.config.n_embd // h
        attn = sum(12 * h * q * (T if w[0] < 0 else min(w[0], T)) for w in m.window_sizes)
        fb = 6 * m.num_matmul_params() + attn
        return fb if backward else fb / 3

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        params = [p for p in self.parameters() if p.requires_grad]
        groups = [{"params": [p for p in params if p.dim() >= 2], "weight_decay": weight_decay},
                  {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0}]
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas, fused=device_type == "cuda")
