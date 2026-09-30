"""
Build a byte-level tokenizer in nanochat's on-disk format, in a separate NANOCHAT_BASE_DIR so the BPE runs stay
untouched. Vocabulary = the 256 raw bytes (no merges) + nanochat's 9 special tokens = 265 ids; <|bos|> = 256.
Self-play outputs and natural text then share one token space, so there is no tokenizer mismatch.
nanochat's val metric is bits per byte, so results remain comparable across tokenizers.

    ~/nanochat/.venv/bin/python make_byte_tokenizer.py ~/.cache/nanochat_bytes ~/.cache/nanochat
(the second argument is the BPE base dir whose ClimbMix shards get symlinked in)
"""
import os
import pickle
import sys

import tiktoken
import torch

sys.path.insert(0, os.path.expanduser("~/nanochat"))
from nanochat.tokenizer import SPECIAL_TOKENS, SPLIT_PATTERN, RustBPETokenizer  # noqa: E402

base, bpe_base = os.path.expanduser(sys.argv[1]), os.path.expanduser(sys.argv[2])
tok_dir = os.path.join(base, "tokenizer")
os.makedirs(tok_dir, exist_ok=True)

enc = tiktoken.Encoding(
    name="bytes",
    pat_str=SPLIT_PATTERN,
    mergeable_ranks={bytes([i]): i for i in range(256)},
    special_tokens={name: 256 + i for i, name in enumerate(SPECIAL_TOKENS)},
)
with open(os.path.join(tok_dir, "tokenizer.pkl"), "wb") as f:
    pickle.dump(enc, f)
token_bytes = torch.tensor([1] * 256 + [0] * len(SPECIAL_TOKENS), dtype=torch.int32)
torch.save(token_bytes, os.path.join(tok_dir, "token_bytes.pt"))

data_link = os.path.join(base, "base_data_climbmix")
if not os.path.exists(data_link):
    os.symlink(os.path.join(bpe_base, "base_data_climbmix"), data_link)

# sanity checks
tok = RustBPETokenizer(enc, "<|bos|>")
s = "Hello, wörld! 123"
ids = tok.encode(s)
assert ids == list(s.encode("utf-8")), ids
assert tok.decode(ids) == s
assert tok.get_bos_token_id() == 256 and tok.get_vocab_size() == 265
print(f"byte tokenizer: vocab {tok.get_vocab_size()}, bos {tok.get_bos_token_id()}, wrote {tok_dir}, data -> {os.readlink(data_link)}")
