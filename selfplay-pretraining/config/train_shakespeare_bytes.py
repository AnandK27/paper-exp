# Byte-level nanoGPT on tiny shakespeare. Pair with --init_from=scratch vs --init_from=<selfplay ckpt>.
# Model shape must match the self-play checkpoint (it is overridden from the ckpt when warm-starting).
dataset = "shakespeare_bytes"
batch_size = 32
block_size = 256
max_iters = 5000
lr_decay_iters = 5000
eval_interval = 100
learning_rate = 1e-3
