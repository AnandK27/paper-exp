# 24.4M-param learner + generator (the paper's largest scale, used for the A.1 warm-start experiment).
# 8192 rounds x 1024 programs x 4096 bytes = 34.36B token budget. Roughly 1-2 days on 1xH200.
out_dir = "out-selfplay-25m"
n_layer = 8
n_head = 8
n_embd = 512
mlp_hidden = 1280  # -> 24.4M params
block_size = 4096
pool_size = 1024
max_rounds = 8192
learning_rate = 1e-3
eval_interval = 128
ckpt_interval = 256
reward_tokens_per_micro = 16384  # manual attention under JVP materializes B x H x T x T
compile = False  # micro-batch lengths vary (short outputs are trimmed) -> recompiles
snap_every = 64  # lookback snapshots kept on CPU: ~e/(2*snap_every) x 100MB
