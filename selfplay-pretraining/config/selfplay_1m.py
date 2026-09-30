# ~1M-param learner + generator: the paper's ablation scale (Table 5). Minutes-to-hours on 1xH200.
out_dir = "out-selfplay-1m"
n_layer = 4
n_head = 4
n_embd = 128
block_size = 4096
pool_size = 1024
max_rounds = 8192
learning_rate = 3e-3
eval_interval = 128
