"""
Minimal additions to karpathy/nanochat scripts/base_train.py for the compute-matched warm-up test:
  --init-from PATH   load a self-play checkpoint's "nanochat_state" after init_weights (fresh optimizer, step 0)
  --seed N           reseed torch before the model is initialized (default: nanochat's fixed 42)
  NANOCHAT_JSONL     if set, append {"step", "val_bpb", "flops", "total_flops"} per eval and a final record;
                     total_flops includes the self-play FLOPs stored in the --init-from checkpoint
Everything else (model, Muon optimizer, data, schedule, batch size) is untouched.
    python apply_patch.py ~/nanochat/scripts/base_train.py
"""
import sys

path = sys.argv[1]
s = open(path).read()
if "--init-from" in s:
    print("already patched")
    sys.exit(0)


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:80])
    s = s.replace(old, new)


rep('''parser.add_argument("--resume-from-step",''',
    '''parser.add_argument("--init-from", type=str, default="", help="warm-start weights from a self-play checkpoint (nanochat_state)")
parser.add_argument("--seed", type=int, default=-1, help="reseed torch before model init (-1 = nanochat default)")
parser.add_argument("--resume-from-step",''')

rep('''model = build_model_meta(args.depth) # 1) Build on meta device (only shapes/dtypes, no data)''',
    '''if args.seed >= 0:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
model = build_model_meta(args.depth) # 1) Build on meta device (only shapes/dtypes, no data)''')

rep('''model.init_weights() # 3) All tensors get initialized
''', '''model.init_weights() # 3) All tensors get initialized
init_flops = 0.0
if args.init_from:
    _ck = torch.load(args.init_from, map_location=device, weights_only=False)
    model.load_state_dict(_ck["nanochat_state"], strict=True)
    init_flops = float(_ck.get("flops", 0.0))
    print0(f"Warm start from {args.init_from} (self-play round {_ck.get('round')}, {init_flops:.3e} FLOPs)")
    del _ck
_jsonl = os.environ.get("NANOCHAT_JSONL")
def _log_jsonl(rec):
    if _jsonl and master_process:
        with open(_jsonl, "a") as _f:
            _f.write(json.dumps(rec) + "\\n")
''')

rep('''        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.6f}")
''', '''        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.6f}")
        _log_jsonl({"step": step, "val_bpb": val_bpb, "flops": flops_so_far, "total_flops": flops_so_far + init_flops})
''')

rep('''    print0(f"Minimum validation bpb: {min_val_bpb:.6f}")
''', '''    print0(f"Minimum validation bpb: {min_val_bpb:.6f}")
    _log_jsonl({"final_val_bpb": val_bpb, "min_val_bpb": min_val_bpb, "num_iterations": num_iterations,
                "train_flops": num_flops_per_token * total_batch_size * num_iterations, "selfplay_flops": init_flops,
                "total_flops": num_flops_per_token * total_batch_size * num_iterations + init_flops,
                "init_from": args.init_from, "seed": args.seed, "depth": args.depth})
''')
open(path, "w").write(s)
print("patched", path)
