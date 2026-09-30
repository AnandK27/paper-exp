"""
Stack-machine substrate (stackm.c), a drop-in alternative to bf.py for selfplay.py (--machine=stack).
Same interface: BODY_TOKENS (18), END, ALPHABET (19), ALPHABET_BYTES, PROG_PREFIX, OUT_PREFIX, run_programs.
Where the BF tape favours local, cell-by-cell patterns, the stack machine favours arithmetic structure:
counters, Fibonacci-like recurrences, powers, modular cycles, xor patterns.
"""
import ctypes
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np

BODY_TOKENS = "01+-*/%^=dsoxr.,[]"
END = "F"
ALPHABET = BODY_TOKENS + END
PROG_PREFIX = ord("S")
OUT_PREFIX = ord("O")
ALPHABET_BYTES = np.frombuffer(ALPHABET.encode(), dtype=np.uint8)
END_BYTE = ord(END)
assert len(set(ALPHABET)) == 19

_here = os.path.dirname(os.path.abspath(__file__))
_src = os.path.join(_here, "stackm.c")
_lib_path = os.path.join(_here, "_stackm.so")


def _load():
    if not os.path.exists(_lib_path) or os.path.getmtime(_lib_path) < os.path.getmtime(_src):
        subprocess.check_call([os.environ.get("CC", "cc"), "-O3", "-shared", "-fPIC", "-o", _lib_path, _src])
    lib = ctypes.CDLL(_lib_path)
    lib.sm_run_batch.restype = None
    lib.sm_run_batch.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int64, ctypes.c_void_p,
    ]
    return lib


_lib = _load()
_pool = None


def run_programs(bodies, T, tape_len=64, max_steps=1 << 17, seeds=None, num_threads=None):
    """tape_len is the stack capacity here (capped at 256). Returns (out, out_len, max_depth)."""
    global _pool
    n = len(bodies)
    enc = [b.encode() for b in bodies]
    stride = max(1, max((len(e) for e in enc), default=1))
    progs = np.zeros((n, stride), dtype=np.uint8)
    lens = np.zeros(n, dtype=np.int32)
    for i, e in enumerate(enc):
        progs[i, : len(e)] = np.frombuffer(e, dtype=np.uint8)
        lens[i] = len(e)
    if seeds is None:
        seeds = np.random.randint(1, 2**63 - 1, size=n, dtype=np.int64).view(np.uint64)
    seeds = np.ascontiguousarray(seeds, dtype=np.uint64)
    out = np.zeros((n, T), dtype=np.uint8)
    out_len = np.zeros(n, dtype=np.int32)
    max_depth = np.zeros(n, dtype=np.int32)
    cap = max(1, min(256, tape_len))
    num_threads = num_threads or min(32, os.cpu_count() or 1)
    if _pool is None or _pool._max_workers != num_threads:
        _pool = ThreadPoolExecutor(num_threads)

    def work(idx):
        if len(idx) == 0:
            return
        a, b = int(idx[0]), int(idx[-1]) + 1
        _lib.sm_run_batch(progs[a:b].ctypes.data, lens[a:b].ctypes.data, b - a, stride,
                          out[a:b].ctypes.data, out_len[a:b].ctypes.data, max_depth[a:b].ctypes.data,
                          T, cap, max_steps, seeds[a:b].ctypes.data)

    list(_pool.map(work, np.array_split(np.arange(n), num_threads)))
    return out, out_len, max_depth


if __name__ == "__main__":
    def show(p, T=12):
        o, l, d = run_programs([p], T=T)
        return o[0, : l[0]].tolist(), int(d[0])

    # counter: push 1, loop { emit, +1 }  -> 1,2,3,...
    assert show("1[.1+]")[0] == list(range(1, 13))
    # fibonacci: stack (a b) -> emit b; swap, over, add -> (b a+b)
    assert show("01[.so+]", T=10)[0] == [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
    # powers of two: 1 loop { emit, dup + }
    assert show("1[.d+]", T=8)[0] == [1, 2, 4, 8, 16, 32, 64, 128]
    # underflow reads 0, unmatched brackets are no-ops, random input varies
    print("x+.]] ->", show("x+.]]"), "| ,., ->", show(",.,.", T=4))
    import time
    rng = np.random.default_rng(0)
    progs = ["".join(rng.choice(list(BODY_TOKENS), size=24)) for _ in range(1024)]
    t = time.time()
    o, l, d = run_programs(progs, T=4096)
    print(f"1024 random programs: {time.time() - t:.3f}s, mean emitted {l.mean():.0f}, max depth {d.max()}")
    print("all tests passed")
