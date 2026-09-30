"""
Python wrapper around bf.c: program alphabet, macro expansion, and a threaded batch executor.
The shared library is compiled on first import (needs a C compiler on PATH).
"""
import ctypes
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# ---------------------------------------------------------------------------
# alphabet (appendix E): 8 Brainf*ck ops + 10 macro tokens + end-of-program F
BF_OPS = "<>+-[].,"
MACROS = {
    "Z": "[-]",
    "R": "[->+<]",
    "L": "[->+++<]",
    "N": "[-<->]",
    "C": "[->+>+<<]",
    "G": "[>]",
    "H": "[<]",
    "W": "[[-]>+<]",
    "V": "[.>]",
    "X": "[-]" + "+" * 16,
}
END = "F"
BODY_TOKENS = BF_OPS + "".join(MACROS)  # 18 body tokens
ALPHABET = BODY_TOKENS + END            # |A| = 19
PROG_PREFIX = ord("S")
OUT_PREFIX = ord("O")
ALPHABET_BYTES = np.frombuffer(ALPHABET.encode(), dtype=np.uint8)
END_BYTE = ord(END)


def expand(body: str) -> str:
    """Expand macro tokens into pure Brainf*ck."""
    return "".join(MACROS.get(c, c) for c in body)


# ---------------------------------------------------------------------------
_here = os.path.dirname(os.path.abspath(__file__))
_src = os.path.join(_here, "bf.c")
_lib_path = os.path.join(_here, "_bf.so")


def _load():
    if not os.path.exists(_lib_path) or os.path.getmtime(_lib_path) < os.path.getmtime(_src):
        cc = os.environ.get("CC", "cc")
        subprocess.check_call([cc, "-O3", "-shared", "-fPIC", "-o", _lib_path, _src])
    lib = ctypes.CDLL(_lib_path)
    lib.bf_run_batch.restype = None
    lib.bf_run_batch.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int64, ctypes.c_void_p,
    ]
    return lib


_lib = _load()
_pool = None


def run_programs(bodies, T, tape_len=256, max_steps=1 << 17, seeds=None, num_threads=None):
    """
    Execute program bodies (strings over BODY_TOKENS, no trailing F) on U with fresh random tapes.
    Returns (out [n, T] uint8, out_len [n] int32, max_depth [n] int32).
    """
    global _pool
    n = len(bodies)
    expanded = [expand(b).encode() for b in bodies]
    stride = max(1, max((len(e) for e in expanded), default=1))
    progs = np.zeros((n, stride), dtype=np.uint8)
    lens = np.zeros(n, dtype=np.int32)
    for i, e in enumerate(expanded):
        progs[i, : len(e)] = np.frombuffer(e, dtype=np.uint8)
        lens[i] = len(e)
    if seeds is None:
        seeds = np.random.randint(1, 2**63 - 1, size=n, dtype=np.int64).view(np.uint64)
    seeds = np.ascontiguousarray(seeds, dtype=np.uint64)
    out = np.zeros((n, T), dtype=np.uint8)
    out_len = np.zeros(n, dtype=np.int32)
    max_depth = np.zeros(n, dtype=np.int32)

    num_threads = num_threads or min(32, os.cpu_count() or 1)
    if _pool is None or _pool._max_workers != num_threads:
        _pool = ThreadPoolExecutor(num_threads)
    chunks = np.array_split(np.arange(n), num_threads)

    def work(idx):
        if len(idx) == 0:
            return
        a, b = int(idx[0]), int(idx[-1]) + 1
        # ctypes releases the GIL for the duration of the call
        _lib.bf_run_batch(
            progs[a:b].ctypes.data, lens[a:b].ctypes.data, b - a, stride,
            out[a:b].ctypes.data, out_len[a:b].ctypes.data, max_depth[a:b].ctypes.data,
            T, tape_len, max_steps, seeds[a:b].ctypes.data,
        )

    list(_pool.map(work, chunks))
    return out, out_len, max_depth


if __name__ == "__main__":
    # the worked example from appendix E: +++[>+.<-]F emits (1, 2, 3, 0, ...)
    o, l, d = run_programs(["+++[>+.<-]"], T=8)
    print(o[0].tolist(), l[0], d[0])
    assert o[0].tolist() == [1, 2, 3, 0, 0, 0, 0, 0] and l[0] == 3 and d[0] == 1
    # unmatched brackets are no-ops, random input produces varying output
    o, l, d = run_programs(["]]+.[", ",.,.,."] * 2, T=4)
    print(o.tolist(), l.tolist())
    import time
    rng = np.random.default_rng(0)
    progs = ["".join(rng.choice(list(BODY_TOKENS), size=24)) for _ in range(1024)]
    t = time.time()
    o, l, d = run_programs(progs, T=4096)
    print(f"1024 random programs, T=4096: {time.time() - t:.3f}s, mean emitted {l.mean():.0f}, max depth {d.max()}")
