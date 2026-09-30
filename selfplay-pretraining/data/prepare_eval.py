"""
Held-out natural byte data for zero-shot bits/byte (never used for self-play gradient updates).
Writes raw byte files to data/eval/*.bin:
  text     - tiny shakespeare (nanoGPT's char dataset)
  python   - Python standard library source code
  random   - uniform random bytes (should stay at 8.0 bpb: a sanity ceiling)
Add any other raw byte file (DCLM text, audio PCM, CIFAR bytes, ...) to data/eval/ and it is picked up.

It also writes byte-level nanoGPT training data for the downstream "does self-play help nanoGPT" test:
  data/shakespeare_bytes/{train,val}.bin
"""
import os
import sysconfig
import urllib.request

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
ev = os.path.join(here, "eval")
os.makedirs(ev, exist_ok=True)

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
shakes = urllib.request.urlopen(url).read()
print(f"shakespeare: {len(shakes):,} bytes")

stdlib = sysconfig.get_paths()["stdlib"]
chunks, total = [], 0
for root, _, files in sorted(os.walk(stdlib)):
    if "site-packages" in root or "test" in root:
        continue
    for f in sorted(files):
        if f.endswith(".py"):
            b = open(os.path.join(root, f), "rb").read()
            chunks.append(b)
            total += len(b)
    if total > 8_000_000:
        break
py = b"\n".join(chunks)
print(f"python stdlib: {len(py):,} bytes")

# zero-shot eval uses the held-out last 10% of shakespeare so downstream training never sees it
n = int(0.9 * len(shakes))
np.frombuffer(shakes[n:], dtype=np.uint8).tofile(os.path.join(ev, "text.bin"))
np.frombuffer(py, dtype=np.uint8).tofile(os.path.join(ev, "python.bin"))
np.random.default_rng(0).integers(0, 256, size=1_000_000, dtype=np.uint8).tofile(os.path.join(ev, "random.bin"))

ds = os.path.join(here, "shakespeare_bytes")
os.makedirs(ds, exist_ok=True)
np.frombuffer(shakes[:n], dtype=np.uint8).tofile(os.path.join(ds, "train.bin"))
np.frombuffer(shakes[n:], dtype=np.uint8).tofile(os.path.join(ds, "val.bin"))
print("wrote", sorted(os.listdir(ev)), "and", ds)
