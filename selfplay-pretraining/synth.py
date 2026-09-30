"""
Hand-designed synthetic warm-up generators, aimed at the skills that matter first for next-byte prediction
on natural text. Each row draws fresh parameters (alphabet, vocabulary, ...), so the structure must be
inferred in context, as with a new document. All rows are zero-free (byte 0 is padding elsewhere).

  dyck      nested brackets, k ~ U{2..16} bracket types mapped to random byte pairs; formal-language
            pre-pretraining on k-Dyck is what Hu et al. (2025) found transfers best to language
  induction random spans over a random alphabet; half the spans are exact copies of an earlier span,
            which trains in-context copying (induction heads)
  zipf      a random word language: a fresh vocabulary of lowercase-ASCII words per row, Zipf unigram
            frequencies, sparse word-to-word transitions, spaces, capitalized sentences, punctuation and
            newlines. This one encodes text-shaped priors on purpose (it is not "zero natural data").
  mix       each row is one of the three above, uniformly
"""
import random
import string

LETTERS = string.ascii_lowercase.encode()


def _distinct_bytes(rng, k):
    return rng.sample(range(1, 256), k)


def sample_dyck(n, rng):
    k = rng.randint(2, 16)
    b = _distinct_bytes(rng, 2 * k)
    opens, closes = b[:k], b[k:]
    max_depth = rng.randint(4, 32)
    p_open = rng.uniform(0.4, 0.6)
    out, stack = [], []
    while len(out) < n:
        if not stack or (len(stack) < max_depth and rng.random() < p_open):
            t = rng.randrange(k)
            stack.append(t)
            out.append(opens[t])
        else:
            out.append(closes[stack.pop()])
    return out


def sample_induction(n, rng):
    alpha = _distinct_bytes(rng, rng.randint(4, 64))
    spans, out = [], []
    while len(out) < n:
        if spans and rng.random() < 0.5:
            s = rng.choice(spans)  # exact copy of an earlier span
        else:
            s = [rng.choice(alpha) for _ in range(rng.randint(4, 32))]
            spans.append(s)
        out.extend(s)
    return out[:n]


def sample_zipf(n, rng):
    V = rng.randint(50, 2000)
    alpha = rng.uniform(0.9, 1.3)
    vocab, seen = [], set()
    while len(vocab) < V:
        w = bytes(rng.choice(LETTERS) for _ in range(min(12, 1 + int(rng.expovariate(1 / 4)))))
        if w not in seen:
            seen.add(w)
            vocab.append(w)
    weights = [1.0 / (i + 1) ** alpha for i in range(V)]
    # each word has a few preferred successors (sparse bigram structure)
    succ = {}
    out = bytearray()
    prev, sent_left = None, 0
    while len(out) < n:
        if sent_left == 0:
            sent_left = rng.randint(4, 20)
            cap = True
        if prev is not None and rng.random() < 0.5:
            if prev not in succ:
                succ[prev] = rng.choices(range(V), weights=weights, k=rng.randint(1, 6))
            w = rng.choice(succ[prev])
        else:
            w = rng.choices(range(V), weights=weights, k=1)[0]
        word = vocab[w]
        if cap:
            word = word[:1].upper() + word[1:]
            cap = False
        out += word
        sent_left -= 1
        if sent_left == 0:
            out += rng.choice([b".", b".", b".", b"?", b"!"])
            out += b"\n" if rng.random() < 0.2 else b" "
        else:
            out += b", " if rng.random() < 0.08 else b" "
        prev = w
    return list(out[:n])


GENERATORS = {"dyck": sample_dyck, "induction": sample_induction, "zipf": sample_zipf}


def sample_mix(n, rng):
    return GENERATORS[rng.choice(sorted(GENERATORS))](n, rng)


GENERATORS["mix"] = sample_mix


if __name__ == "__main__":
    import time
    rng = random.Random(0)
    for name, g in GENERATORS.items():
        t = time.time()
        rows = [g(1024, rng) for _ in range(64)]
        assert all(len(r) == 1024 and 0 not in r and max(r) < 256 for r in rows), name
        dt = (time.time() - t) / 64 * 1000
        print(f"{name:<10} {dt:5.1f} ms/row | {bytes(rows[0][:90])!r}")
