"""
Static figures (PNG, 200 dpi) for the nanochat self-play experiments.
    python report/make_figs.py report/step_curves.json report/nanochat_bytes_results.json report/figs
Uses the same categorical palette as the report page (validated for color-vision deficiency).
"""
import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

steps_path, bytes_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)
S = json.load(open(steps_path))
R = json.load(open(bytes_path))

INK, INK2, MUTED, GRID = "#151514", "#52514e", "#8b8a84", "#e4e3dd"
SHARE = {0.05: ("#2a78d6", "5%"), 0.1: ("#1baf7a", "10%"), 0.2: ("#4a3aa7", "20%"), 0.4: ("#e87ba4", "40%"), 0.6: ("#e34948", "60%")}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "figure.dpi": 200, "savefig.bbox": "tight",
    "figure.constrained_layout.use": True, "axes.titlesize": 10.5, "figure.titlesize": 12,
})
LN2 = math.log(2)


def curve(dir_, stem, bits, start=0):
    runs = [S[k] for k in (f"{dir_}/{stem}_s0", f"{dir_}/{stem}_s1") if k in S]
    if not runs:
        return np.array([]), np.array([])
    n = min(len(r["loss"]) for r in runs)
    x = np.array([runs[0]["loss"][i][0] for i in range(n)])
    y = np.mean([[r["loss"][i][1] for i in range(n)] for r in runs], axis=0)
    y = y / LN2 if bits else y
    keep = x >= start
    return x[keep], y[keep]


def gap(a, b):
    xa, ya = a
    xb, yb = b
    common = np.intersect1d(xa, xb)
    return common, ya[np.isin(xa, common)] - yb[np.isin(xb, common)]


def save(fig, name):
    fig.savefig(os.path.join(out, name))
    plt.close(fig)
    print("wrote", os.path.join(out, name))


# 1) free checkpoint vs scratch: curves + gap
sc = curve("nc-bytes", "scratch", True)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
a1.plot(*sc, color=INK2, ls="--", lw=1.8, label="Scratch")
for sh in (0.4, 0.6):
    c = curve("nc-bytes", f"free_f{sh}", True)
    if len(c[0]):
        a1.plot(*c, color=SHARE[sh][0], lw=1.8, label=f"Free {SHARE[sh][1]} self-play checkpoint")
        g = gap(c, sc)
        m = g[0] >= 50
        a2.plot(g[0][m], g[1][m], color=SHARE[sh][0], lw=1.8, label=f"Free {SHARE[sh][1]} − scratch")
a1.set(ylim=(1.2, 2.6), xlabel="nanochat training step", ylabel="training loss (bits/byte)", title="Training loss by step")
a1.legend()
a2.axhline(0, color=INK, lw=1)
a2.set(ylim=(-0.3, 0.05), xlabel="nanochat training step", ylabel="loss difference (bits/byte)",
       title="Free checkpoint − scratch (below 0 = self-play ahead; axis clipped at −0.3)")
a2.legend()
fig.suptitle("Byte-tokenizer nanochat d8: self-play checkpoint (not charged) + full budget vs scratch")
save(fig, "free_vs_scratch_steps.png")

# 2) warm - control gap by step, all shares (bytes and BPE)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
for sh in (0.05, 0.1, 0.2, 0.4, 0.6):
    g = gap(curve("nc-bytes", f"bf_f{sh}", True), curve("nc-bytes", f"control_f{sh}", True))
    if len(g[0]):
        m = g[0] >= 50
        a1.plot(g[0][m], g[1][m], color=SHARE[sh][0], lw=1.6, label=f"{SHARE[sh][1]} self-play")
for sh in (0.05, 0.1, 0.2):
    g = gap(curve("nc-cm", f"bf_f{sh}", False), curve("nc-cm", f"short{1 - sh:.2f}", False))
    if len(g[0]):
        m = g[0] >= 50
        a2.plot(g[0][m], g[1][m], color=SHARE[sh][0], lw=1.6, label=f"{SHARE[sh][1]} self-play")
for ax, t, u in ((a1, "Byte tokenizer", "bits/byte"), (a2, "BPE tokenizer", "nats/token")):
    ax.axhline(0, color=INK, lw=1)
    ax.set(ylim=(-0.3, 0.4), xlabel="nanochat training step", ylabel=f"warm − random init ({u})", title=t)
    ax.legend(ncol=2)
fig.suptitle("Warm start minus matched random init, by step (below 0 = self-play ahead)")
save(fig, "warm_minus_control_steps.png")

# 3) small multiples: warm vs control vs scratch per share
for dir_, bits, shares, ctrl, ylim, unit, name in (
        ("nc-bytes", True, (0.05, 0.1, 0.2, 0.4, 0.6), lambda s: f"control_f{s}", (1.2, 2.8), "bits/byte", "panels_bytes.png"),
        ("nc-cm", False, (0.05, 0.1, 0.2), lambda s: f"short{1 - s:.2f}", (3.0, 5.0), "nats/token", "panels_bpe.png")):
    scr = curve(dir_, "scratch", bits, 20)
    fig, axes = plt.subplots(1, len(shares), figsize=(3.3 * len(shares), 3.4), sharey=True)
    for ax, sh in zip(np.atleast_1d(axes), shares):
        ax.plot(*scr, color=MUTED, ls=":", lw=1.4, label="Scratch")
        ax.plot(*curve(dir_, ctrl(sh), bits, 20), color=INK2, ls="--", lw=1.4, label="Random init")
        ax.plot(*curve(dir_, f"bf_f{sh}", bits, 20), color=SHARE[sh][0], lw=1.8, label="Self-play warm start")
        ax.set(ylim=ylim, title=f"{SHARE[sh][1]} self-play share", xlabel="step")
    np.atleast_1d(axes)[0].set_ylabel(f"training loss ({unit})")
    np.atleast_1d(axes)[0].legend(loc="upper right", fontsize=8)
    fig.suptitle(f"Warm start vs random init given the same steps ({'byte' if bits else 'BPE'} tokenizer)")
    save(fig, name)

# 4) final val vs share (bytes), with free-checkpoint points
g = {}
for r in R:
    g.setdefault((r["kind"], r["share"]), []).append(r["final"])
shares = sorted({s for k, s in g if k == "selfplay"})
scm = np.mean(g[("scratch", 0.0)])
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.axhline(scm, color=INK2, ls="--", lw=1.2, label=f"Scratch ({scm:.4f})")
for kind, color, marker, lab in (("selfplay", "#2a78d6", "o", "Self-play, then nanochat (compute-matched)"),
                                  ("control", "#eb6834", "s", "Random init, same nanochat steps")):
    xs = [0] + [s * 100 for s in shares]
    ys = [scm] + [np.mean(g[(kind, s)]) for s in shares]
    es = [0] + [np.std(g[(kind, s)]) for s in shares]
    ax.errorbar(xs, ys, yerr=es, color=color, marker=marker, ms=6, lw=1.8, capsize=3, label=lab)
fs = [s for s in shares if ("free", s) in g]
if fs:
    ax.scatter([s * 100 for s in fs], [np.mean(g[("free", s)]) for s in fs], color="#1baf7a", marker="D", s=40, zorder=3,
               label="Free checkpoint + full budget (uncharged)")
ax.set(xlabel="share of the budget spent on self-play (learner FLOPs), %", ylabel="final val bits/byte",
       title="Byte-tokenizer nanochat d8: final loss vs self-play share")
ax.legend(fontsize=8)
save(fig, "final_vs_share_bytes.png")
