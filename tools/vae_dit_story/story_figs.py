"""The 5 new figures for the interactive story page.

1. rf_growth      -- VAE receptive-field growth per stage + latent-cell RF similarity matrix
2. gravity        -- the "gravity" decomposition: weight x displacement, per sigma
3. eff_sources    -- effective number of sources vs sigma (+ why "many sources" is not "blurry")
4. same_xs_diff_c -- same x_sigma, different condition c -> different output
5. mask           -- the training attention mask (4 segments, 3 rules) + KV cache layout
"""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow

import pathlib as _pl
OUT = str(_pl.Path(__file__).resolve().parent / "docs") + "/"
D = np.array([-1.0, 1.0])

# ============================================================ 1. RF growth
stages = ["conv_in", "down0", "down1", "down2", "down3", "mid_res0", "mid_attn", "mid_res1", "conv_out"]
sig_dp = [1.708, 4.031, 8.342, 16.719, 26.369, 27.841, 28.585, 29.032, 29.160]
r90_dp = [3.536, 8.631, 17.734, 35.715, 54.578, 57.332, 58.543, 59.157, 59.382]
sig_ag = [1.678, 4.159, 8.993, 17.943, 24.441, 25.339, 25.831, 26.434, 28.283]
grids = ["64x64", "32x32", "16x16", "8x8", "8x8", "8x8", "8x8", "8x8", "8x8"]

fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.6), dpi=150)
ax = axes[0]
x = np.arange(len(stages))
ax.plot(x, sig_dp, "o-", color="#c0392b", lw=2, ms=6, label="RF $\\sigma$ (path-count, weight-free)")
ax.plot(x, sig_ag, "s--", color="#c0392b", lw=1.2, ms=4, alpha=0.55, label="RF $\\sigma$ (autograd, random weights)")
ax.plot(x, r90_dp, "^-", color="#2471a3", lw=2, ms=6, label="RF $r_{90}$ (path-count)")
ax.axhline(64, color="k", ls=":", lw=1.2)
ax.text(5.2, 66, "half the frame (64 px)", fontsize=8)
ax.axvspan(3.5, 8.5, color="#f39c12", alpha=0.12)
ax.text(6.0, 6, "coverage reaches\n100% of the frame", fontsize=8.5, color="#b9770e", ha="center")
for i, (s, g) in enumerate(zip(sig_dp, grids)):
    ax.annotate(f"{s:.2f}", (i, s), textcoords="offset points", xytext=(0, 9), fontsize=7.5, ha="center", color="#c0392b")
    ax.annotate(g, (i, -3), textcoords="offset points", xytext=(0, 0), fontsize=7, ha="center", color="#555")
ax.set_xticks(x)
ax.set_xticklabels(stages, rotation=35, ha="right", fontsize=8)
ax.set_ylabel("pixels", fontsize=9.5)
ax.set_ylim(-8, 78)
ax.set_title("(1) one latent cell's receptive field grows 17x\n"
             "from 1.71 px (conv_in) to 29.16 px (conv_out)", fontsize=10)
ax.legend(fontsize=7.6, frameon=False, loc="upper left")
ax.tick_params(labelsize=8)

# similarity matrix of latent-cell RFs (8x8 offsets)
S = np.array([
 [1.0000,0.9959,0.9931,0.9798,0.9676,0.9489,0.9527,0.9586][:8],
 [0.9959,1.0000,0.9959,0.9931,0.9798,0.9676,0.9489,0.9527][:8],
 [0.9931,0.9959,1.0000,0.9959,0.9931,0.9798,0.9676,0.9489][:8],
 [0.9798,0.9931,0.9959,1.0000,0.9959,0.9931,0.9798,0.9676][:8],
 [0.9676,0.9798,0.9931,0.9959,1.0000,0.9959,0.9931,0.9798][:8],
 [0.9489,0.9676,0.9798,0.9931,0.9959,1.0000,0.9959,0.9931][:8],
 [0.9586,0.9489,0.9676,0.9798,0.9931,0.9959,1.0000,0.9959][:8],
 [0.9670,0.9586,0.9489,0.9676,0.9798,0.9931,0.9959,1.0000][:8],
])
ax = axes[1]
im = ax.imshow(S, cmap="viridis", vmin=0.94, vmax=1.0)
for i in range(8):
    for j in range(8):
        ax.text(j, i, f"{S[i,j]:.3f}"[2:], ha="center", va="center", fontsize=6.5,
                color="w" if S[i, j] < 0.985 else "k")
ax.set_xlabel("latent cell offset (columns)", fontsize=9)
ax.set_ylabel("latent cell offset (rows)", fontsize=9)
ax.set_title("(2) neighbouring latent cells have almost the same RF:\n"
             "adjacent cosine 0.9959, still 0.9489 at offset 4", fontsize=10)
plt.colorbar(im, ax=ax, fraction=0.046, label="RF cosine similarity")
ax.tick_params(labelsize=8)
fig.tight_layout()
fig.savefig(OUT + "story_rf_growth.png", bbox_inches="tight")
plt.close(fig)
print("1/5 story_rf_growth.png")

# ============================================================ 2. gravity
def post(x, s, prior=(0.5, 0.5)):
    w = np.array(prior)
    lik = w[:, None] * np.exp(-0.5 * ((x - (1 - s) * D[:, None]) / s) ** 2)
    return lik / lik.sum(0)

XS = 0.1
fig, axes = plt.subplots(1, 4, figsize=(15.0, 4.3), dpi=150, sharey=False)
for ax, s in zip(axes, [0.7, 0.5, 0.3, 0.2]):
    p = post(XS, s)
    wts = [p[0, 0], p[1, 0]]
    pulls = [D[0] - XS, D[1] - XS]
    contrib = [w * pl for w, pl in zip(wts, pulls)]
    net = sum(contrib)
    for k, (d, w, pl, c) in enumerate(zip(D, wts, pulls, contrib)):
        col = "#2471a3" if d < 0 else "#c0392b"
        ax.annotate("", xy=(XS + c * 1.25, 1 + k * 1.0), xytext=(XS, 1 + k * 1.0),
                    arrowprops=dict(arrowstyle="-|>", lw=1 + 7 * w, color=col, alpha=0.9))
        ax.text(XS + c * 1.25 + (0.06 if c > 0 else -0.06), 1 + k * 1.0,
                f"$w$={w:.3f}\n$\\times$ {pl:+.2f}\n= {c:+.3f}", fontsize=7,
                ha="left" if c > 0 else "right", va="center", color=col)
    ax.annotate("", xy=(XS + net * 1.25, 3.30), xytext=(XS, 3.30),
                arrowprops=dict(arrowstyle="-|>", lw=3.2, color="#1e8449"))
    ax.text(XS, 3.78, f"resultant {net:+.4f}", fontsize=8.5, ha="center", color="#1e8449")
    for d in D:
        ax.plot([d], [0.25], "o", color="#2c3e50", ms=6)
        ax.text(d, 0.02, f"$x_0$={d:+.0f}", ha="center", fontsize=8)
    ax.axvline(XS, color="k", ls="--", lw=1.0)
    ax.text(XS, -0.22, f"$x_\\sigma$={XS}", ha="center", fontsize=8)
    mu = XS + net
    direction = ("net pull: RIGHT" if net > 0.02 else
                 ("net pull: LEFT (towards the centre)" if net < -0.02 else "net pull: ~ zero"))
    ax.set_title(f"$\\sigma$ = {s}\nmean $x_0$ = {mu:+.3f}  →  {direction}", fontsize=9.5)
    ax.set_xlim(-1.7, 1.7); ax.set_ylim(-0.5, 4.25)
    ax.set_yticks([]); ax.tick_params(labelsize=8)
    ax.set_xlabel("$x$", fontsize=9)
axes[0].set_ylabel("gravity pulls (thickness $\\propto$ weight)", fontsize=9)
fig.suptitle("Every sample is a gravity source at $x_0$; its strength is weight = prior x likelihood. "
             "The output is the NORMALISED weighted sum of the pull vectors.", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig(OUT + "story_gravity.png", bbox_inches="tight")
plt.close(fig)
print("2/5 story_gravity.png")

# ============================================================ 3. effective sources
rng = np.random.default_rng(0)
N = 20000
x0 = rng.normal(size=N)
XQ = 0.15
sigs = np.logspace(0, -3.5, 60)
eff, sd, theory = [], [], []
for s in sigs:
    c = (1 - s) * x0
    lik = np.exp(-0.5 * ((XQ - c) / s) ** 2)
    w = lik / lik.sum()
    eff.append(1 / np.sum(w ** 2))
    mu = np.sum(w * x0); sd.append(math.sqrt(np.sum(w * (x0 - mu) ** 2)))
    q = np.exp(-0.5 * (XQ / (1 - s)) ** 2) / ((1 - s) * math.sqrt(2 * math.pi)) / (1 - s) if s < 1 else np.nan
    theory.append(3.545 * N * q * s)

fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.3), dpi=150)
ax = axes[0]
ax.loglog(sigs, eff, "o-", ms=3, color="#c0392b", lw=1.6, label="measured effective #sources")
ax.loglog(sigs, theory, "--", color="#7f8c8d", lw=2, label="theory  $3.5\\,N\\,q\\,\\sigma$")
ax.axhline(N, color="k", ls=":", lw=1)
ax.text(0.9, N * 0.72, f"all {N:,} samples", fontsize=8)
ax.axhline(1, color="#1e8449", ls=":", lw=1)
ax.text(0.9, 1.6, "one source", fontsize=8, color="#1e8449")
ax.set_xlabel("$\\sigma$", fontsize=9.5); ax.set_ylabel("effective number of sources  $1/\\sum w^2$", fontsize=9.5)
ax.set_title("(1) the number of sources that actually pull you\ndrops LINEARLY with $\\sigma$, not instantly to 1", fontsize=10)
ax.legend(fontsize=8, frameon=False)
ax.tick_params(labelsize=8)
ax = axes[1]
ax.semilogx(sigs, sd, "o-", ms=3, color="#2471a3", lw=1.8)
ax.set_xlabel("$\\sigma$", fontsize=9.5); ax.set_ylabel("weighted std of $x_0$ among the sources", fontsize=9.5)
ax.set_title("(2) but those many sources AGREE with each other\n(their spread collapses) — 'many sources' $\\neq$ 'blurry'", fontsize=10)
ax.tick_params(labelsize=8)
fig.tight_layout()
fig.savefig(OUT + "story_eff_sources.png", bbox_inches="tight")
plt.close(fig)
print("3/5 story_eff_sources.png")

# ============================================================ 4. same x_sigma, different c
def N_(x, m, s): return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * math.sqrt(2 * math.pi))
lA, lB = N_(0.1, 0.5, 0.5), N_(0.1, -0.5, 0.5)
cases = [("no condition", 0.6, 0.4, "#7f8c8d"),
         ("$c$ = grasp", 0.9, 0.1, "#c0392b"),
         ("$c$ = place", 0.1, 0.9, "#2471a3"),
         ("$c$ = random label", 0.6, 0.4, "#bdc3c7")]
fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.4), dpi=150)
ax = axes[0]
xs = np.linspace(-1.6, 1.6, 600)
ax.plot(xs, np.exp(-0.5 * ((xs - 0.5) / 0.5) ** 2), color="#c0392b", lw=2.2, label="$x_0=+1$: $N(+0.5,0.25)$")
ax.plot(xs, np.exp(-0.5 * ((xs + 0.5) / 0.5) ** 2), color="#2471a3", lw=2.2, label="$x_0=-1$: $N(-0.5,0.25)$")
ax.axvline(0.1, color="k", ls="--", lw=1.2)
ax.text(0.14, 0.93, "the same reading\n$x_\\sigma=0.1$", fontsize=8.5)
ax.set_xlabel("$x$", fontsize=9.5); ax.set_ylabel("likelihood (peak scaled to 1)", fontsize=9.5)
ax.set_title("(1) conditioning does NOT touch these two curves\n"
             "(same $x_0$, same $\\sigma$: the Gaussians are identical)", fontsize=10)
ax.legend(fontsize=8, frameon=False)
ax.tick_params(labelsize=8)
ax = axes[1]
names = [c[0] for c in cases]
vals = [c[1] * (-1.8) + c[2] * (2.2) for c in cases]
cols = [c[3] for c in cases]
bars = ax.barh(range(len(cases)), vals, color=cols)
for i, (v, (nm, pa, pb, _)) in enumerate(zip(vals, cases)):
    ax.text(v + (0.08 if v > 0 else -0.08), i, f"{v:+.3f}", va="center",
            ha="left" if v > 0 else "right", fontsize=9)
    ax.text(-2.05, i, f"weights  A:B = {pa:.1f} : {pb:.1f}", fontsize=7.6, color="#555", va="center")
ax.axvline(-0.20, color="#7f8c8d", ls=":", lw=1.4)
ax.text(-0.20, 3.62, "unconditional answer", fontsize=7.6, color="#7f8c8d", ha="center")
ax.set_yticks(range(len(cases))); ax.set_yticklabels(names, fontsize=9)
ax.set_xlim(-2.35, 2.1); ax.set_ylim(-0.6, 3.9)
ax.set_xlabel("the output the network learns for this input", fontsize=9.5)
ax.set_title("(2) same input except $c$ -> totally different output\n(the lens re-weights each source)", fontsize=10)
ax.tick_params(labelsize=8)
fig.tight_layout()
fig.savefig(OUT + "story_same_xs_diff_c.png", bbox_inches="tight")
plt.close(fig)
print("4/5 story_same_xs_diff_c.png")

# ============================================================ 5. attention mask
B = 1
L_F, L_H, L_W = 4, 8, 16          # latent frames x (8x16) -> patch (1,2,2) -> 4x8 per frame
A_F, A_H, A_W = 4, 4, 1
p0, p1, p2 = 1, 2, 2
chunk_size, window = 2, 64
LTok = (L_F // p0) * (L_H // p1) * (L_W // p2)
ATok = A_F * A_H * A_W
lat_seq = np.zeros(LTok, int); act_seq = np.zeros(ATok, int)
lat_fr = np.repeat(np.arange(L_F), (L_H // p1) * (L_W // p2))
act_fr = np.repeat(np.arange(A_F), A_H * A_W)
seq = np.concatenate([lat_seq, lat_seq, act_seq, act_seq])
frame = np.concatenate([lat_fr // chunk_size * 2] * 2 + [act_fr // chunk_size * 2 + 1] * 2)
noise = np.concatenate([np.zeros(LTok, int), np.ones(LTok, int), np.zeros(ATok, int), np.ones(ATok, int)])
total = len(seq)
pad = (128 - total % 128) % 128
seq = np.pad(seq, (0, pad), constant_values=-1)
frame = np.pad(frame, (0, pad), constant_values=-1)
noise = np.pad(noise, (0, pad), constant_values=-1)
n = len(seq)
q = np.arange(n)[:, None]; k = np.arange(n)[None, :]
m_cc = (noise[q] == 1) & (noise[k] == 1) & (frame[k] <= frame[q])
m_nc = (noise[q] == 0) & (noise[k] == 1) & (frame[k] < frame[q])
m_nn = (noise[q] == 0) & (noise[k] == 0) & (frame[k] == frame[q])
mask = (m_cc | m_nc | m_nn) & (seq[q] == seq[k]) & (seq[q] >= 0) & (np.abs(frame[q] - frame[k]) <= window)

fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), dpi=150)
ax = axes[0]
ax.imshow(mask, cmap="gray_r", interpolation="nearest", vmin=0, vmax=1)
segs = [0, LTok, 2 * LTok, 2 * LTok + ATok, 2 * LTok + 2 * ATok]
for b in segs[1:-1]:
    ax.axvline(b - 0.5, color="#c0392b", lw=1.2); ax.axhline(b - 0.5, color="#c0392b", lw=1.2)
ax.axvline(total - 0.5, color="#2471a3", lw=1.2); ax.axhline(total - 0.5, color="#2471a3", lw=1.2)
lab = [("noisy\nlatent", 0, LTok), ("clean\nlatent", LTok, 2 * LTok),
       ("noisy\naction", 2 * LTok, 2 * LTok + ATok), ("clean\naction", 2 * LTok + ATok, total),
       ("padding", total, n)]
for t, a, b in lab:
    ax.text((a + b) / 2, 1.015, t.replace("\n", " "), transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=7.5, color="#2c3e50")
ax.set_title(f"(1) the training attention mask ({n}x{n})   -   black = allowed to attend", fontsize=10, pad=30)
ax.set_xlabel("key (what it reads)", fontsize=9); ax.set_ylabel("query (what it writes)", fontsize=9)
ax.tick_params(labelsize=7)

ax = axes[1]
z = 160
ax.imshow(mask[:z, :z], cmap="gray_r", interpolation="nearest", vmin=0, vmax=1)
for b in [64, 128]:
    ax.axvline(b - 0.5, color="#c0392b", lw=1.0); ax.axhline(b - 0.5, color="#c0392b", lw=1.0)
ax.annotate("", xy=(30, 80), xytext=(30, 48), arrowprops=dict(arrowstyle="<->", lw=1.0, color="#2471a3"))
ax.text(70, 64, "rule 3: noisy -> noisy\n(same chunk only)", fontsize=7.2, color="#2471a3", va="center")
ax.annotate("", xy=(96, 150), xytext=(148, 150), arrowprops=dict(arrowstyle="<->", lw=1.0, color="#1e8449"))
ax.text(2, 150, "rule 2: noisy chunk-1 -> clean chunk-0", fontsize=7.2, color="#1e8449",
        bbox=dict(fc="w", ec="none", alpha=0.9))
ax.text(68, 22, "the first chunk has no history:\nrule 2 gives it nothing",
        fontsize=7.2, color="#7f8c8d")
ax.set_title("(2) zoom 160x160 (latent part):\n"
             "own chunk (rule 3) + earlier clean chunk (rule 2); clean->noisy is blocked", fontsize=10)
ax.set_xlabel("key", fontsize=9); ax.set_ylabel("query", fontsize=9)
ax.text(0.5, -0.24, "rule 1: clean -> clean (causal)      rule 2: noisy -> clean of EARLIER chunks\n"
                    "rule 3: noisy -> noisy, same chunk only      rule 4: clean -> noisy = FORBIDDEN",
        transform=ax.transAxes, fontsize=7.6, ha="center", va="top",
        bbox=dict(fc="w", ec="#999", alpha=0.95))
ax.tick_params(labelsize=7)
fig.tight_layout()
fig.savefig(OUT + "story_mask.png", bbox_inches="tight")
plt.close(fig)
print(f"5/5 story_mask.png   (tokens: latent {LTok} + action {ATok} -> total {total} -> padded {n})")
