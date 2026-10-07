"""Gravity-field visualisation: samples on the x-axis, sigma on the y-axis.

Each row (one sigma) is the gravity-field weight vector over the samples:
    w_k(sigma)  propto  prior_k * N(x_sigma ; (1-sigma) x0_k , sigma^2)
Row-normalised, so every row is a distribution over "which samples pull you".
The model output is the weighted direction  v = (x_sigma - sum_k w_k x0_k)/sigma.

Top row  : unconditional
Bottom   : the same samples with a re-weighted prior (the "lens"), p(+cluster)=0.9
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

OUT = "/Users/kevin/code/lingbot-va/.claude/scratch/rf/story_gravity_field.png"
rng = np.random.default_rng(7)

# ---------------- dataset: two 1-D clusters (bimodal), 300 samples each
NC, CS, CC = 300, 0.35, 1.30
x0 = np.concatenate([rng.normal(-CC, CS, NC), rng.normal(+CC, CS, NC)])
clu = np.concatenate([np.zeros(NC, int), np.ones(NC, int)])          # 0 = left, 1 = right
order = np.argsort(x0)
x0, clu = x0[order], clu[order]
N = len(x0)
XS = 0.15                                                            # the reading
SIG = np.exp(np.linspace(np.log(1.0), np.log(0.05), 46))             # sigma, top -> bottom


def weights(prior):
    W = np.zeros((len(SIG), N))
    for i, s in enumerate(SIG):
        w = prior * np.exp(-0.5 * ((XS - (1 - s) * x0) / s) ** 2)
        W[i] = w / w.sum()
    return W


P_UN = np.where(clu == 1, 0.5 / NC, 0.5 / NC)                         # 50 / 50
P_CO = np.where(clu == 1, 0.9 / NC, 0.1 / NC)                         # 90 / 10  (the lens)
Wu, Wc = weights(P_UN), weights(P_CO)
mu_u = Wu @ x0
mu_c = Wc @ x0
v_u = (XS - mu_u) / SIG
v_c = (XS - mu_c) / SIG

print(f"数据集:{N} 个样本,两团中心 ±{CC}(团内 std {CS}),读数 x_σ = {XS}")
for s_target in [1.0, 0.5, 0.2, 0.1, 0.05]:
    i = int(np.argmin(np.abs(SIG - s_target)))
    nu = 1 / np.sum(Wu[i] ** 2); nc = 1 / np.sum(Wc[i] ** 2)
    print(f"  σ={SIG[i]:.3f}: 有效源个数 {nu:7.1f} / {nc:7.1f}   "
          f"加权均值 E[x₀]={mu_u[i]:+.4f} / {mu_c[i]:+.4f}   v={v_u[i]:+8.3f} / {v_c[i]:+8.3f}")

# ---------------- figure
fig = plt.figure(figsize=(13.8, 9.2), dpi=150)
gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 0.95], hspace=0.42, wspace=0.20)

lnorm = LogNorm(vmin=1e-7, vmax=1.0)
heat_axes = []
for col, (W, mu, title) in enumerate([
        (Wu, mu_u, r"(1) unconditional  ($p = 0.5/0.5$)"),
        (Wc, mu_c, r"(2) with the lens  ($p = 0.9/0.1$)")]):
    ax = fig.add_subplot(gs[0, col])
    heat_axes.append(ax)
    im = ax.imshow(W, aspect="auto", origin="upper", cmap="inferno", norm=lnorm,
                   extent=[0, N, np.log10(SIG[0]), np.log10(SIG[-1])])
    ax.set_yticks([np.log10(v) for v in [1.0, 0.5, 0.2, 0.1, 0.05]])
    ax.set_yticklabels(["1.0", "0.5", "0.2", "0.1", "0.05"], fontsize=8)
    if col:
        ax.set_yticklabels([])
    ax.set_title(title + "\nevery row = ONE gravity field at that $\\sigma$", fontsize=10, pad=22)
    ax.set_xlabel("sample index  (sorted by $x_0$)", fontsize=9.5)
    top = ax.secondary_xaxis("top")
    ti = [int(np.argmin(np.abs(x0 - t))) for t in [-2, -1, 0, 1, 2]]
    top.set_xticks(ti); top.set_xticklabels(["-2", "-1", "0", "+1", "+2"], fontsize=8)
    top.set_xlabel("sample position $x_0$", fontsize=8.5)
    mean_idx = [int(np.argmin(np.abs(x0 - m))) for m in mu]
    ax.plot(mean_idx, np.log10(SIG), "o-", color="#2ecc71", ms=3, lw=1.6,
            label="weighted mean $E[x_0]$")
    ax.legend(fontsize=7.8, frameon=False, loc="lower left")
    for c, lab, colr in [(0, "cluster $-1.3$", "#8ab4d8"), (1, "cluster $+1.3$", "#f0b27a")]:
        idx = np.nonzero(clu == c)[0]
        ax.text(idx.mean(), np.log10(SIG[0]) - 0.07, lab, ha="center", fontsize=8, color=colr)
heat_axes[0].set_ylabel(r"$\sigma$:  1.0 (top) $\rightarrow$ 0.05 (bottom)", fontsize=9.5)
cax = fig.add_axes([0.965, 0.60, 0.011, 0.265])
cb = fig.colorbar(im, cax=cax)
cb.set_label("row-normalised weight (log scale)", fontsize=8.5)
cb.ax.tick_params(labelsize=8)

# ---- (3) collapse each row with its weights -> the target
ax = fig.add_subplot(gs[1, 0])
ax.semilogx(SIG, mu_u, "o-", ms=3.5, color="#c0392b", lw=2, label="unconditional  $E[x_0]$")
ax.semilogx(SIG, mu_c, "s-", ms=3.5, color="#2471a3", lw=2, label="with lens  $E[x_0|c]$")
ax.axhline(XS, color="k", ls=":", lw=1.2)
ax.text(0.95, XS + .05, "the reading $x_\\sigma$", fontsize=8)
for c, lab in [(-CC, "cluster $-1.3$"), (CC, "cluster $+1.3$")]:
    ax.axhline(c, color="#bdc3c7", ls="--", lw=1)
    ax.text(0.055, c + .05, lab, fontsize=8, color="#7f8c8d")
ax.annotate("very small $\\sigma$: only the nearest sample\nsurvives, so the cluster-level prior cancels\n(the two curves merge)",
            xy=(0.055, mu_u[-1]), xytext=(0.10, -0.75), fontsize=7.8,
            arrowprops=dict(arrowstyle="->", lw=0.8))
ax.set_xlabel(r"$\sigma$", fontsize=9.5)
ax.set_ylabel("weighted mean of $x_0$  =\n'where the gravity pulls it to'", fontsize=9.5)
ax.set_title(r"(3) collapse the row with its weights:  $\sum_k w_k x_{0,k}$", fontsize=10)
ax.invert_xaxis(); ax.legend(fontsize=8.5, frameon=False, loc="lower right"); ax.tick_params(labelsize=8)

# ---- (4) the model output
ax = fig.add_subplot(gs[1, 1])
ax.semilogx(SIG, -v_u, "o-", ms=3.5, color="#c0392b", lw=2, label="unconditional  $-v$")
ax.semilogx(SIG, -v_c, "s-", ms=3.5, color="#2471a3", lw=2, label="with lens  $-v$")
ax.axhline(0, color="k", lw=1)
ax.set_xlabel(r"$\sigma$", fontsize=9.5)
ax.set_ylabel(r"$-v=\frac{E[x_0]-x_\sigma}{\sigma}$   =  move per unit $\sigma$", fontsize=9.5)
ax.set_title(r"(4) the model output: ONE number per $(\sigma,\,x_\sigma)$", fontsize=10)
ax.invert_xaxis(); ax.legend(fontsize=8.5, frameon=False); ax.tick_params(labelsize=8)
ax.annotate("early ($\\sigma$ large): every sample pulls,\nthey mostly cancel  $\\rightarrow$  small move",
            xy=(0.72, -v_u[7]), xytext=(0.16, 0.55), fontsize=7.8,
            arrowprops=dict(arrowstyle="->", lw=0.8))
ax.annotate("late: one or two samples pull  $\\rightarrow$  big move",
            xy=(0.055, -v_u[-1]), xytext=(0.11, -v_u[-1] * 0.35), fontsize=7.8,
            arrowprops=dict(arrowstyle="->", lw=0.8))

fig.suptitle(y=0.975, t="The gravity field, sample by sample:   x-axis = samples,   y-axis = $\\sigma$;   "
             "each row is one gravity field, and the model output is its weighted direction",
             fontsize=10.5)
fig.subplots_adjust(top=0.855, bottom=0.075, left=0.065, right=0.955)
fig.savefig(OUT, bbox_inches="tight")
print("\nwrote", OUT)
