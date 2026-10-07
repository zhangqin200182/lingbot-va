"""From density field to velocity.

The chain, panel by panel:
  (1) every sample owns a GAUSSIAN DENSITY FIELD  N(x ; (1-sigma) x0 , sigma^2)
      -> in the (x, sigma) plane each sample is a straight world-line x = (1-sigma) x0
         whose width is sigma; at sigma = 1 all bells collapse onto x = 0 (no information).
  (2) slice at one sigma: the individual bells (scaled by ONE common factor) + the total
      density; the height AT the reading x_sigma is the LIKELIHOOD of each sample.
  (3) weight  ∝  likelihood x prior, then normalise  ->  the prior shows up as the SLOPE.
  (4) weight x direction, summed  ->  the VELOCITY the network must output.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pathlib as _pl
OUT = str(_pl.Path(__file__).resolve().parent / "docs" / "story_density_field.png")
rng = np.random.default_rng(7)

# ---------------- dataset: two 1-D clusters
NC, CS, CC = 300, 0.35, 1.30
x0 = np.concatenate([rng.normal(-CC, CS, NC), rng.normal(+CC, CS, NC)])
clu = np.concatenate([np.zeros(NC, int), np.ones(NC, int)])
order = np.argsort(x0)
x0, clu = x0[order], clu[order]
N = len(x0)
P0 = 0.5 / NC                                    # unconditional prior, per sample
P1 = np.where(clu == 1, 0.9 / NC, 0.1 / NC)      # with the "lens"
XS = 0.15                                        # the reading
XLIM = (-2.3, 2.3)
SLICE = 0.35                                     # the sigma we zoom into
SIG = np.exp(np.linspace(np.log(1.0), np.log(0.02), 60))


def lik_at(x, s):
    return np.exp(-0.5 * ((x - (1 - s) * x0) / s) ** 2)


def weights_at(s, prior):
    w = prior * lik_at(XS, s)
    return w / w.sum()


lk = lik_at(XS, SLICE)
w_un = weights_at(SLICE, P0)
w_co = weights_at(SLICE, P1)
mu = float(w_un @ x0)
v = (XS - mu) / SLICE
eff = 1 / np.sum(w_un ** 2)

# ============================================================ figure
fig = plt.figure(figsize=(14.2, 9.6), dpi=150)
gs = fig.add_gridspec(2, 3, height_ratios=[1.22, 1.0], hspace=0.42, wspace=0.30)

# ---------- (1) the density field in (x, sigma)
xs = np.linspace(*XLIM, 460)
field = np.zeros((len(SIG), len(xs)))
for i, s in enumerate(SIG):
    d = np.sum(np.exp(-0.5 * ((xs[:, None] - (1 - s) * x0[None, :]) / s) ** 2), axis=1)
    field[i] = d / d.max()

ax = fig.add_subplot(gs[0, :])
ax.imshow(field, aspect="auto", origin="upper", cmap="magma",
          extent=[XLIM[0], XLIM[1], np.log10(SIG[0]), np.log10(SIG[-1])], vmin=0, vmax=1)
for c, colr in [(0, "#5dade2"), (1, "#f5b041")]:
    for k in np.nonzero(clu == c)[0][::4]:
        ax.plot((1 - SIG) * x0[k], np.log10(SIG), color=colr, lw=0.35, alpha=0.5)
k_demo = np.nonzero(clu == 1)[0][40]
for s in [0.75, 0.28]:
    cxd = (1 - s) * x0[k_demo]
    ax.plot([cxd - s, cxd + s], [np.log10(s)] * 2, color="#2ecc71", lw=2.4, solid_capstyle="butt")
    ax.plot([cxd], [np.log10(s)], "o", color="#2ecc71", ms=3)
ax.annotate("green bar = $\\pm\\sigma$:\nthe width of ONE sample's bell",
            xy=(1.72, np.log10(0.28)), xytext=(0.95, np.log10(0.10)), fontsize=8.2, color="#2ecc71",
            arrowprops=dict(arrowstyle="->", lw=0.9, color="#2ecc71"))
ax.axvline(XS, color="#2ecc71", ls="--", lw=1.5)
ax.text(XS + 0.04, np.log10(0.024), "the reading $x_\\sigma$", fontsize=8.5, color="#2ecc71")
ax.axhline(np.log10(SLICE), color="#fff", ls=":", lw=1.3)
ax.text(XLIM[0] + 0.06, np.log10(SLICE) + 0.045, f"$\\sigma$ = {SLICE}   (the slice below)",
        fontsize=8.5, color="#fff")
ax.set_yticks([np.log10(v) for v in [1.0, 0.5, 0.3, 0.15, 0.08, 0.04, 0.02]])
ax.set_yticklabels(["1.0", "0.5", "0.3", "0.15", "0.08", "0.04", "0.02"], fontsize=8)
ax.set_ylabel("$\\sigma$", fontsize=10)
ax.set_xlabel("$x$   (latent value space)", fontsize=9.5)
ax.set_title("(1) every sample owns a GAUSSIAN DENSITY FIELD  $N(x\\,;\\,(1-\\sigma)x_0,\\,\\sigma^2)$;  "
             "in the $(x,\\sigma)$ plane that is a straight world-line of width $\\sigma$\n"
             "at $\\sigma=1$ all bells sit on $x=0$ (no information); going down they fan out "
             "→ the later the slice, the fewer samples still matter", fontsize=10.5)
ax.text(-2.25, np.log10(0.93), "cluster $-1.3$", color="#5dade2", fontsize=8.5)
ax.text(2.25, np.log10(0.93), "cluster $+1.3$", color="#f5b041", fontsize=8.5, ha="right")

# ---------- (2) the slice
ax = fig.add_subplot(gs[1, 0])
xs2 = np.linspace(*XLIM, 900)
tot = np.zeros_like(xs2)
for k in range(N):
    tot += P0 * np.exp(-0.5 * ((xs2 - (1 - SLICE) * x0[k]) / SLICE) ** 2)
SCALE = 220.0
show = np.argsort(-w_un)[:26]
for k in show:
    ax.plot(xs2, SCALE * P0 * np.exp(-0.5 * ((xs2 - (1 - SLICE) * x0[k]) / SLICE) ** 2),
            color=("#f5b041" if clu[k] else "#5dade2"), lw=0.9, alpha=0.8)
    ax.plot([XS], [SCALE * P0 * lk[k]], "o", color=("#f5b041" if clu[k] else "#5dade2"), ms=3)
ax.plot(xs2, tot, color="#111", lw=2.6, label="total density $p(x_\\sigma\\,|\\,\\sigma)$")
ax.axvline(XS, color="#2ecc71", ls="--", lw=1.6)
ax.annotate("marker height =\nLIKELIHOOD of that sample", xy=(XS, SCALE * P0 * lk.max() * 0.75),
            xytext=(0.04, 0.72), textcoords="axes fraction", fontsize=8,
            arrowprops=dict(arrowstyle="->", lw=0.9))
ax.text(0.04, 0.90, f"individual bells $\\times${SCALE:.0f} (same factor for all)",
        transform=ax.transAxes, fontsize=7.6, color="#7f8c8d", va="top")
ax.set_xlabel("$x$", fontsize=9.5); ax.set_ylabel(f"density   (bells $\\times${SCALE:.0f})", fontsize=9)
ax.set_title(f"(2) slice at $\\sigma$ = {SLICE}:  each bell is one sample", fontsize=10)
ax.legend(fontsize=8, frameon=False, loc="upper right"); ax.tick_params(labelsize=8)

# ---------- (3) weight = likelihood x prior
ax = fig.add_subplot(gs[1, 1])
sum_un = float(P0 * lk.sum())
sum_co = float(np.sum(P1 * lk))
slope_un = (0.5 / NC) / sum_un                       # both clusters share this slope
slope_co1 = (0.9 / NC) / sum_co                      # cluster +1.3 with the lens
slope_co0 = (0.1 / NC) / sum_co                      # cluster -1.3 with the lens
xx = np.array([0, lk.max() * 1.02])
ax.plot(xx, slope_un * xx, "--", color="#7f8c8d", lw=1.2,
        label="equal prior: one line, slope %.2fe-3" % (0.5 / NC / sum_un * 1e3))
ax.plot(xx, slope_co1 * xx, "--", color="#b9770e", lw=1.2,
        label="lens, cluster $+1.3$: slope %.2fe-3" % (0.9 / NC / sum_co * 1e3))
ax.plot(xx, slope_co0 * xx, "--", color="#1a5276", lw=1.2,
        label="lens, cluster $-1.3$: slope %.2fe-3" % (0.1 / NC / sum_co * 1e3))
for k in range(N):
    if lk[k] < 1e-5:
        continue
    col = "#f5b041" if clu[k] else "#5dade2"
    ax.plot(lk[k], w_un[k], "o", ms=3.0, color=col, alpha=0.7)
    ax.plot(lk[k], w_co[k], "s", ms=2.6, mfc="none", mec=col, alpha=0.85)
ax.set_xlabel("likelihood  (height of that sample's bell AT the reading)", fontsize=9)
ax.set_ylabel("posterior weight", fontsize=9)
ax.set_title("(3) weight $\\propto$ likelihood $\\times$ prior:\n"
             "filled = equal prior,  hollow = the lens;  dashed = the slope (= prior / $\\Sigma$)",
             fontsize=9.4)
ax.legend(fontsize=7.2, frameon=False, loc="upper left")
ax.tick_params(labelsize=8)

# ---------- (4) weight x direction -> velocity
ax = fig.add_subplot(gs[1, 2])
keep = np.nonzero(w_un > w_un.max() / 40)[0]
ax.vlines(x0[keep], 0, w_un[keep] / w_un.max(),
          color=["#f5b041" if clu[k] else "#5dade2" for k in keep], lw=1.8)
ax.axvline(XS, color="#2ecc71", ls="--", lw=1.5)
ax.axvline(mu, color="#1e8449", lw=2.4)
ax.annotate("", xy=(mu, 1.13), xytext=(XS, 1.13),
            arrowprops=dict(arrowstyle="-|>", lw=2.6, color="#1e8449"))
ax.text((XS + mu) / 2, 1.19, "weighted direction", ha="center", fontsize=8.5, color="#1e8449")
ax.text(XS, -0.14, "$x_\\sigma$", ha="center", fontsize=8.5, color="#2ecc71")
ax.text(mu, -0.14, f"$E[x_0]$ = {mu:+.3f}", ha="center", fontsize=8.5, color="#1e8449")
ax.set_ylim(-0.24, 1.30); ax.set_xlim(XLIM)
ax.set_xlabel("sample value $x_0$   (bar = its weight)", fontsize=9)
ax.set_ylabel("weight", fontsize=9)
ax.set_title(f"(4) $\\sum_k w_k\\,x_{{0,k}}$ → velocity  $v = (x_\\sigma - E[x_0])/\\sigma = {v:+.3f}$",
             fontsize=9.8)
ax.tick_params(labelsize=8)
ax.text(0.03, 0.90, f"{len(keep)} of {N} samples carry visible weight\neffective #sources = {eff:.0f}",
        transform=ax.transAxes, fontsize=7.6, color="#5d6d7e", va="top",
        bbox=dict(fc="w", ec="none", alpha=0.75))

fig.suptitle("From density field to velocity:  samples  →  one Gaussian bell each, at every $\\sigma$  →  "
             "height at the reading = likelihood  →  $\\times$ prior  →  weights  →  weighted direction",
             fontsize=11)
fig.subplots_adjust(top=0.875, bottom=0.075, left=0.055, right=0.985)
fig.savefig(OUT, bbox_inches="tight")

print(f"dataset {N} samples (two clusters at ±{CC}, within-cluster std {CS}); reading x_sigma = {XS}")
print(f"slice sigma = {SLICE}:  E[x0] = {mu:+.4f}   v = {v:+.4f}   effective sources = {eff:.1f}")
print(f"   top sample: x0={x0[show[0]]:+.3f}  likelihood={lk[show[0]]:.4f}  weight={w_un[show[0]]:.5f}")
print(f"   conditional weights: E[x0]={float(w_co@x0):+.4f}  v={(XS-float(w_co@x0))/SLICE:+.4f}")
print("wrote", OUT)
