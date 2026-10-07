"""A contour-map view of the gravity field (light background).

In 2-D latent space:
  * every sample is a POINT SOURCE (a mass) at x0_k
  * at noise level sigma each source emits a Gaussian  N(x ; (1-sigma) x0_k , sigma^2 I)
  * their prior-weighted sum is the DENSITY FIELD rho_sigma(x)      -> the CONTOUR LINES
  * E[x0 | x] is the weighted mean of the sources                   -> the ARROW FIELD
        v(x) = (x - E[x0|x]) / sigma
  * a particle that rides the field traces a STREAMLINE.

Light colormap on purpose: contours are easier to read on white.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import pathlib as _pl
OUT = str(_pl.Path(__file__).resolve().parent / "docs" / "story_gravity_contour.png")
rng = np.random.default_rng(11)

# ---------------- 3 clusters of samples in the plane (= the "masses")
CENTERS = np.array([[-1.55, -0.55], [1.55, -0.45], [0.05, 1.55]])
PER, CS = 60, 0.22
x0 = np.concatenate([c + CS * rng.normal(size=(PER, 2)) for c in CENTERS])
lab = np.repeat(np.arange(3), PER)
prior = np.full(len(x0), 1.0 / len(x0))
XS = np.array([0.10, -0.15])                     # the reading x_sigma
Z0 = np.array([1.55, 2.10])                      # where the particle starts (sigma = 1)

LO, HI, NG = -2.5, 2.5, 200
gx = np.linspace(LO, HI, NG)
GX, GY = np.meshgrid(gx, gx)
PTS = np.stack([GX.ravel(), GY.ravel()], 1)


def fields(s):
    """density rho and velocity v on the grid, at noise level s"""
    c = (1 - s) * x0                                     # centres of the bells
    d2 = ((PTS[:, None, :] - c[None, :, :]) ** 2).sum(-1)
    w = prior[None, :] * np.exp(-0.5 * d2 / s ** 2)
    tot = w.sum(1)
    rho = (tot / tot.max()).reshape(NG, NG)
    ex0 = (w @ x0) / tot[:, None]                        # E[x0 | x]
    v = (PTS - ex0) / s
    return rho, v.reshape(NG, NG, 2), c


SIGS = [1.0, 0.6, 0.35, 0.15]
SIGGRID = np.exp(np.linspace(np.log(1.0), np.log(0.05), 900))
traj = [Z0.copy()]
x = Z0.copy()
for i in range(len(SIGGRID) - 1):
    s = SIGGRID[i]
    rho, v, c = fields(s)
    # single-particle field value (exact, cheap): recompute at that point
    d2 = ((x[None, :] - c) ** 2).sum(-1)
    w = prior * np.exp(-0.5 * d2 / s ** 2)
    ex0 = (w[:, None] * x0).sum(0) / w.sum()
    x = x + (x - ex0) / s * (SIGGRID[i + 1] - SIGGRID[i])
    traj.append(x.copy())
traj = np.array(traj)

print(f"{len(x0)} samples in 3 clusters; reading x_sigma = {XS}")
print(f"{'sigma':>6} {'rho at reading':>15} {'E[x0] at reading':>18} {'v at reading':>18} {'max|v|':>9}")
for s in SIGS:
    C = (1 - s) * x0
    d2 = ((XS[None, :] - C) ** 2).sum(-1)
    w = prior * np.exp(-0.5 * d2 / s ** 2)
    ex0 = (w[:, None] * x0).sum(0) / w.sum()
    v = (XS - ex0) / s
    rho, V, _ = fields(s)
    print(f"{s:>6} {rho.max():>15.3f} {ex0[0]:>+9.3f},{ex0[1]:>+7.3f} {v[0]:>+9.2f},{v[1]:>+7.2f} {np.abs(V).max():>9.1f}")

# ============================================================ figure
fig, axes = plt.subplots(1, 4, figsize=(17.0, 4.9), dpi=140)
levels = 10 ** np.linspace(-3.2, 0, 14)
CMAP = plt.get_cmap("YlOrBr")
for ax, s in zip(axes, SIGS):
    rho, V, C = fields(s)
    ax.contourf(gx, gx, rho, levels=levels, cmap=CMAP, alpha=0.85, norm=LogNorm())
    ax.contour(gx, gx, rho, levels=levels, colors="#8a6d3b", linewidths=0.45, alpha=0.75)
    # arrow field
    st = 13
    qx, qy = gx[::st], gx[::st]
    VX, VY = V[::st, ::st, 0], V[::st, ::st, 1]
    mag = np.hypot(VX, VY)
    keep = mag > mag.max() * 0.06
    sc = 0.40 / max(mag.max(), 1e-9)
    ax.quiver(qx[None, :].repeat(len(qy), 0)[keep], qy[:, None].repeat(len(qx), 1)[keep],
              VX[keep] * sc, VY[keep] * sc, color="#2c3e50", angles="xy",
              scale_units="xy", scale=1, width=0.0035, alpha=0.8)
    # the masses
    for k, col in zip(range(3), ["#c0392b", "#2471a3", "#1e8449"]):
        m = lab == k
        ax.plot(x0[m, 0], x0[m, 1], "o", ms=3.4, color=col, mec="w", mew=0.5, alpha=0.95)
    # the reading
    ax.plot(*XS, marker="*", ms=15, color="#111", mec="w", mew=0.8, zorder=6)
    # the particle trail
    m = SIGGRID >= s
    ax.plot(traj[m, 0], traj[m, 1], "-", color="#111", lw=1.6, alpha=0.85, zorder=5)
    j = int(np.argmin(np.abs(SIGGRID - s)))
    ax.plot(*traj[j], "o", ms=6.5, color="w", mec="#111", mew=1.6, zorder=7)
    ax.set_xlim(LO, HI); ax.set_ylim(LO, HI); ax.set_aspect("equal")
    ax.set_title(f"$\\sigma$ = {s}      max$|v|$ = {np.abs(V).max():.1f}", fontsize=11)
    ax.tick_params(labelsize=8)
    ax.set_xlabel("$x_1$", fontsize=9.5)
axes[0].set_ylabel("$x_2$", fontsize=9.5)
axes[0].text(-2.35, 2.30, "$\\sigma$=1: every source's bell sits on the origin\n"
             "→ one circular blob; the arrows all point inward",
             fontsize=8.4, color="#4e342e", va="top",
             bbox=dict(fc="w", ec="#d7ccc8", alpha=0.92))
axes[-1].text(-2.35, 2.30, "$\\sigma$ small: blobs shrink onto the samples\n"
              "→ arrows point at the nearest ones",
              fontsize=8.4, color="#4e342e", va="top",
              bbox=dict(fc="w", ec="#d7ccc8", alpha=0.92))
fig.suptitle("The gravity map as a contour plot:  dots = samples (masses),  rings = level sets of the density field "
             "(darker = higher, each panel normalised to its own max),\n"
             "arrows = the field $v(x)=(x-E[x_0|x])/\\sigma$,  star = the reading $x_\\sigma$,  "
             "white dot + trail = one particle riding the field", fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, 0.895])
fig.savefig(OUT, bbox_inches="tight")
print("\nwrote", OUT)
