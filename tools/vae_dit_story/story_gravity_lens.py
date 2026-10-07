"""Contours with and without the "lens" (conditioning).

The lens changes the PRIOR over the sources  ->  which changes
  * the density field itself   (the contour rings move towards the favoured cluster),
  * the velocity field         (the arrows swing towards it),
  * and therefore the trajectory: the SAME starting point can end in a DIFFERENT cluster.

Same samples, same noise levels; only the prior differs.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import pathlib as _pl
OUT = str(_pl.Path(__file__).resolve().parent / "docs" / "story_gravity_lens.png")
rng = np.random.default_rng(11)

CENTERS = np.array([[-1.55, -0.55], [1.55, -0.45], [0.05, 1.55]])
PER, CS = 60, 0.22
x0 = np.concatenate([c + CS * rng.normal(size=(PER, 2)) for c in CENTERS])
lab = np.repeat(np.arange(3), PER)
N = len(x0)

P_UN = np.full(N, 1.0 / N)                                    # no condition: every sample equal
LENS = np.array([0.1, 0.8, 0.1])                              # the lens favours cluster #1 (right)
P_CO = LENS[lab] / (LENS[lab].sum() * PER)                    # normalised
XS = np.array([0.10, -0.15])

LO, HI, NG = -2.5, 2.5, 200
gx = np.linspace(LO, HI, NG)
GX, GY = np.meshgrid(gx, gx)
PTS = np.stack([GX.ravel(), GY.ravel()], 1)
SIGGRID = np.exp(np.linspace(np.log(1.0), np.log(0.04), 1600))


def field_at(p, s, prior):
    c = (1 - s) * x0
    w = prior * np.exp(-0.5 * ((p[None, :] - c) ** 2).sum(-1) / s ** 2)
    ex0 = (w[:, None] * x0).sum(0) / w.sum()
    return ex0


def grids(s, prior):
    c = (1 - s) * x0
    d2 = ((PTS[:, None, :] - c[None, :, :]) ** 2).sum(-1)
    w = prior[None, :] * np.exp(-0.5 * d2 / s ** 2)
    tot = w.sum(1)
    rho = (tot / tot.max()).reshape(NG, NG)
    ex0 = (w @ x0) / tot[:, None]
    v = (PTS - ex0) / s
    return rho, v.reshape(NG, NG, 2)


def trajectory(z, prior):
    out = [z.copy()]
    x = z.copy()
    for i in range(len(SIGGRID) - 1):
        s = SIGGRID[i]
        x = x + (x - field_at(x, s, prior)) / s * (SIGGRID[i + 1] - SIGGRID[i])
        out.append(x.copy())
    return np.array(out)


# ---- pick a start whose destination DIFFERS between the two priors
best = None
for zx in np.linspace(-2.0, 2.0, 21):
    for zy in np.linspace(-2.0, 2.0, 21):
        z = np.array([zx, zy])
        e1 = trajectory(z, P_UN)[-1]
        e2 = trajectory(z, P_CO)[-1]
        c1 = int(np.argmin(((CENTERS - e1) ** 2).sum(1)))
        c2 = int(np.argmin(((CENTERS - e2) ** 2).sum(1)))
        if c1 != c2:
            d = np.hypot(*(e2 - e1))
            if best is None or d > best[0]:
                best = (d, z.copy(), e1, e2, c1, c2)
d, Z0, E1, E2, C1, C2 = best
T_UN, T_CO = trajectory(Z0, P_UN), trajectory(Z0, P_CO)
print(f"start z = ({Z0[0]:+.2f}, {Z0[1]:+.2f})")
print(f"  no condition -> ({E1[0]:+.2f}, {E1[1]:+.2f})  = cluster #{C1}")
print(f"  with lens    -> ({E2[0]:+.2f}, {E2[1]:+.2f})  = cluster #{C2}")
print(f"  same start, different destination; separation {d:.2f}")
for s in [0.35, 0.10]:
    for nm, pr in [("uncond", P_UN), ("lens", P_CO)]:
        ex0 = field_at(XS, s, pr)
        print(f"  sigma={s} {nm:6s}: E[x0] at reading = ({ex0[0]:+.3f},{ex0[1]:+.3f})")

# ============================================================ figure
levels = 10 ** np.linspace(-3.2, 0, 14)
fig, axes = plt.subplots(2, 2, figsize=(11.6, 11.2), dpi=150)
for row, s in enumerate([0.35, 0.10]):
    for col, (prior, tag, tcol) in enumerate([(P_UN, "no condition", "#c0392b"),
                                              (P_CO, "with the lens  $p=0.1/0.8/0.1$", "#2471a3")]):
        ax = axes[row, col]
        rho, V = grids(s, prior)
        ax.contourf(gx, gx, rho, levels=levels, cmap="YlOrBr", alpha=0.85, norm=LogNorm())
        ax.contour(gx, gx, rho, levels=levels, colors="#8a6d3b", linewidths=0.45, alpha=0.7)
        st = 13
        VX, VY = V[::st, ::st, 0], V[::st, ::st, 1]
        mag = np.hypot(VX, VY)
        keep = mag > mag.max() * 0.06
        sc = 0.38 / max(mag.max(), 1e-9)
        QX, QY = np.meshgrid(gx[::st], gx[::st])
        ax.quiver(QX[keep], QY[keep], VX[keep] * sc, VY[keep] * sc, color="#2c3e50",
                  angles="xy", scale_units="xy", scale=1, width=0.0032, alpha=0.75)
        # basins: which cluster does the field at this sigma point to?
        c = (1 - s) * x0
        d2 = ((PTS[:, None, :] - c[None, :, :]) ** 2).sum(-1)
        w = prior[None, :] * np.exp(-0.5 * d2 / s ** 2)
        ex0 = (w @ x0) / w.sum(1)[:, None]
        nearest = ((PTS[:, None, :] - CENTERS[None, :, :]) ** 2).sum(-1)
        # (skip explicit basin shading: the contours already carry it)
        for k, colr in zip(range(3), ["#c0392b", "#2471a3", "#1e8449"]):
            m = lab == k
            ax.plot(x0[m, 0], x0[m, 1], "o", ms=3.2, color=colr, mec="w", mew=0.5, alpha=0.95)
        k = int(np.argmin(np.abs(SIGGRID - s)))
        for T, c2, ls in [(T_UN, "#c0392b", "-"), (T_CO, "#2471a3", "-")]:
            ax.plot(T[:, 0], T[:, 1], ls, color=c2, lw=1.6, alpha=0.9, zorder=5)
            ax.plot(*T[k], "o", ms=6.0, color=c2, mec="w", mew=1.3, zorder=6)
        ax.plot(*Z0, marker="X", ms=10, color="#111", mec="w", mew=1.0, zorder=7)
        ax.plot(*XS, marker="*", ms=15, color="#111", mec="w", mew=0.8, zorder=6)
        ax.set_xlim(LO, HI); ax.set_ylim(LO, HI); ax.set_aspect("equal")
        ax.set_title(f"$\\sigma$ = {s}   ·   {tag}\nmax$|v|$ = {np.abs(V).max():.1f}", fontsize=10.5)
        ax.tick_params(labelsize=8)
        if row == 1:
            ax.set_xlabel("$x_1$", fontsize=9.5)
        if col == 0:
            ax.set_ylabel("$x_2$", fontsize=9.5)
axes[0, 0].text(LO + 0.08, HI - 0.10, "X = start (same for both)\nsolid red = trajectory without condition\n"
                "solid blue = trajectory with the lens\nstar = the reading $x_\\sigma$",
                fontsize=8.4, va="top", bbox=dict(fc="w", ec="#d7ccc8", alpha=0.93))
axes[0, 1].text(LO + 0.08, HI - 0.10,
                "the lens bends the contours and the arrows\ntowards the favoured cluster — and the blue\n"
                "trajectory ends in a different cluster",
                fontsize=8.4, va="top", bbox=dict(fc="w", ec="#d7ccc8", alpha=0.93))
fig.suptitle("Contours with and without the lens:  conditioning re-weights the sources, so the DENSITY FIELD "
             "(rings), the VELOCITY FIELD (arrows) and the TRAJECTORY all change", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT, bbox_inches="tight")
print("\nwrote", OUT)
