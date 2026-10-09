r"""Step 2 的两个恒等式,配图。

(a) ∇log p_σ(x) 是什么:密度上升方向(在峰值处为 0)
(b) 恒等式 (a):读数 x → 沿 score 走 σ² → 再除以 (1−σ) → 得到 E[x₀|x]
(c) 恒等式 (b):从模型输出 v 反算 score,与直接对密度求导的结果一致
"""
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
for _f in ["/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
           "/System/Library/Fonts/STHeiti Medium.ttc"]:
    try:
        font_manager.fontManager.addfont(_f)
        matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=_f).get_name()
        break
    except Exception:
        continue
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

RED, BLUE, GREEN, DARK, GREY, ORANGE = "#c0392b", "#2471a3", "#1e8449", "#2c3e50", "#95a5a6", "#b9770e"
RF = "/Users/kevin/code/lingbot-va/.claude/scratch/rf/"

# 两原子数据:p_data = 0.5 δ(+1) + 0.5 δ(−1)
MU = np.array([1.0, -1.0]); PR = np.array([0.5, 0.5]); SIG = 0.5


def weights(x):
    m = (1 - SIG) * MU
    lw = np.log(PR) - 0.5 * np.log(2 * np.pi * SIG ** 2) - (x - m) ** 2 / (2 * SIG ** 2)
    lw = lw - lw.max()
    return np.exp(lw) / np.exp(lw).sum()


def p_sig(x):
    m = (1 - SIG) * MU
    return float(np.sum(PR * np.exp(-(x - m) ** 2 / (2 * SIG ** 2)) / np.sqrt(2 * np.pi * SIG ** 2)))


def score(x):
    m = (1 - SIG) * MU
    return float(weights(x) @ (-(x - m) / SIG ** 2))


def Ex0(x):
    return float(weights(x) @ MU)


def vstar(x):
    return (x - Ex0(x)) / SIG


fig = plt.figure(figsize=(15.0, 5.9), dpi=150)

# ---------------- (a) score 是密度上升方向
ax = fig.add_axes([0.05, 0.19, 0.29, 0.60])
xg = np.linspace(-2.0, 2.0, 700)
ax.plot(xg, [p_sig(x) for x in xg], color=DARK, lw=2.2)
for x0 in [-1.2, -0.6, 0.0, 0.6, 1.2]:
    sc = score(x0)
    ax.plot([x0], [p_sig(x0)], "o", ms=7, color=GREEN if abs(sc) < 1e-6 else BLUE, zorder=6)
    if abs(sc) > 1e-6:
        L = 0.30 * np.tanh(abs(sc) / 2.0)
        ax.add_patch(FancyArrowPatch((x0, p_sig(x0) + 0.06), (x0 + np.sign(sc) * L, p_sig(x0) + 0.06),
                                     arrowstyle="-|>", mutation_scale=13, lw=2.2, color=BLUE))
        ax.text(x0, p_sig(x0) + 0.135, f"{sc:+.2f}", fontsize=8.6, color=BLUE, ha="center")
ax.axvline(0, color=GREY, ls=":", lw=1.0)
ax.text(0.05, p_sig(0) - 0.085, "峰值处 score=0", fontsize=9.4, color=GREEN, ha="left")
ax.set_xlabel("$x$", fontsize=10.5); ax.set_ylabel("$p_\\sigma(x)$", fontsize=10.5)
ax.set_title("(a) $\\nabla\\log p_\\sigma(x)$ = 密度上升方向", fontsize=11.5)
ax.tick_params(labelsize=9)

# ---------------- (b) 恒等式 (a)
ax2 = fig.add_axes([0.40, 0.19, 0.27, 0.60])
X = 0.6
sc, ex = score(X), Ex0(X)
walk = X + SIG ** 2 * sc
ax2.axhline(0, color=GREY, lw=0.9)
ax2.plot([X], [0], "o", ms=13, color=GREEN, zorder=6)
ax2.text(X - 0.04, -0.085, f"读数 $x_\\sigma={X:+.2f}$", fontsize=10.2, color=GREEN, ha="right", va="top")
ax2.add_patch(FancyArrowPatch((X, 0.34), (walk, 0.34), arrowstyle="-|>", mutation_scale=14,
                              lw=2.6, color=BLUE))
ax2.text((X + walk) / 2, 0.40, f"$+\\sigma^2\\nabla\\log p={SIG**2*sc:+.4f}$", fontsize=9.6,
         color=BLUE, ha="center")
ax2.plot([walk], [0.34], "|", ms=14, color=BLUE, mew=2.2)
ax2.add_patch(FancyArrowPatch((walk, -0.30), (ex, -0.30), arrowstyle="-|>", mutation_scale=16,
                              lw=2.8, color=RED))
ax2.text((walk + ex) / 2, -0.38, f"$\\div(1-\\sigma)=\\div{SIG}$", fontsize=9.8, color=RED,
         ha="center", va="top")
ax2.plot([ex], [0], "o", ms=13, color=RED, zorder=6)
ax2.text(ex + 0.03, 0.085, f"$E[x_0|x_\\sigma]={ex:+.4f}$", fontsize=10.2, color=RED, ha="left", va="bottom")
ax2.text(-0.20, 0.86,
         f"先沿 score 走:$x_\\sigma\\to x_\\sigma+\\sigma^2\\nabla\\log p={walk:+.4f}$(score<0,所以往左)\n"
         f"再 $\\div(1-\\sigma)$:$\\to E[x_0|x_\\sigma]={ex:+.4f}$\n"
         f"直接核验:${0.9168:.4f}(+1)+{0.0832:.4f}(-1)={ex:+.4f}$ ✓",
         fontsize=9.3, color=DARK, va="top", bbox=dict(fc="#eef7f1", ec="#a9d5bb"))
ax2.set_xlim(-0.24, 1.10); ax2.set_ylim(-0.72, 1.02)
ax2.set_yticks([]); ax2.set_xlabel("$x$ / $x_0$", fontsize=10.5)
ax2.set_title("(b) 恒等式 (a):\n$(1-\\sigma)E[x_0|x] = x + \\sigma^2\\nabla\\log p$", fontsize=11.2)
ax2.tick_params(labelsize=9)

# ---------------- (c) 恒等式 (b):v 里已经含 score
ax3 = fig.add_axes([0.735, 0.19, 0.245, 0.60])
ax3.axis("off")
XS = [0.6, 0.1, -0.5]
rows = []
for x in XS:
    rows.append((x, score(x), Ex0(x), vstar(x), -(x + (1 - SIG) * vstar(x)) / SIG))
tbl = [["$x_\\sigma$", "直接算\n$\\nabla\\log p$", "由 $v$ 反算\n$-\\frac{x+(1-\\sigma)v}{\\sigma}$"],
       *[[f"{r[0]:+.2f}", f"{r[1]:+.4f}", f"{r[4]:+.4f}"] for r in rows]]
t = ax3.table(cellText=tbl, loc="upper center", cellLoc="center",
              colWidths=[0.22, 0.36, 0.42])
t.auto_set_font_size(False); t.set_fontsize(8.6); t.scale(1.0, 1.55)
for (r, c), cell in t.get_celld().items():
    if r == 0:
        cell.set_facecolor("#eef2f7")
ax3.text(0.5, 0.06,
         "(c) 恒等式 (b):両列完全一致\n"
         "$\\nabla\\log p_\\sigma(x)=-\\dfrac{x+(1-\\sigma)v(x,\\sigma)}{\\sigma}$\n"
         "⇒ 做 SDE 不需要第二个网络",
         transform=ax3.transAxes, ha="center", va="bottom", fontsize=9.8, color=DARK)

fig.suptitle("Step 2 的两个恒等式:score 是什么、它和模型的 $v$ 是什么关系", fontsize=13)
fig.savefig(RF + "fg_tweedie_two_identities.png", bbox_inches="tight")

print("σ =", SIG, " 数据 = 两原子 ±1(先验各 0.5)")
print(f"{'x':>6} {'权重(+1/−1)':>18} {'直接算 ∇logp':>14} {'E[x0|x]':>10} {'v*':>10} {'由v反算 ∇logp':>16}")
for x in [0.6, 0.1, 0.0, -0.5]:
    w = weights(x)
    print(f"{x:>6.2f} {w[0]:>8.4f}/{w[1]:<8.4f} {score(x):>14.4f} {Ex0(x):>10.4f} "
          f"{vstar(x):>10.4f} {-(x+(1-SIG)*vstar(x))/SIG:>16.4f}")
print("\n恒等式(a)核验 (1−σ)E[x0|x] ?= x + σ²∇logp :")
for x in [0.6, 0.1, -0.5]:
    print(f"  x={x:+.2f}: {(1-SIG)*Ex0(x):+.6f} vs {x+SIG**2*score(x):+.6f}")
print("wrote", RF + "fg_tweedie_two_identities.png")
