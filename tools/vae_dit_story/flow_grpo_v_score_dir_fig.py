r"""v 与 score 的方向关系。

关键:`v = dx/dσ` 是【σ 递增(加噪)】方向的速度;生成是 σ 递减,实际运动方向是 −v。
所以"往数据走"的那个方向是 u := −v,而不是 v。

(a) σ=0.5(边缘单峰):score 指向 p_σ 的峰(左),而 u 指向后验均值(右)→ 两者相反
(b) σ=0.25(边缘双峰):score 与 u 同向(都指向 +0.75 那个峰)
(c) 公式分解 + 数值核对
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
MU = np.array([1.0, -1.0]); PR = np.array([0.5, 0.5])


def wts(x, s):
    m = (1 - s) * MU
    lw = np.log(PR) - 0.5 * np.log(2 * np.pi * s ** 2) - (x - m) ** 2 / (2 * s ** 2)
    lw -= lw.max()
    w = np.exp(lw)
    return w / w.sum()


def p_sig(x, s):
    m = (1 - s) * MU
    return float(np.sum(PR * np.exp(-(x - m) ** 2 / (2 * s ** 2)) / np.sqrt(2 * np.pi * s ** 2)))


def score(x, s):
    m = (1 - s) * MU
    return float(wts(x, s) @ (-(x - m) / s ** 2))


def Ex0(x, s):
    return float(wts(x, s) @ MU)


fig = plt.figure(figsize=(15.2, 5.5), dpi=150)
CASES = [(0.5, 0.6), (0.25, 0.6)]
axes = []
for k, (S, X) in enumerate(CASES):
    ax = fig.add_axes([0.055 + k * 0.335, 0.17, 0.28, 0.64])
    axes.append(ax)
    xg = np.linspace(-2.0, 2.0, 800)
    ax.plot(xg, [p_sig(x, S) for x in xg], color=DARK, lw=2.4)
    yy = np.array([p_sig(x, S) for x in xg])
    imax = np.where((yy[1:-1] > yy[:-2]) & (yy[1:-1] > yy[2:]))[0] + 1
    peaks = xg[imax] if len(imax) else xg[[int(np.argmax(yy))]]
    if len(peaks) >= 2 and abs(yy[imax[0]] - yy[imax[1]]) < 1e-3 * yy.max():
        pk_lab = f"\\pm{abs(peaks[0]):.2f}"; pk_line = 0.0
    else:
        pk_lab = f"{peaks[0]:+.2f}"; pk_line = float(peaks[0])
    if pk_line != 0.0 or len(peaks) == 1:
        ax.axvline(pk_line, color=GREY, ls=":", lw=1.1)
    else:
        ax.axvline(0.0, color=GREY, ls=":", lw=1.1)
    ax.text(pk_line if pk_line != 0 else 0.0, yy.max() * 1.20, f"$p_\\sigma$ 的峰\n$={pk_lab}$",
            fontsize=9.2, color="#5d6d7e", ha="center")
    sc, ex, v = score(X, S), Ex0(X, S), (X - Ex0(X, S)) / S
    u = -v
    ax.plot([X], [p_sig(X, S)], "o", ms=9, color=DARK, zorder=6)

    yb = p_sig(X, S) + 0.10
    ax.add_patch(FancyArrowPatch((X, yb), (X + np.sign(sc) * 0.45, yb), arrowstyle="-|>",
                                 mutation_scale=14, lw=2.6, color=BLUE))
    ax.text(X + np.sign(sc) * 0.225, yb + 0.025, f"$\\nabla\\log p={sc:+.2f}$", fontsize=9.6, color=BLUE,
            ha="center", va="bottom")
    yb2 = p_sig(X, S) + 0.19
    ax.add_patch(FancyArrowPatch((X, yb2), (X + np.sign(u) * 0.45, yb2), arrowstyle="-|>",
                                 mutation_scale=14, lw=2.6, color=ORANGE))
    ax.text(X + np.sign(u) * 0.225, yb2 + 0.025, f"$u=-v={u:+.2f}$", fontsize=9.6, color=ORANGE,
            ha="center", va="bottom")
    ax.plot([ex], [0], "*", ms=17, color=GREEN, zorder=6)
    ax.text(0.0, -0.075, f"读数 $x_\\sigma={X:+.2f}$(黑点)　$E[x_0|x_\\sigma]={ex:+.2f}$(绿星)",
            fontsize=9.4, color=DARK, ha="center", va="top")
    same = np.sign(sc) == np.sign(u)
    ax.text(-2.0, max(p_sig(x, S) for x in xg) * 1.28,
            f"σ={S}:score 与 $u$ {'同向' if same else '【相反】'}",
            fontsize=11, color=GREEN if same else RED)
    ax.set_ylim(-0.20, max(p_sig(x, S) for x in xg) * 1.52)
    ax.set_xlim(-2.05, 2.05)
    ax.set_xlabel("$x$", fontsize=10.5)
    if k == 0:
        ax.set_ylabel("$p_\\sigma(x)$", fontsize=10.5)
    ax.set_title(f"({'ab'[k]}) σ={S}:" + ("边缘单峰 → 两方向相反" if not same else "边缘双峰 → 两方向一致"),
                 fontsize=11.2)
    ax.tick_params(labelsize=9)

ax3 = fig.add_axes([0.745, 0.17, 0.235, 0.64]); ax3.axis("off")
ax3.text(0.5, 0.99, "(c) 公式上把两项拆开", ha="center", va="top", fontsize=11.5, color=DARK)
ax3.text(0.0, 0.86,
         "由恒等式 (b):\n"
         "$v=-\\dfrac{x}{1-\\sigma}-\\dfrac{\\sigma}{1-\\sigma}\\nabla\\log p$\n"
         "　收缩项　　　反 score 项\n"
         "取反(生成方向):\n"
         "$u=-v=\\dfrac{x}{1-\\sigma}+\\dfrac{\\sigma}{1-\\sigma}\\nabla\\log p$",
         fontsize=10, color=DARK, va="top")
ax3.text(0.0, 0.42,
         "⇒ $v$ 里含【反 score】分量(加噪把人\n推离高密度);$u$ 里含【正 score】分量\n"
         "(生成往高密度走)。\n"
         "但两项都要加上收缩项 $\\pm x/(1-\\sigma)$,\n所以合矢量方向不一定与 score 平行。",
         fontsize=9.4, color=DARK, va="top")
ax3.text(0.0, 0.05,
         "另:score 指向 $p_\\sigma$ 的峰,\n$u$ 指向 $E[x_0|x_\\sigma]$ —— 两者不是一回事。",
         fontsize=9.4, color=RED, va="top", bbox=dict(fc="#fdecea", ec="#e6a9a0"))

fig.suptitle("$v$ 与 $\\nabla\\log p$ 的方向关系:$v$ 是【加噪方向】的速度,所以带负号", fontsize=13, y=0.995)
fig.savefig(RF + "fg_v_vs_score.png", bbox_inches="tight")

for S, X in [(0.5, 0.6), (0.25, 0.6), (0.5, 0.1)]:
    sc, ex, v = score(X, S), Ex0(X, S), (X - Ex0(X, S)) / S
    xg = np.linspace(-2, 2, 4001)
    yy = np.array([p_sig(x, S) for x in xg])
    imax = np.where((yy[1:-1] > yy[:-2]) & (yy[1:-1] > yy[2:]))[0] + 1
    pk = ", ".join(f"{xg[i]:+.2f}" for i in imax) if len(imax) else f"{xg[int(np.argmax(yy))]:+.2f}"
    print(f"σ={S}, x={X}: ∇logp={sc:+.4f}  v={v:+.4f}  u=-v={-v:+.4f}  E[x0]={ex:+.4f}  "
          f"p_σ峰在 {pk}  {'同向' if np.sign(sc)==np.sign(-v) else '相反'}")
print("wrote", RF + "fg_v_vs_score.png")
