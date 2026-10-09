r"""∂_σ p = −∂_x F 是什么意思。

取 σ=0.7(此时加噪会让分布【变宽】),高斯数据 N(0,1) → V(σ)=(1−σ)²+σ²,σ=0.7 时 V=0.58
  f(x)=0.69x(向外扩)  F=f·p(通量)  ∂_σ p = −∂_x F
(a) F(x) 与它的局部斜率:通量往哪流、在哪儿增长
(b) −∂_x F(x):密度在哪儿变密、哪儿变疏 → 合起来就是"变宽"
(c) 离散记账 ↔ 连续方程
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

V = 0.58
p = lambda x: np.exp(-x ** 2 / (2 * V)) / np.sqrt(2 * np.pi * V)
dp = lambda x: -x / V * p(x)
F = lambda x: 0.69 * x * p(x)
dF = lambda x: 0.69 * (p(x) + x * dp(x))

fig = plt.figure(figsize=(15.2, 5.4), dpi=150)
xs = np.linspace(-2.6, 2.6, 700)

# ---------------- (a) 通量 F 与局部斜率
ax = fig.add_axes([0.05, 0.17, 0.30, 0.67])
ax.plot(xs, F(xs), color=GREEN, lw=2.6, label="通量 $F(x)=f\\cdot p$")
ax.axhline(0, color=GREY, lw=0.8)
for x0 in [-1.5, -0.76, 0.76, 1.5]:
    m = dF(x0)
    xx = np.linspace(x0 - 0.28, x0 + 0.28, 20)
    ax.plot(xx, F(x0) + m * (xx - x0), color=RED, lw=1.6, ls="--")
    ax.plot([x0], [F(x0)], "o", ms=6, color=DARK)
ax.annotate("$x>0$:通量为正\n(往右流)", xy=(1.2, F(1.2)), xytext=(0.35, 0.16),
            fontsize=9.2, color=GREEN, arrowprops=dict(arrowstyle="-|>", lw=1.2, color=GREEN))
ax.annotate("$x<0$:通量为负\n(往左流)", xy=(-1.2, F(-1.2)), xytext=(-2.5, -0.16),
            fontsize=9.2, color=GREEN, arrowprops=dict(arrowstyle="-|>", lw=1.2, color=GREEN))
ax.text(-0.72, 0.135, "虚线 = 局部斜率 $\\partial_x F$", fontsize=9.2, color=RED)
ax.set_xlim(-2.6, 2.6); ax.set_ylim(-0.235, 0.235)
ax.set_xlabel("$x$", fontsize=10.5); ax.set_ylabel("通量 $F$", fontsize=10.5)
ax.set_title("(a) $F$ 是流量:向两边流出去\n$\\partial_x F$ 是它沿路增长多少", fontsize=11.3)
ax.legend(fontsize=9, frameon=False, loc="upper left")
ax.tick_params(labelsize=9)

# ---------------- (b) −∂_x F 与密度变化
ax2 = fig.add_axes([0.40, 0.17, 0.315, 0.67])
ax2.plot(xs, p(xs) / p(0), color=DARK, lw=2.0, alpha=0.55, label="密度 $p(x)$(归一化)")
ax2.plot(xs, -dF(xs) / abs(dF(0)), color=BLUE, lw=2.6, label="$-\\partial_x F$ = 密度的变化率")
ax2.axhline(0, color=GREY, lw=0.8)
ax2.axvline(0.76, color=ORANGE, ls=":", lw=1.4); ax2.axvline(-0.76, color=ORANGE, ls=":", lw=1.4)
ax2.axvspan(-0.76, 0.76, color=RED, alpha=0.07)
ax2.text(0.0, -1.05, "中心区 $|x|<0.76$\n密度【变疏】", fontsize=9.2, color=RED, ha="center")
ax2.text(-2.15, 0.55, "尾巴 $|x|>0.76$\n密度【变密】", fontsize=9.2, color=GREEN, ha="left")
ax2.set_xlim(-2.6, 2.6); ax2.set_ylim(-1.25, 1.25)
ax2.set_xlabel("$x$", fontsize=10.5); ax2.set_ylabel("相对大小", fontsize=10.5)
ax2.set_title("(b) 中心变疏 + 尾巴变密 = 分布【变宽】\n(σ=0.7 时加噪本就该变宽)✓", fontsize=11.3)
ax2.legend(fontsize=8.8, frameon=False, loc="upper left")
ax2.tick_params(labelsize=9)

# ---------------- (c) 记账
ax3 = fig.add_axes([0.745, 0.17, 0.235, 0.67]); ax3.axis("off")
ax3.text(0.5, 0.99, "(c) 两句话", ha="center", va="top", fontsize=11.8, color=DARK)
ax3.text(0.02, 0.88,
         "离散(你已经懂的):\n"
         "$\\Delta p_i=-\\dfrac{F_i-F_{i-1}}{\\Delta x}\\Delta\\sigma$\n"
         "\"这个格子的变化 = 左边流进 − 右边流出\"\n\n"
         "连续(同一件事):\n"
         "$\\partial_\\sigma p=-\\partial_x F$\n"
         "\"差\"变成\"导数\",别的都一样。",
         fontsize=9.4, color=DARK, va="top")
tbl = [["$x$", "$F$", "$\\partial_x F$", "$-\\partial_x F$"],
       ["0", "0", "+0.36", "−0.36(变疏)"],
       ["±1.0", "±0.15", "−0.11", "+0.11(变密)"],
       ["±2.0", "±0.04", "−0.026", "+0.026(变密)"]]
tb = ax3.table(cellText=tbl, cellLoc="center", bbox=[0.0, 0.235, 1.0, 0.27])
tb.auto_set_font_size(False); tb.set_fontsize(8.4)
for (r, c), cell in tb.get_celld().items():
    if r == 0:
        cell.set_facecolor("#eef2f7")
ax3.text(0.0, 0.20,
         "读法:某处密度涨得有多快\n= 左边流进来的 − 右边流出去的\n"
         "= −(净流出)= $-\\partial_x F$",
         fontsize=9.4, color=GREEN, va="top", bbox=dict(fc="#eef7f1", ec="#a9d5bb"))

fig.suptitle("$\\partial_\\sigma p=-\\partial_x F$ 就是「收支平衡」:本地存量的变化 = 流入 − 流出", fontsize=12.6, y=0.99)
fig.savefig(RF + "fg_continuity_meaning.png", bbox_inches="tight")
print("数值核对(σ=0.7, V=0.58):")
for x0 in [0.0, 1.0, 2.0]:
    print(f"  x={x0}: p={p(x0):.4f}  F={F(x0):+.4f}  ∂xF={dF(x0):+.4f}  -∂xF={-dF(x0):+.4f}")
print("  F' 变号位置:", np.round(np.sqrt(1/ (1/V)) if False else np.sqrt(V), 4), "(理论值 |x|=√V=%.4f)" % np.sqrt(V))
print("wrote", RF + "fg_continuity_meaning.png")
