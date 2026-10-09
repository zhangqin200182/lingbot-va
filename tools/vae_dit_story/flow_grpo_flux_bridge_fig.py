r"""从"粒子速度"到"密度方程":通量是怎么冒出来的。

视角转换:同一件事的两种记法
  粒子视角(你熟悉的):  dx = f(x,σ) dσ        "这个粒子去哪了"
  密度视角(现在要用的):  ∂_σ p = −∂_x(F)       "这里还剩多少"

(a) 通量 = 密度 × 速度:F = f·p(画出 p、速度箭头 f、以及它们的乘积 F)
(b) 最土的记账:格子里的粒子数怎么变 = 左边流进 − 右边流出 → ∂_σ p = −∂_x F
(c) 两种通量:漂移通量 f·p 与扩散通量 −(g²/2)·p'
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
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle

RED, BLUE, GREEN, DARK, GREY, ORANGE = "#c0392b", "#2471a3", "#1e8449", "#2c3e50", "#95a5a6", "#b9770e"
RF = "/Users/kevin/code/lingbot-va/.claude/scratch/rf/"

fig = plt.figure(figsize=(15.2, 5.6), dpi=150)

# ---------------- (a) 通量 = 密度 × 速度
ax = fig.add_axes([0.05, 0.16, 0.30, 0.68])
xs = np.linspace(-3.0, 3.0, 600)
P = lambda x: np.exp(-x ** 2 / (2 * 0.58)) / np.sqrt(2 * np.pi * 0.58)   # p_σ (σ=0.3 的高斯)
F = lambda x: -0.69 * x * P(x)                                            # F = f·p
ax.plot(xs, P(xs), color=DARK, lw=2.6, label="密度 $p(x)$")
ax.plot(xs, F(xs), color=GREEN, lw=2.4, label="通量 $F=f\\cdot p$")
ax.axhline(0, color=GREY, lw=0.8)
for x0 in [-2.0, -1.2, -0.6, 0.6, 1.2, 2.0]:
    L = -np.sign(x0) * 0.28
    ax.add_patch(FancyArrowPatch((x0, P(x0) + 0.055), (x0 + L, P(x0) + 0.055),
                                 arrowstyle="-|>", mutation_scale=11, lw=1.8, color=BLUE))
    ax.text(x0, P(x0) + 0.085, "$f$", fontsize=8.4, color=BLUE, ha="center")
ax.text(0.0, -0.115, "通量 $F(x)=f(x)\\cdot p(x)$:\n密度 × 速度 = 单位时间穿过该点的粒子数",
        fontsize=9.6, color=GREEN, ha="center",
        bbox=dict(fc="#eef7f1", ec="#a9d5bb"))
ax.set_xlim(-3.0, 3.0); ax.set_ylim(-0.30, 0.62)
ax.set_xlabel("$x$", fontsize=10.5); ax.set_ylabel("密度 / 通量", fontsize=10.5)
ax.set_title("(a) 通量就是这个:$F=f\\times p$\n密度大、速度大 → 流量就大", fontsize=11.5)
ax.legend(fontsize=9, frameon=False, loc="upper right")
ax.tick_params(labelsize=9)

# ---------------- (b) 数格子记账
ax2 = fig.add_axes([0.41, 0.16, 0.29, 0.68]); ax2.axis("off")
ax2.text(0.5, 0.99, "(b) 最土的记账:数格子", ha="center", va="top", fontsize=11.8, color=DARK)
# 三个格子
for k, (x0, n) in enumerate(zip([0.05, 0.38, 0.71], [12, 30, 11])):
    ax2.add_patch(Rectangle((x0, 0.63), 0.26, 0.26, fc="#eaf2fa", ec="#8aa8c8", lw=1.4))
    ax2.text(x0 + 0.13, 0.83, f"$p_{k}$", fontsize=11, color=DARK, ha="center")
    ax2.text(x0 + 0.13, 0.715, f"{n} 个粒子", fontsize=9.2, color=DARK, ha="center")
for x0, lab, dy in [(0.31, "流入 $F_{i-1}$", 0), (0.665, "流出 $F_i$", 0)]:
    ax2.add_patch(FancyArrowPatch((x0 - 0.05, 0.76), (x0 + 0.05, 0.76), arrowstyle="-|>",
                                  mutation_scale=15, lw=2.4, color=GREEN))
    ax2.text(x0, 0.76 + dy + 0.055, lab, fontsize=9.4, color=GREEN, ha="center")
ax2.text(0.02, 0.53,
         "中间格子的净变化 = 左边流进 − 右边流出:\n"
         "$\\Delta n_i = F_{i-1}\\,\\Delta\\sigma-F_i\\,\\Delta\\sigma$\n"
         "两边除以格宽 $\\Delta x$($p_i=n_i/\\Delta x$):\n"
         "$\\partial_\\sigma p=-\\dfrac{F_i-F_{i-1}}{\\Delta x}$\n"
         "令 $\\Delta x\\to0$:$\\;\\partial_\\sigma p=-\\partial_x F$",
         fontsize=9.8, color=DARK, va="top")
ax2.text(0.02, 0.05,
         "所以那个方程只是【记账】:\n密度的变化 = −(净流出),没有新物理。",
         fontsize=9.6, color=GREEN, va="bottom", bbox=dict(fc="#eef7f1", ec="#a9d5bb"))

# ---------------- (c) 两种通量 + 视角对照
ax3 = fig.add_axes([0.725, 0.16, 0.255, 0.68]); ax3.axis("off")
ax3.text(0.5, 0.99, "(c) 两种通量,两个视角", ha="center", va="top", fontsize=11.5, color=DARK)
rows = [["粒子视角(你熟的)", "密度视角(这里的)"],
        ["位置 $x$", "密度 $p(x)$"],
        ["速度 $f(x,\\sigma)$", "通量 $F=f\\,p$"],
        ["走一步 $dx=f\\,d\\sigma$", "$\\partial_\\sigma p=-\\partial_x F$"],
        ["随机晃 $g\\,dW$", "扩散通量 $-\\frac{g^2}{2}\\partial_x p$"]]
tb = ax3.table(cellText=rows, cellLoc="center", bbox=[0.0, 0.52, 1.0, 0.36])
tb.auto_set_font_size(False); tb.set_fontsize(8.8)
for (r, c), cell in tb.get_celld().items():
    if r == 0:
        cell.set_facecolor("#eef2f7")
ax3.text(0.0, 0.46,
         "把两种通量加起来:\n"
         "$F_{\\rm total}=f\\,p-\\frac{g^2}{2}\\partial_x p$\n"
         "(前一项:被速度搬走;后一项:扩散)",
         fontsize=9.2, color=DARK, va="top")
ax3.text(0.0, 0.30,
         "第二项就是【扩散】:\n从密的地方往疏的地方流\n(斐克定律),流量正比于\n浓度梯度 $\\partial_x p$。\n\n"
         "代进记账方程就得:\n$\\partial_\\sigma p=-\\partial_x(fp)+\\frac{g^2}{2}p''$",
         fontsize=9.2, color=DARK, va="top")
ax3.text(0.0, 0.03, "于是两次求导都只是「记账」,\n不再是跳跃。",
         fontsize=9.4, color=GREEN, va="bottom", bbox=dict(fc="#eef7f1", ec="#a9d5bb"))

fig.suptitle("补上那个跳跃:从「粒子速度」到「密度方程」—— 中间只隔着一个【通量 = 密度 × 速度】", fontsize=12.4, y=0.99)
fig.savefig(RF + "fg_flux_bridge.png", bbox_inches="tight")
print("wrote", RF + "fg_flux_bridge.png")
