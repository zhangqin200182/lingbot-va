r"""澄清:"x 项"是啥?—— 它是【补偿项展开后的一半】。

  ∇log p_σ(x) = -(x + (1-σ)v_θ)/σ        ← 确实是用 v_θ 算出来的(不需要第二个网络)
  但代入后它自己【裂成两项】:
       -(g²/2)∇log p = (g²/(2σ))·x  +  (g²(1-σ)/(2σ))·v_θ
                        └ 含 x 的一半 ┘   └ 含 v_θ 的一半 ┘

  漂移 f = v_θ + 补偿 = (g²/(2σ))x + [1 + g²(1-σ)/(2σ)]·v_θ
                         └ 纯补偿 ┘    └ v_θ 自己 + 补偿的另一半 ┘

  ⇒ "沿 ∇log p(往高密度收)" 和 "写成 x 的系数" 不矛盾:
     ∇log p 本身就同时含 x 与 v_θ
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
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

RED, BLUE, GREEN, DARK, GREY, ORANGE, PURPLE = ("#c0392b", "#2471a3", "#1e8449",
                                                "#2c3e50", "#95a5a6", "#b9770e", "#7d3c98")
RF = "/Users/kevin/code/lingbot-va/.claude/scratch/rf/"


def box(ax, x, y, w, h, text, fc="w", ec="#c8d4e0", fs=8.6, tc=DARK, lw=1.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc)


fig = plt.figure(figsize=(15.2, 5.6), dpi=150)

# ---------------- (a) 展开树
ax = fig.add_axes([0.03, 0.08, 0.34, 0.82]); ax.axis("off")
ax.text(0.5, 1.00, "(a) 补偿项展开成两半", ha="center", va="top", fontsize=12, color=DARK)
box(ax, 0.10, 0.84, 0.80, 0.10, "补偿 $=-\\frac{g^2}{2}\\nabla\\log p_\\sigma$", "#fffbe6", "#e0c060", fs=9.6)
ax.add_patch(FancyArrowPatch((0.50, 0.835), (0.28, 0.72), arrowstyle="-|>", mutation_scale=11, lw=1.6, color=DARK))
ax.add_patch(FancyArrowPatch((0.50, 0.835), (0.72, 0.72), arrowstyle="-|>", mutation_scale=11, lw=1.6, color=DARK))
box(ax, 0.02, 0.60, 0.45, 0.115, "$+\\frac{g^2}{2\\sigma}\\,x$\n(我表格里的\"$x$ 项\")",
    "#fdf3e7", ORANGE, fs=9.0)
box(ax, 0.53, 0.60, 0.45, 0.115, "$+\\frac{g^2(1-\\sigma)}{2\\sigma}\\,v_\\theta$\n(并入 $v_\\theta$ 的系数)",
    "#eaf2fa", BLUE, fs=9.0)
ax.text(0.5, 0.53,
        "恒等式:$\\nabla\\log p_\\sigma=-\\dfrac{x+(1-\\sigma)v_\\theta}{\\sigma}$\n"
        "⇒ 它本来就【同时含 $x$ 与 $v_\\theta$】\n"
        "所以\"沿 $\\nabla\\log p$ 收\"与\"写成 $x$ 的系数\"是一回事",
        fontsize=8.8, color=DARK, ha="center", va="top",
        bbox=dict(fc="#f7f9fb", ec="#c8d4e0"))
box(ax, 0.05, 0.26, 0.90, 0.14,
    "$f=v_\\theta+$补偿$=\\underbrace{\\frac{g^2}{2\\sigma}x}_{\\rm 纯补偿}"
    "+\\underbrace{\\Big[1+\\frac{g^2(1-\\sigma)}{2\\sigma}\\Big]v_\\theta}_{v_\\theta\\rm 自己\\,+\\,补偿另一半}$"
    .replace("\\underbrace", "").replace("\\Big[", "[").replace("\\Big]", "]")
    .replace("_{\\rm 纯补偿}", "").replace("_{v_\\theta\\rm 自己\\,+\\,补偿另一半}", ""),
    "#eef7f1", GREEN, fs=9.2)
ax.text(0.5, 0.14, "系数表里的两个数,就是这个式子的两个系数", fontsize=9.0, color=GREEN, ha="center")
ax.set_xlim(0, 1); ax.set_ylim(0.05, 1.02)

# ---------------- (b) 数值核对
ax2 = fig.add_axes([0.40, 0.08, 0.29, 0.82]); ax2.axis("off")
ax2.text(0.5, 1.00, "(b) 数值核对", ha="center", va="top", fontsize=11.8, color=DARK)
ax2.text(0.0, 0.90,
         "$\\sigma=0.7,\\ nl=0.5,\\ g^2=0.5833$\n"
         "$x=1.0,\\ v_\\theta=0.5$\n\n"
         "$\\nabla\\log p=-\\dfrac{1.0+0.3\\times0.5}{0.7}=-1.6429$\n\n"
         "补偿 $=-\\dfrac{g^2}{2}\\nabla\\log p=+0.4792$\n"
         "　含 $x$ 的一半 $=\\dfrac{g^2}{2\\sigma}x=+0.4167$\n"
         "　含 $v_\\theta$ 的一半 $=\\dfrac{g^2(1-\\sigma)}{2\\sigma}v_\\theta=+0.0625$\n"
         "　相加 $=+0.4792$ ✓ 与补偿一致",
         fontsize=9.0, color=DARK, va="top", bbox=dict(fc="#f7f9fb", ec="#c8d4e0"))
ax2.text(0.0, 0.36,
         "漂移两种写法都要给同一个数:\n\n"
         "$v_\\theta+$补偿 $=0.5+0.4792=+0.9792$\n"
         "$\\dfrac{g^2}{2\\sigma}x+[1+\\dfrac{g^2(1-\\sigma)}{2\\sigma}]v_\\theta$\n"
         "$\\quad=0.4167+1.125\\times0.5=+0.9792$ ✓",
         fontsize=9.0, color=GREEN, va="top", bbox=dict(fc="#eef7f1", ec=GREEN))

# ---------------- (c) 两个系数 vs σ
ax3 = fig.add_axes([0.735, 0.17, 0.255, 0.66])
sg = np.linspace(0.05, 0.95, 300); nl = 0.5; T = 19
c_x = sg * nl ** 2 * (0.5 * sg / (1 - sg)) / sg  # 占位
g2 = sg / (1 - sg) * nl ** 2
cx = g2 / (2 * sg)
cv = np.ones_like(sg) * (1 + nl ** 2 / 2)
ax3.plot(sg, cx, color=ORANGE, lw=2.6, label="$x$ 项系数 $=g^2/(2\\sigma)$")
ax3.plot(sg, cv, color=BLUE, lw=2.6, label="$v_\\theta$ 项(除以 $d\\sigma$)\n$=1+nl^2/2$,恒定")
ax3.set_xlabel("$\\sigma$", fontsize=10.5); ax3.set_ylabel("系数", fontsize=10.5)
ax3.set_title("(c) 靠噪声端(σ→1)\n$x$ 项(补偿)迅速变大", fontsize=11.2)
ax3.legend(fontsize=8.2, frameon=False, loc="upper left")
ax3.tick_params(labelsize=9)
ax3.set_ylim(0, 6)

fig.suptitle("\"$x$ 项\"是什么:它是【补偿项展开后的一半】—— $\\nabla\\log p$ 同时含 $x$ 与 $v_\\theta$,代入后自然裂成两项",
             fontsize=11.8, y=0.99)
fig.savefig(RF + "fg_x_term_meaning.png", bbox_inches="tight")
print("wrote", RF + "fg_x_term_meaning.png")
