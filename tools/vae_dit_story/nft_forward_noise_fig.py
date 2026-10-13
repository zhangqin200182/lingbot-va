r"""④ 前向加噪 —— 一步步拆开。

一句话:训练时我们【自己造】一个带噪点,而不是用采样时真正走过的轨迹。

  取一张采到的干净图 x₀,再抽一个新的 ε 和一个 t:
      x_t = α_t x₀ + σ_t ε          ← 【我们把点造出来】(不是从轨迹里取的)
      v   = α̇_t x₀ + σ̇_t ε          ← 目标速度 = 这条直线的【时间导数】(闭式,精确)
  ⇒ 网络输入 (x_t, t, c),输出 v_θ,目标 v

与前面 σ 记法的对照:t 是"噪声水平"(t=0 干净,t=1 纯噪声),即 α_t=1−t, σ_t=t
      v = α̇ x₀ + σ̇ ε = (−1)x₀ + (1)ε = ε − x₀      ← 和 Flow-GRPO 的 v 完全一样!

数值例:x₀=1.0, ε=−0.5, t=0.3 → x_t = 0.55, v = −1.5
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

RED, BLUE, GREEN, DARK, GREY, ORANGE, PURPLE = ("#c0392b", "#2471a3", "#1e8449",
                                                "#2c3e50", "#95a5a6", "#b9770e", "#7d3c98")
RF = "/Users/kevin/code/lingbot-va/.claude/scratch/rf/"

fig = plt.figure(figsize=(15.2, 5.8), dpi=150)

# ---------------- (a) 前向直线 + t 网格
ax = fig.add_axes([0.05, 0.15, 0.28, 0.70])
x0, eps = 1.0, -0.5
ts = np.linspace(0, 1, 200)
ax.plot(ts, (1 - ts) * x0 + ts * eps, color=DARK, lw=2.4, label="这条直线:$x_t=\\alpha_t x_0+\\sigma_t\\epsilon$")
grid = np.linspace(0, 1, 11)
for t in grid:
    xt = (1 - t) * x0 + t * eps
    ax.plot([t], [xt], "o", ms=6.5, color=BLUE, zorder=6)
ax.plot([0], [x0], "o", ms=11, color=GREEN, zorder=7)
ax.text(0.02, x0 + 0.12, "干净图 $x_0$(采样得到)", fontsize=9.0, color=GREEN)
ax.plot([1], [eps], "o", ms=11, color=ORANGE, zorder=7)
ax.text(0.72, eps - 0.22, "噪声 $\\epsilon$(【重新】抽的)", fontsize=9.0, color=ORANGE)
t_show = 0.3
ax.plot([t_show, t_show], [-1.0, (1 - t_show) * x0 + t_show * eps], color=RED, lw=1.6, ls=":")
ax.plot([t_show], [(1 - t_show) * x0 + t_show * eps], "o", ms=11, color=RED, zorder=8)
ax.text(t_show + 0.03, (1 - t_show) * x0 + t_show * eps + 0.12,
        "训练样本:现造的 $(x_t,v)$\n$t$ 从采样用的网格里取", fontsize=8.8, color=RED)
ax.set_xlabel("噪声水平 $t$(0=干净,1=纯噪声)", fontsize=10.4)
ax.set_ylabel("$x$", fontsize=10.4)
ax.set_title("(a) ④ 前向加噪 = 【自己造】带噪点", fontsize=11.3)
ax.legend(fontsize=8.2, frameon=False, loc="lower left")
ax.tick_params(labelsize=9)
ax.set_ylim(-1.15, 1.35)

# ---------------- (b) 目标 v 从哪来
ax2 = fig.add_axes([0.38, 0.15, 0.30, 0.70]); ax2.axis("off")
ax2.text(0.5, 1.00, "(b) 目标 $v$ 为什么是闭式的", ha="center", va="top", fontsize=11.6, color=DARK)
ax2.text(0.0, 0.91,
         "因为这条直线是【我们自己画的】,\n"
         "对 $t$ 求导就直接得到速度:\n\n"
         "　$x_t=\\alpha_t x_0+\\sigma_t\\epsilon$\n"
         "　$v=\\dfrac{dx_t}{dt}=\\dot\\alpha_t x_0+\\dot\\sigma_t\\epsilon$\n\n"
         "不需要求期望 —— 因为 $x_0$ 和 $\\epsilon$ 都是【已知】的\n"
         "(生成时不知道是哪个 $x_0$,才需要取条件期望)",
         fontsize=9.0, color=DARK, va="top")
ax2.text(0.0, 0.50,
         "记号对照(取 $\\alpha_t=1-t,\\ \\sigma_t=t$):\n"
         "　$v=\\dot\\alpha x_0+\\dot\\sigma\\epsilon=(-1)x_0+(1)\\epsilon=\\epsilon-x_0$\n"
         "　← 和 Flow-GRPO 里的 $v=\\epsilon-x_0$ 【完全一样】",
         fontsize=9.0, color=GREEN, va="top", bbox=dict(fc="#eef7f1", ec=GREEN))
ax2.text(0.0, 0.31,
         "数值例:$x_0=1.0,\\ \\epsilon=-0.5,\\ t=0.3$\n"
         "　$x_t=0.7\\times1.0+0.3\\times(-0.5)=0.55$\n"
         "　$v=(-1)(1.0)+(1)(-0.5)=-1.5$\n"
         "　核对:$\\epsilon-x_0=-1.5$ ✓",
         fontsize=9.0, color=DARK, va="top", bbox=dict(fc="#fffbe6", ec="#e0c060"))

# ---------------- (c) 三个"为什么"
ax3 = fig.add_axes([0.72, 0.15, 0.27, 0.70]); ax3.axis("off")
ax3.text(0.5, 1.00, "(c) 三个\"为什么\"", ha="center", va="top", fontsize=11.6, color=DARK)
ax3.text(0.0, 0.92,
         "① 为什么要加噪?\n"
         "　因为网络学的是 $v_\\theta(x_t,t)$ ——\n"
         "　它【在所有噪声水平上】都要会回答\n"
         "　\"这里的速度是多少\"。\n"
         "　干净图本身没有\"速度\"可言。\n\n"
         "② 为什么不需要轨迹?\n"
         "　$x_t$ 是【现造】的,不是从轨迹里取的\n"
         "　⇒ 只要 $x_0$ 和奖励,中间态全丢\n"
         "　⇒ 这是省显存的关键\n\n"
         "③ 为什么用采样时的 $t$ 网格?\n"
         "　让训练落在【solver 真正走过】的\n"
         "　噪声水平上;再加时间权重 $w(t)$\n"
         "　(消融:大 $t$ 权重高才稳,\n"
         "　 用 $w(t)=1-t$ 会崩)",
         fontsize=8.6, color=DARK, va="top")

fig.suptitle("④ 前向加噪:$x_t=\\alpha_t x_0+\\sigma_t\\epsilon$ 是【现造】的训练点,$v=\\dot\\alpha_t x_0+\\dot\\sigma_t\\epsilon$ 是它闭式的速度目标",
             fontsize=11.6, y=0.99)
fig.savefig(RF + "fg_forward_noising.png", bbox_inches="tight")
print("x0=1.0, eps=-0.5, t=0.3 → x_t=%.4f, v=%.4f, eps-x0=%.4f" % (
    (1 - 0.3) * 1.0 + 0.3 * (-0.5), (-1) * 1.0 + 1 * (-0.5), -0.5 - 1.0))
print("wrote", RF + "fg_forward_noising.png")
