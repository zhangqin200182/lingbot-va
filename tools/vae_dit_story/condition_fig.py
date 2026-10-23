r"""条件(c)怎么影响去噪方向?—— 它不改变起点,它改变【场】。

  同一个起点 ε,在两个条件下被送到完全不同的终点:
      无条件 / c=+1:数据在 +45° 直线 ⇒ ODE 把 x_t 压到那条线上
      c=-1        :数据在 −45° 直线 ⇒ 压到另一条线
  ⇒ 起点分布不变(N(0,I)),变的是映射 F_c: ε ↦ x₀(每 c 一个微分同胚)

  条件何时主导? 后验 p(x₀|x_t,c) ∝ p_data(x₀|c) · N(x_t;(1−t)x₀, t²I)
      把高斯看作 x₀ 的函数,其宽度 = t/(1−t)
      t 大 ⇒ 平坦 ⇒ 后验≈先验 ⇒ 【条件主导结构】
      t 小 ⇒ 尖锐 ⇒ 后验≈点质量 ⇒ 【采样路径主导细节】
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


def u(c):
    return np.array([1.0, c]) / np.sqrt(2)


def Ev(x, t, c):
    uc = u(c)
    V = (1 - t) ** 2 + t ** 2
    s = (1 - t) * np.dot(x, uc) / V
    return (x - s * uc) / t


eps = np.array([0.9, -0.5])
tracks = {}
for cc in [+1, -1]:
    x = eps.copy()
    SIG = np.linspace(1, 0, 600)
    tr = [x.copy()]
    for i in range(599):
        t, sn = SIG[i], SIG[i + 1]
        x = x + Ev(x, t, cc) * (sn - t)
        tr.append(x.copy())
    tracks[cc] = np.array(tr)

fig = plt.figure(figsize=(15.2, 5.7), dpi=150)

# ---------------- (a) 同一个 ε,两条轨迹
ax = fig.add_axes([0.05, 0.13, 0.30, 0.72])
for c, col, lab in [(+1, BLUE, "$c=+1$:数据在 +45° 线"), (-1, ORANGE, "$c=-1$:数据在 −45° 线")]:
    tr = tracks[c]
    ax.plot(tr[:, 0], tr[:, 1], color=col, lw=2.4, label=lab)
    ax.plot(tr[-1, 0], tr[-1, 1], "o", ms=10, color=col, zorder=6)
    ax.annotate(f"终点 ({tr[-1,0]:+.2f},{tr[-1,1]:+.2f})", xy=(tr[-1, 0], tr[-1, 1]),
                xytext=(tr[-1, 0] + 0.06, tr[-1, 1] - 0.16), fontsize=8.4, color=col)
s = np.linspace(-1.4, 1.4, 10)
ax.plot(s / np.sqrt(2), s / np.sqrt(2), ":", color=BLUE, lw=1.2)
ax.plot(s / np.sqrt(2), -s / np.sqrt(2), ":", color=ORANGE, lw=1.2)
ax.plot([eps[0]], [eps[1]], "o", ms=13, color=DARK, zorder=7)
ax.text(eps[0] + 0.06, eps[1] + 0.06, "同一个起点 $\\epsilon$\n(同一次采样)", fontsize=9.2, color=DARK)
ax.plot([0], [0], "+", ms=10, color=GREY)
ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.1)
ax.set_xlabel("$x^{(1)}$", fontsize=10.4); ax.set_ylabel("$x^{(2)}$", fontsize=10.4)
ax.set_title("(a) 起点相同,条件不同 ⇒ 终点不同\n(变的是【场】,不是起点)", fontsize=11.2)
ax.legend(fontsize=8.6, frameon=False, loc="lower left")
ax.tick_params(labelsize=9)
ax.grid(alpha=0.15)

# ---------------- (b) 机制:条件只作为网络的额外输入
ax2 = fig.add_axes([0.39, 0.10, 0.28, 0.78]); ax2.axis("off")
ax2.text(0.5, 1.00, "(b) 条件怎么进网络", ha="center", va="top", fontsize=11.6, color=DARK)
ax2.text(0.0, 0.92,
         "无条件:$v_\\theta(x_t,t)$\n"
         "有条件:$v_\\theta(x_t,t,c)$\n\n"
         "DiT 里 $c$ 的两条通路:\n"
         "· 文本 token 走 cross-attention\n"
         "　 (作为 K/V)\n"
         "· 条件/时间嵌入走 AdaLN\n"
         "　 (每块的 scale/shift/gate)\n\n"
         "⇒ 同一个 $(x_t,t)$,不同 $c$\n"
         "　 网络输出【不同速度】\n"
         "⇒ 采样算法【完全不变】",
         fontsize=9.0, color=DARK, va="top")
ax2.text(0.0, 0.36,
         "于是每个 $c$ 有自己的场,把\n"
         "$N(0,I)$ 推成 $p_{\\rm data}(\\cdot|c)$\n\n"
         "$F_c:\\ \\epsilon\\mapsto x_0$ 是【每 c 一个】\n"
         "不同的映射(微分同胚)",
         fontsize=9.0, color=GREEN, va="top", bbox=dict(fc="#eef7f1", ec=GREEN))
ax2.text(0.0, 0.10,
         "所以\"遵从性\"不是\"约束了起点\",\n"
         "而是\"换了一个把起点送到哪的映射\"。",
         fontsize=9.0, color=RED, va="top", bbox=dict(fc="#fdecea", ec="#e6a9a0"))

# ---------------- (c) 条件何时主导
ax3 = fig.add_axes([0.70, 0.13, 0.28, 0.72])
ts = np.array([0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.97])
width = ts / (1 - ts)
ax3.semilogy(ts, width, "o-", color=PURPLE, lw=2.4, ms=6)
ax3.axhline(1.0, color=GREY, ls="--", lw=1.4)
ax3.axhspan(0.01, 1.0, color=GREEN, alpha=0.10)
ax3.axhspan(1.0, 60, color=BLUE, alpha=0.10)
ax3.text(0.03, 1.6, "后验≈先验\n【条件主导结构】", fontsize=8.8, color=BLUE, va="bottom")
ax3.text(0.03, 0.35, "后验≈点质量\n【采样路径主导细节】", fontsize=8.8, color=GREEN, va="top")
ax3.set_xlabel("噪声水平 $t$", fontsize=10.4)
ax3.set_ylabel("后验高斯的宽度 $t/(1-t)$", fontsize=10.0)
ax3.set_title("(c) $p(x_0|x_t,c)\\propto p_{\\rm data}(x_0|c)\\cdot N(x_t;(1-t)x_0,t^2I)$\n"
              "把高斯看成 $x_0$ 的函数,宽度 $=t/(1-t)$", fontsize=10.2)
ax3.tick_params(labelsize=9)
ax3.set_xlim(0, 1)

fig.suptitle("语言条件怎么影响去噪方向:起点仍是 $N(0,I)$,但【场】变成 $v_\\theta(x_t,t,c)$ —— 同一个 $\\epsilon$ 被送到不同的 $x_0$",
             fontsize=11.2, y=0.99)
fig.savefig(RF + "fg_condition_mechanism.png", bbox_inches="tight")
print("wrote", RF + "fg_condition_mechanism.png")
