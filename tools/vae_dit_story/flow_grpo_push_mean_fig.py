r"""验证你的理解:某步的漂移+噪声把位置挪了,若轨迹得分高,RL 就让"这次的移动"更可能出现。

精确化三点:
  (a) 被增加的是【这一步的条件转移概率】π_θ(x_{j+1}|x_j),不是"某个位置的密度"
  (b) 推动方向 = 当时那次噪声的方向 ξ_j(θ 只能动均值,均值就往"实际落点−均值"方向挪)
  (c) 决定推不推的不是"这个 step 的得分",而是【整条轨迹的得分】—— 优势被广播到所有 step

梯度:∇L = −A·Σ_j (ξ_j/s_j)·∂m_j/∂θ     ← 每步沿自己的噪声方向推,力度同一个 A

(a) 机制示意
(b) 1 维 toy 真跑:m 从 0 收敛到奖励峰值 1;G 越大越平滑(运气方差 ∝ 1/√G)
(c) 公式 + 副作用(信用分配噪声)
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

RED, BLUE, GREEN, DARK, GREY, ORANGE, PURPLE = ("#c0392b", "#2471a3", "#1e8449",
                                                "#2c3e50", "#95a5a6", "#b9770e", "#7d3c98")
RF = "/Users/kevin/code/lingbot-va/.claude/scratch/rf/"

# ---------------- toy:m 是均值参数,奖励在 x=1 最高
def run(G, seed, iters=400, lr=0.08):
    rng = np.random.default_rng(seed)
    m, hist = 0.0, []
    for t in range(iters):
        xi = rng.normal(size=G)
        x = m + xi
        R = -(x - 1.0) ** 2
        A = (R - R.mean()) / (R.std() + 1e-8)
        m = m + lr * float(np.mean(A * xi))       # θ ← θ + lr·A·ξ
        hist.append(m)
    return np.array(hist)


fig = plt.figure(figsize=(15.2, 5.5), dpi=150)

# ---------------- (a) 机制示意
ax = fig.add_axes([0.05, 0.17, 0.30, 0.66])
xs = np.linspace(-4, 4, 600)
N = lambda x, m: np.exp(-(x - m) ** 2 / 2) / np.sqrt(2 * np.pi)
ax.plot(xs, N(xs, 0.0), color=DARK, lw=2.6, label="这一步的高斯(当前 $\\theta$)")
ax.plot(xs, N(xs, 0.5), color=GREEN, lw=2.2, ls="--", label="推之后:均值移到 $+0.5$")
ax.axvline(1.0, color=GREY, ls=":", lw=1.2)
ax.axvline(0.0, color=BLUE, lw=2.0); ax.axvline(1.0, color=ORANGE, lw=2.0)
ax.annotate("", xy=(1.0, 0.30), xytext=(0.0, 0.30),
            arrowprops=dict(arrowstyle="-|>", lw=2.6, color=ORANGE))
ax.text(0.5, 0.33, "这一步实际落点(被 $s\\xi$ 带过去的)", fontsize=9.0, color=ORANGE, ha="center")
ax.text(-3.9, 0.42, "轨迹得分高 $\\Rightarrow$ 沿 $\\xi$ 方向把均值挪过去\n"
                    "$m\\leftarrow m+lr\\cdot A\\cdot\\xi$,且 A 对【所有 step】相同",
        fontsize=9.2, color=DARK, va="top", bbox=dict(fc="#fffbe6", ec="#e0c060"))
ax.set_xlim(-4, 4); ax.set_ylim(-0.02, 0.50)
ax.set_xlabel("$x$(这一步的位置)", fontsize=10.5); ax.set_ylabel("密度", fontsize=10.5)
ax.set_title("(a) 机制:噪声给出方向,奖励决定要不要\n把均值往那个方向挪", fontsize=11.4)
ax.legend(fontsize=8.8, frameon=False, loc="upper left")
ax.tick_params(labelsize=9)

# ---------------- (b) toy 收敛
ax2 = fig.add_axes([0.42, 0.17, 0.31, 0.66])
cols = [BLUE, RED]
for G, col in zip([4, 16], cols):
    for k in range(3):
        h = run(G, seed=10 * G + k)
        ax2.plot(h, color=col, lw=1.3, alpha=0.85,
                 label=(f"$G={G}$" if k == 0 else None))
ax2.axhline(1.0, color=GREY, ls="--", lw=1.8)
ax2.text(380, 1.03, "奖励峰值 $x=1$", fontsize=9.2, color=GREY, ha="right")
ax2.set_xlabel("迭代次数", fontsize=10.5); ax2.set_ylabel("学到的均值 $m$", fontsize=10.5)
ax2.set_ylim(-0.25, 1.35)
ax2.set_title("(b) 1 维 toy 真跑:同一个奖励,只改组大小\n"
              "两种都收敛,但 $G$ 大更平滑(运气方差 $\\propto1/\\sqrt{G}$)", fontsize=11.0)
ax2.legend(fontsize=9, frameon=False, loc="lower right")
ax2.tick_params(labelsize=9)

# ---------------- (c) 公式与副作用
ax3 = fig.add_axes([0.755, 0.17, 0.24, 0.66]); ax3.axis("off")
ax3.text(0.5, 0.99, "(c) 公式与副作用", ha="center", va="top", fontsize=11.6, color=DARK)
ax3.text(0.0, 0.88,
         "整条轨迹的梯度:\n"
         "$\\nabla_\\theta\\mathcal{L}=-A\\sum_j\\dfrac{\\xi_j}{s_j}\\dfrac{\\partial m_j}{\\partial\\theta}$\n\n"
         "读法:\n"
         "· 每个 step 沿【自己那次的 $\\xi_j$】方向推\n"
         "· 力度由【同一个】$A$ 决定(不是逐步得分)\n"
         "· θ 只能动均值 ⇒ 挪不了噪声本身",
         fontsize=9.0, color=DARK, va="top")
ax3.text(0.0, 0.40,
         "副作用(信用分配噪声):\n"
         "同一条轨迹里【所有】噪声方向都被一起\n"
         "强化了,包括无关甚至有害的那些。\n\n"
         "为什么最终还能学到:\n"
         "真正有用的方向会在很多条轨迹里\n"
         "反复与高奖励相关;随机方向互相抵消。",
         fontsize=8.8, color=RED, va="top", bbox=dict(fc="#fdecea", ec="#e6a9a0"))
ax3.text(0.0, 0.03,
         "这也是为什么\"运气\"不能也不必被抵消\n—— 它是方向信息的来源。",
         fontsize=8.8, color=GREEN, va="bottom", bbox=dict(fc="#eef7f1", ec="#a9d5bb"))

fig.suptitle("你的理解是对的:噪声把位置挪到某处,若轨迹得分高,RL 就让\"这次的移动\"更可能出现", fontsize=12.2, y=0.99)
fig.savefig(RF + "fg_rl_pushes_mean.png", bbox_inches="tight")

print("1 维 toy 收敛结果(m 应趋于奖励峰值 1.0):")
for G in [4, 16]:
    ends = [run(G, seed=10 * G + k)[-1] for k in range(3)]
    last = [run(G, seed=10 * G + k)[-60:].std() for k in range(3)]
    print(f"  G={G:<3}: 末值 {['%.3f' % e for e in ends]}  末段波动 {['%.4f' % s for s in last]}")
print("wrote", RF + "fg_rl_pushes_mean.png")
