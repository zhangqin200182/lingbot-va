r"""为什么"奖励"落在每步上?—— 其实奖励只有终点一个,每步只是"分摊"。

  轨迹的概率 = 各步条件概率的【乘积】:  π_θ(traj) = Π_j π_θ(x_{j+1}|x_j)
  取对数就变成【求和】:              log π_θ(traj) = Σ_j log π_θ(x_{j+1}|x_j)
  求导:                              ∇log π_θ(traj) = Σ_j ∇log π_θ(x_{j+1}|x_j)

⇒ 不是"我们选择奖励每步",而是"轨迹概率的对数天然是各步之和";
   θ 也只能通过每步的均值起作用,所以梯度必然落在每步上。

(a) 轨迹概率 = 各步相乘
(b) 两种等价写法:轨迹级比值 = 每步比值的乘积;以及逐步裁剪为什么更合理
(c) 真正的"逐步奖励"需要什么(per-step value / 中间奖励)——生成任务里没有
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

fig = plt.figure(figsize=(15.2, 5.6), dpi=150)

# ---------------- (a) 轨迹概率 = 各步相乘
ax = fig.add_axes([0.04, 0.12, 0.34, 0.76]); ax.axis("off")
ax.text(0.5, 1.00, "(a) 轨迹的概率 = 各步条件概率【相乘】", ha="center", va="top", fontsize=11.8, color=DARK)
xs = np.linspace(0.05, 0.95, 6)
ys = np.array([0.75, 0.86, 0.78, 0.62, 0.48, 0.34])
for i in range(len(xs) - 1):
    ax.add_patch(FancyArrowPatch((xs[i] + 0.015, ys[i]), (xs[i + 1] - 0.015, ys[i + 1]),
                                 arrowstyle="-|>", mutation_scale=12, lw=2.0, color=DARK))
    ax.text((xs[i] + xs[i + 1]) / 2, (ys[i] + ys[i + 1]) / 2 + 0.055,
            f"$\\pi_{{{i+1}}}$", fontsize=10.5, color=BLUE, ha="center")
ax.plot(xs, ys, "o", ms=8, color=DARK, zorder=5)
ax.text(0.05, 0.13, "$\\pi_\\theta(\\mathrm{traj})=\\prod_j \\pi_\\theta(x_{j+1}|x_j)$\n"
                    "$\\log\\pi_\\theta=\\sum_j \\log\\pi_\\theta(x_{j+1}|x_j)$\n"
                    "$\\nabla\\log\\pi_\\theta=\\sum_j \\nabla\\log\\pi_\\theta(x_{j+1}|x_j)$",
        fontsize=10.5, color=DARK, va="top", bbox=dict(fc="#fffbe6", ec="#e0c060"))
ax.text(0.5, 0.965, "奖励只有一个(终点);但这个【求和】是概率的数学结构,不是我们的选择",
        fontsize=9.2, color=GREEN, ha="center", va="top")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)

# ---------------- (b) 两种等价写法
ax2 = fig.add_axes([0.42, 0.16, 0.31, 0.68])
T = 19; eps = 1e-3
steps = np.arange(1, T + 1)
ax2.plot(steps, (1 + eps) ** steps, "o-", color=BLUE, lw=2.2, ms=4,
         label="逐步裁剪:$(1+\\varepsilon)^j$ 上界")
ax2.axhline(1 + eps, color=GREY, ls="--", lw=1.4)
ax2.text(9, 1 + eps + 0.0015, "若只裁剪轨迹级:整条只能动 0.1%", fontsize=8.8, color=GREY)
ax2.axhline((1 + eps) ** T, color=RED, ls=":", lw=1.6)
ax2.text(4.0, 1.0205, f"逐步裁剪:整条上界 $=1.001^{{19}}={((1+eps)**T):.4f}$", fontsize=8.8, color=RED, ha="left")
ax2.set_xlabel("去噪步数 $j$", fontsize=10.5); ax2.set_ylabel("轨迹概率比的上界", fontsize=10.5)
ax2.set_title("(b) 两种写法在数学上等价(轨迹级比值 = 每步比值之积)", fontsize=11.0)
ax2.legend(fontsize=8.8, frameon=False, loc="upper left")
ax2.set_ylim(0.999, 1.023)
ax2.tick_params(labelsize=9)
ax2.text(9.5, 1.0075, "逐步裁剪给出 1.9% 的轨迹级自由度,\n比「只裁轨迹级 0.1%」合理得多",
         fontsize=8.8, color=GREEN, ha="center", bbox=dict(fc="#eef7f1", ec="#a9d5bb"))

# ---------------- (c) 真正的逐步奖励需要什么
ax3 = fig.add_axes([0.755, 0.16, 0.24, 0.68]); ax3.axis("off")
ax3.text(0.5, 0.99, "(c) 那\"真正的逐步奖励\"呢?", ha="center", va="top", fontsize=11.4, color=DARK)
rows = [["要逐 step 定分,需要", "生成任务里"],
        ["每步的中间奖励", "没有\n(半步噪声没法打分)"],
        ["value 网络\n(给中间状态估值)", "GRPO 就是要\n避免它"],
        ["终点的分数", "有\n(检测器/OCR/偏好)"]]
tb = ax3.table(cellText=rows, cellLoc="center", bbox=[0.0, 0.56, 1.0, 0.40])
tb.auto_set_font_size(False); tb.set_fontsize(8.0)
for (r, c), cell in tb.get_celld().items():
    if r == 0:
        cell.set_facecolor("#eef2f7")
    if r == 3:
        cell.set_facecolor("#eef7f1")
    if r in (1, 2):
        cell.set_facecolor("#fdecea")
ax3.text(0.0, 0.50,
         "所以现在的做法:\n"
         "$A$ 是【整条轨迹】的一个数,\n"
         "广播到所有步(代码里 .repeat(1,T))\n"
         "⇒ 推的是【整条轨迹的概率】\n"
         "　 而不是单独某一步的概率。\n\n"
         "代价:没有 per-step 信用分配\n"
         "(无关方向的推挤也在里面)。",
         fontsize=8.6, color=DARK, va="top",
         bbox=dict(fc="#fffbe6", ec="#e0c060"))
ax3.text(0.0, 0.03,
         "要真的做逐步信用分配,\n得引入 value 或中间监督\n—— 那是另一条路线。",
         fontsize=8.6, color=GREEN, va="bottom", bbox=dict(fc="#eef7f1", ec="#a9d5bb"))

fig.suptitle("奖励只有终点一个;每步被\"分摊\"是因为【轨迹概率的对数 = 各步之和】,不是我们选择奖励每步",
             fontsize=12.0, y=0.99)
fig.savefig(RF + "fg_reward_terminal.png", bbox_inches="tight")
print(f"逐步裁剪: 整条上界 (1+1e-3)^19 = {((1+eps)**T):.6f} ({( (1+eps)**T-1)*100:.2f}%)")
print(f"只裁轨迹级 1e-3 ⇒ 每步 (1+1e-3)^(1/19) = {((1+eps)**(1/T)):.8f} ({( (1+eps)**(1/T)-1)*100:.5f}%)")
print("wrote", RF + "fg_reward_terminal.png")
