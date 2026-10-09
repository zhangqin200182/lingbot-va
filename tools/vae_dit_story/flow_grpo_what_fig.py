r"""GRPO 到底改了什么、能学到什么。

(a) 目的地会被【故意改掉】:训练前 P(+1)=0.5 → 训练后 =0.8(奖励偏向 +1)
(b) 机制链条:同一 prompt 采 G 个 → 打分 → 组内标准化 → 优势 → 好的概率上升
(c) 为什么必须是 SDE:ODE 给不出似然 ⇒ 没有 ratio ⇒ 没有梯度
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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

RED, BLUE, GREEN, DARK, GREY, ORANGE, PURPLE = ("#c0392b", "#2471a3", "#1e8449",
                                                "#2c3e50", "#95a5a6", "#b9770e", "#7d3c98")
RF = "/Users/kevin/code/lingbot-va/.claude/scratch/rf/"

fig = plt.figure(figsize=(15.2, 5.4), dpi=150)

# ---------------- (a) 目的地被改掉
ax = fig.add_axes([0.05, 0.17, 0.29, 0.66])
xr = np.linspace(-2.2, 2.2, 500)
R = 0.5 + 0.45 * np.tanh(1.6 * (xr - 0.0))          # 奖励函数:偏向 +1
ax.plot(xr, R, color=ORANGE, lw=2.2, ls="--", label="奖励 $R(x)$(偏向 $+1$)")
ax.bar([-1], [0.5], width=0.20, color=GREY, alpha=0.75, label="训练前:$0.5/0.5$")
ax.bar([1], [0.5], width=0.20, color=GREY, alpha=0.75)
ax.bar([-1 + 0.26], [0.2], width=0.20, color=GREEN, label="训练后:$0.8/0.2$")
ax.bar([1 + 0.26], [0.8], width=0.20, color=GREEN)
ax.annotate("", xy=(1.26, 0.80), xytext=(1.0, 0.52),
            arrowprops=dict(arrowstyle="-|>", lw=2.4, color=GREEN))
ax.annotate("", xy=(-1.26, 0.20), xytext=(-1.0, 0.48),
            arrowprops=dict(arrowstyle="-|>", lw=2.4, color=RED))
ax.text(1.05, 0.875, "高奖励 → 概率↑", fontsize=9.6, color=GREEN, ha="center")
ax.text(-1.62, 0.30, "低奖励\n→ 概率↓", fontsize=9.6, color=RED, ha="center")
ax.set_xlim(-2.2, 2.2); ax.set_ylim(0, 1.02)
ax.set_xticks([-1, 1]); ax.set_xlabel("输出(两个模态)", fontsize=10.5)
ax.set_ylabel("概率", fontsize=10.5)
ax.set_title("(a) 目的地【会被改】:\n训练让它偏向高奖励的那一侧", fontsize=11.5)
ax.legend(fontsize=8.6, frameon=False, loc="upper left")
ax.tick_params(labelsize=9)

# ---------------- (b) 机制链条
ax2 = fig.add_axes([0.40, 0.17, 0.31, 0.66]); ax2.axis("off")
ax2.text(0.5, 0.99, "(b) GRPO 的机制链条", ha="center", va="top", fontsize=11.8, color=DARK)
steps = [("① 同一个 prompt,用 SDE 采 $G$ 个", "彼此【不同】(才有可比性)", GREEN),
         ("② 每个打分 $R_i$", "任务成功 / 物体数量 / 文字 / 审美…", DARK),
         ("③ 组内标准化", "$A_i=(R_i-\\bar R)/\\mathrm{std}$(比同组平均好多少)", BLUE),
         ("④ 算似然比 $\\rho=\\exp(\\log\\pi_\\theta-\\log\\pi_{\\rm old})$", "每步条件高斯,可算", BLUE),
         ("⑤ 更新 $\\theta$:最大化 $A_i\\cdot\\rho$(裁剪)", "好样本概率↑,差样本↓", RED),
         ("⑥ 加 KL 约束", "别偏离原模型太远(防塌缩)", PURPLE)]
y = 0.86
for name, note, col in steps:
    ax2.text(0.0, y, name, fontsize=9.8, color=col, va="top")
    ax2.text(0.05, y - 0.052, note, fontsize=8.8, color="#5d6d7e", va="top")
    y -= 0.132
ax2.text(0.0, 0.02,
         "如果组内奖励全相同 → $A_i\\equiv0$ → 梯度 $=0$ → 什么也学不到",
         fontsize=9.4, color=RED, va="bottom", bbox=dict(fc="#fdecea", ec="#e6a9a0"))

# ---------------- (c) 为什么必须 SDE
ax3 = fig.add_axes([0.735, 0.17, 0.245, 0.66]); ax3.axis("off")
ax3.text(0.5, 0.99, "(c) 为什么必须用 SDE", ha="center", va="top", fontsize=11.6, color=DARK)
rows = [["", "ODE", "SDE"],
        ["输出", "确定性映射\n$\\epsilon\\mapsto x$", "有条件分布\n$\\pi_\\theta(x'|x)$"],
        ["似然", "$\\delta$ 函数\n(算不了)", "高斯\n(能算)"],
        ["$\\log\\pi$", "无定义", "有"],
        ["似然比 $\\rho$", "无法构造", "可构造"],
        ["策略梯度", "无法估计", "可以"]]
tb = ax3.table(cellText=rows, cellLoc="center", bbox=[0.0, 0.44, 1.0, 0.46])
tb.auto_set_font_size(False); tb.set_fontsize(8.4)
for (r, c), cell in tb.get_celld().items():
    if r == 0:
        cell.set_facecolor("#eef2f7")
    if c == 1 and r >= 3:
        cell.set_facecolor("#fdecea")
    if c == 2 and r >= 3:
        cell.set_facecolor("#eef7f1")
ax3.text(0.0, 0.38,
         "不是「为了探索」才用 SDE,\n而是【没有似然就没有策略梯度】。\n\n"
         "SDE 顺带的好处:探索更强。",
         fontsize=9.0, color=DARK, va="top")
ax3.text(0.0, 0.075,
         "一句话:SDE 把\n\"确定性生成器\"变成了\n\"可优化的随机策略\"。",
         fontsize=9.4, color=GREEN, va="bottom", bbox=dict(fc="#eef7f1", ec="#a9d5bb"))

fig.suptitle("那 GRPO 的差异是什么:采样器换了(目的地暂不变)→ 参数改了(目的地故意变)", fontsize=12.4, y=0.99)
fig.savefig(RF + "fg_grpo_what_changes.png", bbox_inches="tight")
print("wrote", RF + "fg_grpo_what_changes.png")
