r"""每一步的 log-prob 是怎么来的 + loss 在奖励/惩罚什么。

(a) 一步 = 一个高斯 → log-prob = 该高斯在【实际抽到的点】上的对数高度
    θ 只能通过【均值】改动它(方差 s 与 θ 无关)
(b) loss 在奖励/惩罚什么(2×2)+ REINFORCE 梯度形式 + "loss 里没有什么"
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
N = lambda x, m: np.exp(-(x - m) ** 2 / 2) / np.sqrt(2 * np.pi)

fig = plt.figure(figsize=(15.2, 5.5), dpi=150)

# ---------------- (a) 一步的 log-prob
ax = fig.add_axes([0.05, 0.16, 0.36, 0.68])
xs = np.linspace(-4, 4, 800)
ax.plot(xs, N(xs, 0.0), color=DARK, lw=2.6, label="这一步的高斯 $N(m_\\theta,\\,s^2)$")
ax.plot(xs, N(xs, 0.2), color=ORANGE, lw=2.2, ls="--",
        label="$\\theta$ 把均值挪 $\\delta/s=0.2$ 之后")
XI = 1.5
ax.axvline(XI, color=GREY, ls=":", lw=1.2)
ax.plot([XI], [N(XI, 0.0)], "o", ms=10, color=BLUE, zorder=6)
ax.plot([XI], [N(XI, 0.2)], "o", ms=10, color=ORANGE, zorder=6)
ax.annotate("实际抽到的点 $\\xi=1.5$\n($x'=m_{old}+s\\xi$)", xy=(XI, N(XI, 0.0)),
            xytext=(1.75, 0.20), fontsize=9.6, color=BLUE,
            arrowprops=dict(arrowstyle="-|>", lw=1.3, color=BLUE))
ax.annotate("$\\log\\pi$ 就是这条曲线\n在 $\\xi$ 处的【高度】取对数", xy=(XI, N(XI, 0.2)),
            xytext=(1.35, 0.055), fontsize=9.6, color=ORANGE,
            arrowprops=dict(arrowstyle="-|>", lw=1.3, color=ORANGE))
ax.text(-3.9, 0.42,
        "$\\log\\pi_\\theta(x'\\,|\\,x)=-\\dfrac{\\xi^2}{2}-\\log s-\\frac{1}{2}\\log 2\\pi$\n"
        "$s=0.05,\\ \\xi=1.5$:  $-1.125+2.996-0.919=+0.952$\n"
        "$\\theta$ 只改【均值】$\\Rightarrow$ 同一点高度变成 $+1.232$,$\\rho=e^{0.28}=1.32$",
        fontsize=9.2, color=DARK, va="top",
        bbox=dict(fc="#fffbe6", ec="#e0c060"))
ax.set_xlim(-4, 4); ax.set_ylim(-0.03, 0.52)
ax.set_xlabel("$\\xi=$(实际值 $-$ 均值)$/s$   (标准化噪声)", fontsize=10.5)
ax.set_ylabel("密度", fontsize=10.5)
ax.set_title("(a) 每一步的 log-prob:\n高斯在实际抽到的那一点上的对数高度", fontsize=11.5)
ax.legend(fontsize=8.8, frameon=False, loc="upper left")
ax.tick_params(labelsize=9)

# ---------------- (b) loss 在奖励什么
ax2 = fig.add_axes([0.45, 0.16, 0.53, 0.68]); ax2.axis("off")
ax2.text(0.5, 1.00, "(b) loss 在奖励什么、惩罚什么", ha="center", va="top", fontsize=12, color=DARK)
rows = [["样本", "$\\rho$ 变化", "对 $\\mathcal{L}=-A\\rho$ 的影响", "等效于"],
        ["$A>0$(比同组好)", "$\\uparrow$", "loss $\\downarrow$", "**奖励**它更常出现"],
        ["$A>0$", "$\\downarrow$", "loss $\\uparrow$", "**惩罚**它变少"],
        ["$A<0$(比同组差)", "$\\uparrow$", "loss $\\uparrow$", "**惩罚**它更常出现"],
        ["$A<0$", "$\\downarrow$", "loss $\\downarrow$", "**奖励**它变少"],
        ["偏离参考模型", "—", "KL 项 $\\uparrow$", "**惩罚**跑太远"]]
tb = ax2.table(cellText=rows, cellLoc="center", bbox=[0.0, 0.52, 1.0, 0.40])
tb.auto_set_font_size(False); tb.set_fontsize(8.6)
for (r, c), cell in tb.get_celld().items():
    if r == 0:
        cell.set_facecolor("#eef2f7")
    if r in (1, 4):
        cell.set_facecolor("#eef7f1")
    if r in (2, 3):
        cell.set_facecolor("#fdecea")
    if r == 5:
        cell.set_facecolor("#f4ecf7")
    if c == 3:
        t = cell.get_text().get_text().replace("**", "")
        cell.get_text().set_text(t)
ax2.text(0.0, 0.46,
         "梯度形式(默认配置下 $\\rho\\equiv1$):\n"
         "$\\nabla_\\theta\\mathcal{L}=-\\mathbb{E}[\\,A\\cdot\\nabla_\\theta\\log\\pi_\\theta\\,]$"
         "   ← REINFORCE + 组内基线\n"
         "而 $\\nabla_\\theta\\log\\pi_\\theta=\\dfrac{\\xi}{s}\\cdot\\dfrac{\\partial m_\\theta}{\\partial\\theta}$"
         "  ⇒ 噪声 $\\xi$ 就是探索方向:\n"
         "结果好($A>0$)就沿这个方向把均值多推一点,结果差就反着推。",
         fontsize=9.2, color=DARK, va="top")
ax2.text(0.0, 0.10,
         "loss 里【没有】的东西:目标图像、正确答案、逐像素损失。\n"
         "唯一的信号是【组内相对好坏】—— 它只回答\"哪个输出更好\",不回答\"正确输出长什么样\"。",
         fontsize=9.4, color=GREEN, va="bottom", bbox=dict(fc="#eef7f1", ec="#a9d5bb"))

fig.suptitle("每一步的 log-prob = 那一步高斯在实际抽到点上的对数密度;loss = 按组内相对优势增减这些概率", fontsize=12.2, y=0.99)
fig.savefig(RF + "fg_logprob_and_loss.png", bbox_inches="tight")
print("wrote", RF + "fg_logprob_and_loss.png")
