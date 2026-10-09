r"""GRPO 的 loss / 优势 / 重要性采样,按代码实际算法画出来。

(a) 计算链条:采样 → 打分 → 组内标准化 → 广播+裁剪 → 逐步 ratio → 裁剪 surrogate + KL
(b) 数值例子(G=4)与裁剪行为
(c) 超参 + "ratio 什么时候真的起作用"
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

fig = plt.figure(figsize=(15.4, 5.8), dpi=150)

# ---------------- (a) 链条
ax = fig.add_axes([0.03, 0.10, 0.40, 0.78]); ax.axis("off")
ax.text(0.5, 1.00, "(a) 一次更新的完整计算链", ha="center", va="top", fontsize=12, color=DARK)
chain = [
    ("① 采样(no_grad)", "同一 prompt 用 SDE 采 $G$ 个,记下【每一步】的 $\\log\\pi_{\\rm old}$", GREEN),
    ("② 打分", "每个样本一个标量奖励 $R_i$(检测器/OCR/偏好模型…)", DARK),
    ("③ 优势(组内标准化)", "$A_i=\\dfrac{R_i-\\bar R_{group}}{\\mathrm{std}_{group}+10^{-4}}$", BLUE),
    ("④ 广播到各步 + 裁剪", "$A_i$ 复制到 $T$ 步;再 $\\mathrm{clamp}(A_i,\\pm5)$", BLUE),
    ("⑤ 每步重要性比", "$\\rho_{j}=\\exp(\\log\\pi_\\theta(x_j)-\\log\\pi_{\\rm old}(x_j))$", ORANGE),
    ("⑥ 裁剪 surrogate + KL", "$\\mathcal{L}=\\overline{\\max(-A\\rho,-A\\,\\mathrm{clip}(\\rho,1\\pm10^{-3}))}+\\beta\\mathrm{KL}$", RED),
]
y = 0.95
for name, expr, col in chain:
    ax.add_patch(plt.Rectangle((0.0, y - 0.135), 1.0, 0.125, fc="#fbfcfd", ec="#e3e8ee", lw=1.2))
    ax.text(0.02, y - 0.022, name, fontsize=10.2, color=col, va="top", fontweight="bold")
    ax.text(0.05, y - 0.075, expr, fontsize=9.4, color=DARK, va="top")
    y -= 0.137
ax.text(0.02, 0.02, "KL 是【闭式高斯】:$\\|m_\\theta-m_{\\rm ref}\\|^2/(2s^2)$,参考策略 = 关掉 LoRA",
        fontsize=9.0, color=PURPLE, va="bottom")

# ---------------- (b) 数值例子
ax2 = fig.add_axes([0.455, 0.10, 0.31, 0.78]); ax2.axis("off")
ax2.text(0.5, 1.00, "(b) 数值例:$G=4$,奖励 $[1,0,1,1]$", ha="center", va="top", fontsize=11.8, color=DARK)
ax2.text(0.0, 0.91,
         "$\\bar R=0.75$,  $\\mathrm{std}=0.433$  →  $A=[+0.577,\\,-1.732,\\,+0.577,\\,+0.577]$\n"
         "(好的样本拿正优势,差的拿负优势;组内平均被减掉 = 不需要 value 网络)",
         fontsize=9.4, color=DARK, va="top")
rows = [["$A$", "$\\rho$", "未裁剪 $-A\\rho$", "裁剪后", "$\\mathcal{L}=\\max$"],
        ["$+0.577$", "$1.00$", "$-0.5772$", "$-0.5772$", "$-0.5772$"],
        ["$+0.577$", "$1.01$", "$-0.5830$", "$-0.5778$", "$-0.5778$"],
        ["$+0.577$", "$0.50$", "$-0.2886$", "$-0.5766$", "$-0.2886$"],
        ["$-1.732$", "$1.50$", "$+2.5975$", "$+1.7334$", "$+2.5975$"],
        ["$-1.732$", "$0.50$", "$+0.8658$", "$+1.7299$", "$+1.7299$"]]
tb = ax2.table(cellText=rows, cellLoc="center", bbox=[0.0, 0.44, 1.0, 0.34])
tb.auto_set_font_size(False); tb.set_fontsize(8.2)
for (r, c), cell in tb.get_celld().items():
    if r == 0:
        cell.set_facecolor("#eef2f7")
    if r in (2, 5):
        cell.set_facecolor("#fff4e5")
ax2.text(0.0, 0.38,
         "读法(标准 PPO 信赖域):\n"
         "· $A>0$(好样本):鼓励 $\\rho\\uparrow$;但 $\\rho>1+\\varepsilon$ 后\n"
         "　【不再给额外收益】(被裁住),防止一步跑太远\n"
         "· $A<0$(差样本):鼓励 $\\rho\\downarrow$,同样被裁住\n"
         "· $\\varepsilon=10^{-3}$ 极小 → 每步只许动 0.1%",
         fontsize=9.0, color=DARK, va="top")
ax2.text(0.0, 0.03,
         "另:$\\mathrm{approx\\_KL}=0.5\\,\\overline{(\\log\\pi_\\theta-\\log\\pi_{\\rm old})^2}$\n"
         "以及 clipfrac 只用于监控,不参与优化。",
         fontsize=8.8, color=GREY, va="bottom")

# ---------------- (c) 超参 + ratio 何时起作用
ax3 = fig.add_axes([0.79, 0.10, 0.20, 0.78]); ax3.axis("off")
ax3.text(0.5, 1.00, "(c) 超参(通用配置)", ha="center", va="top", fontsize=11.6, color=DARK)
rows3 = [["项", "值"],
         ["组大小 $G$", "4(或 24)"],
         ["训练步 $T$", "$\\lfloor20\\times0.99\\rfloor=19$"],
         ["clip_range", "$10^{-3}$"],
         ["advantage clip", "$\\pm5$"],
         ["KL 权重 $\\beta$", "0.004"],
         ["inner epochs", "1"]]
tb3 = ax3.table(cellText=rows3, cellLoc="center", bbox=[0.0, 0.56, 1.0, 0.40])
tb3.auto_set_font_size(False); tb3.set_fontsize(8.4)
for (r, c), cell in tb3.get_celld().items():
    if r == 0:
        cell.set_facecolor("#eef2f7")
ax3.text(0.0, 0.50,
         "★ 一个容易漏掉的点:\n"
         "梯度累积覆盖【整条去噪链】\n"
         "($\\mathrm{gas}=1\\times T$),且 inner epochs $=1$\n"
         "⇒ 同一个子批次内部 $\\theta$ 不动\n"
         "⇒ 该批次里 $\\rho\\equiv1$、裁剪不起作用\n"
         "⇒ 更新退化为【普通策略梯度 + 组内基线】",
         fontsize=8.6, color=DARK, va="top",
         bbox=dict(fc="#fffbe6", ec="#e0c060"))
ax3.text(0.0, 0.10,
         "所以:代码实现了完整 PPO,\n但默认配置下它是\n\"几乎纯 on-policy\"的更新。",
         fontsize=8.8, color=GREEN, va="bottom", bbox=dict(fc="#eef7f1", ec="#a9d5bb"))

fig.suptitle("GRPO 在这里是怎么算的:组内标准化出优势 → 逐步重要性比 → 裁剪 surrogate + 闭式高斯 KL", fontsize=12.6, y=0.995)
fig.savefig(RF + "fg_grpo_computation.png", bbox_inches="tight")
print("wrote", RF + "fg_grpo_computation.png")
