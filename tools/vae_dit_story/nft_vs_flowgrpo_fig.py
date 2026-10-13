r"""Flow-GRPO vs DiffusionNFT:反向过程策略梯度 vs 前向过程监督回归。

DiffusionNFT 的 loss(论文 Algorithm 1 / Theorem 3.2):
  v⁺_θ = (1−β) v_old + β v_θ        (隐式"好策略")
  v⁻_θ = (1+β) v_old − β v_θ        (隐式"坏策略")
  L = r·‖v⁺_θ − v‖² + (1−r)·‖v⁻_θ − v‖²        ← 纯监督回归!

β=1 时最直观:
  r≈1(好样本): v_θ → v            (模仿它)
  r≈0(坏样本): v_θ → 2v_old − v    (绕 v_old 反射 = 反着学)

(a) 两条流水线对照
(b) 隐式正/负策略的几何(β=1)
(c) 对照表 + 消融要点 + 他们对 GRPO 的独立推导
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


def box(ax, x, y, w, h, text, fc, ec, fs=8.2, tc=DARK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.010", fc=fc, ec=ec, lw=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc)


fig = plt.figure(figsize=(15.2, 6.0), dpi=150)

# ---------------- (a) 两条流水线
ax = fig.add_axes([0.02, 0.05, 0.46, 0.85]); ax.axis("off")
ax.text(0.5, 1.00, "(a) 两条流水线", ha="center", va="top", fontsize=12, color=DARK)

ax.text(0.0, 0.905, "Flow-GRPO:在【反向采样过程】上做策略梯度", fontsize=9.8, color=RED)
steps_a = [("SDE 采 $G$ 个\n记每步 $\\log\\pi$", "#fdecea"),
           ("打分 $R_i$", "#fdecea"),
           ("组内优势 $A_i$", "#fdecea"),
           ("每步 $\\rho_j$\n+clip+KL", "#fdecea"),
           ("更新 $\\theta$", "#fdecea")]
for k, (txt, fc) in enumerate(steps_a):
    x0 = 0.005 + k * 0.198
    box(ax, x0, 0.72, 0.175, 0.135, txt, fc, RED)
    if k < 4:
        ax.add_patch(FancyArrowPatch((x0 + 0.175, 0.787), (x0 + 0.198, 0.787),
                                     arrowstyle="-|>", mutation_scale=10, lw=1.5, color=DARK))
ax.text(0.0, 0.66, "必须:① 把 ODE 换成 SDE(要似然) ② 记整条轨迹的逐步 log-prob ③ 推理时还要 CFG",
        fontsize=8.4, color=RED)

ax.text(0.0, 0.56, "DiffusionNFT:在【前向过程】上做监督回归", fontsize=9.8, color=GREEN)
steps_b = [("任意 solver 采 $K$ 个\n(不记 log-prob)", "#eef7f1"),
           ("打分 $r_{raw}$ → 组内归一\n→ $r=0.5+0.5\\mathrm{clip}$", "#eef7f1"),
           ("只留【干净样本】$x_0$\n前向加噪 $x_t$", "#eef7f1"),
           ("加权 FM 回归\n$r\\|v^+_\\theta-v\\|^2+(1-r)\\|v^-_\\theta-v\\|^2$", "#eef7f1"),
           ("更新 $\\theta$", "#eef7f1")]
for k, (txt, fc) in enumerate(steps_b):
    x0 = 0.005 + k * 0.198
    box(ax, x0, 0.36, 0.175, 0.145, txt, fc, GREEN)
    if k < 4:
        ax.add_patch(FancyArrowPatch((x0 + 0.175, 0.4325), (x0 + 0.198, 0.4325),
                                     arrowstyle="-|>", mutation_scale=10, lw=1.5, color=DARK))
ax.text(0.0, 0.30, "不需要:① 似然 ② 轨迹(只要干净样本) ③ CFG   —— 也【没有】ratio/clip/KL",
        fontsize=8.4, color=GREEN)
ax.text(0.0, 0.19,
        "他们对 GRPO 的独立推导(Appendix B.3):\n"
        "$\\nabla_\\theta\\mathcal{L}=\\frac{m(s,t)}{n(s,t)}\\nabla_\\theta[\\frac{1}{N}\\sum_i (A^{(i)}\\epsilon^{(i)})^\\top v_\\theta]$\n"
        "⇒ \"FlowGRPO aligns the velocity field with the\n"
        "　 advantage-weighted noise\" —— 与我们推的 $\\mathbb{E}[A\\xi]=s\\,\\partial\\mathbb{E}[R]/\\partial m$ 一致",
        fontsize=8.4, color=DARK, va="top", bbox=dict(fc="#fffbe6", ec="#e0c060"))

# ---------------- (b) 隐式正/负策略的几何
ax2 = fig.add_axes([0.51, 0.10, 0.28, 0.76]); ax2.axis("off")
ax2.text(0.5, 1.00, "(b) 隐式正/负策略($\\beta=1$)", ha="center", va="top", fontsize=11.6, color=DARK)
ax2.annotate("", xy=(0.98, 0.55), xytext=(0.02, 0.55),
             arrowprops=dict(arrowstyle="-|>", lw=1.6, color=GREY))
ax2.text(0.5, 0.50, "速度空间", fontsize=8.6, color=GREY, ha="center")
# 轴上的位置: v_old=0.5, v_sample=0.8
xo, vs = 0.5, 0.80
for x, lab, c in [(xo, "$v_{\\rm old}$", DARK), (vs, "样本目标 $v$\n(这一张图)", BLUE),
                  (2 * xo - vs, "$2v_{\\rm old}-v$\n(反射点)", ORANGE)]:
    ax2.plot([x], [0.55], "o", ms=10, color=c, zorder=5)
    ax2.plot([x, x], [0.55, 0.63], color=c, lw=1.2, ls=":")
    ax2.text(x, 0.655, lab, fontsize=8.6, color=c, ha="center", va="bottom")
ax2.add_patch(FancyArrowPatch((0.30, 0.40), (vs, 0.40), arrowstyle="-|>", mutation_scale=13,
                              lw=2.4, color=BLUE))
ax2.text(0.55, 0.35, "好样本 $r\\approx1$:$v_\\theta\\to v$\n= 模仿它", fontsize=9.0, color=BLUE,
         ha="center", va="top")
ax2.add_patch(FancyArrowPatch((0.70, 0.22), (2 * xo - vs, 0.22), arrowstyle="-|>", mutation_scale=13,
                              lw=2.4, color=ORANGE))
ax2.text(0.55, 0.17, "坏样本 $r\\approx0$:$v_\\theta\\to 2v_{\\rm old}-v$\n= 绕 $v_{\\rm old}$ 反射,反着学",
         fontsize=9.0, color=ORANGE, ha="center", va="top")
ax2.text(0.5, 0.05, "⇒ 负样本不是丢掉,而是当【反向目标】\n(这就是 Negative-aware 的含义)",
         fontsize=9.2, color=GREEN, ha="center", va="bottom",
         bbox=dict(fc="#eef7f1", ec="#a9d5bb"))
ax2.set_xlim(0, 1); ax2.set_ylim(0, 1.02)

# ---------------- (c) 对照表 + 消融
ax3 = fig.add_axes([0.80, 0.10, 0.19, 0.76]); ax3.axis("off")
ax3.text(0.5, 1.00, "(c) 差异与实证", ha="center", va="top", fontsize=11.4, color=DARK)
rows = [["", "GRPO", "NFT"],
        ["优化对象", "反向\n过程", "前向\n过程"],
        ["似然", "必须", "不需要"],
        ["训练数据", "整条\n轨迹", "只干净\n样本"],
        ["solver", "固定\nSDE", "任意"],
        ["CFG", "需要", "不需要"],
        ["ratio\n/clip/KL", "有", "没有"],
        ["负样本", "降概率", "反向\n监督"],
        ["效率", "5k+\n步", "1k\n步"]]
tb = ax3.table(cellText=rows, cellLoc="center", bbox=[0.0, 0.44, 1.0, 0.54])
tb.auto_set_font_size(False); tb.set_fontsize(7.4)
for (r, c), cell in tb.get_celld().items():
    if r == 0:
        cell.set_facecolor("#eef2f7")
    if c == 1 and r > 0:
        cell.set_facecolor("#fdecea")
    if c == 2 and r > 0:
        cell.set_facecolor("#eef7f1")
ax3.text(0.0, 0.38,
         "消融里最关键的两条:\n"
         "· 去掉负样本分支 → 奖励\n"
         "　【几乎立刻崩溃】\n"
         "　(LLM 里 RFT 仍是强 baseline,\n"
         "　 扩散 RL 里不行)\n"
         "· ODE 采样器【优于】SDE\n"
         "　(尤其 PickScore 这种对噪声\n"
         "　 敏感的奖励)",
         fontsize=7.8, color=DARK, va="top",
         bbox=dict(fc="#fffbe6", ec="#e0c060"))
ax3.text(0.0, 0.03,
         "其它:$\\beta\\approx1$ 稳定;\n时间权重应在【大 t】更高;\n软更新 $\\eta$ 由小到大最好",
         fontsize=7.6, color=GREEN, va="bottom", bbox=dict(fc="#eef7f1", ec="#a9d5bb"))

fig.suptitle("Flow-GRPO vs DiffusionNFT:一个是\"反向过程 + 策略梯度(要似然)\",一个是\"前向过程 + 监督回归(不要似然)\"",
             fontsize=12.0, y=0.99)
fig.savefig(RF + "fg_flowgrpo_vs_nft.png", bbox_inches="tight")
print("wrote", RF + "fg_flowgrpo_vs_nft.png")
