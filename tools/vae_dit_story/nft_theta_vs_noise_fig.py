r"""漂移里已经有 θ 了,为什么还要加噪声?

因为这是两个【不同的需求】,由两个不同的来源满足:

  需求 1:参数能改变输出          ← 漂移里的 v_θ 提供(可导的因果链)
  需求 2:输出要有可算的【概率】   ← 只有噪声能提供

  确定性映射 π = δ(mean_θ):θ 在里面(位置随 θ 平移),但密度不可用
  高斯     π = N(mean_θ, s²):位置带 θ(可导)+ 宽度 s(可算密度)

⇒ 只利用需求 1 的路线:可微奖励反传 / DiffusionNFT 式监督目标 → 【不需要噪声】
⇒ 需要需求 2 的路线:策略梯度(GRPO)                    → 【必须加噪声】
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
gauss = lambda x, m, s: np.exp(-(x - m) ** 2 / (2 * s ** 2)) / (s * np.sqrt(2 * np.pi))

fig = plt.figure(figsize=(15.2, 5.6), dpi=150)

# ---------------- (a) 两个需求
ax = fig.add_axes([0.04, 0.10, 0.36, 0.78]); ax.axis("off")
ax.text(0.5, 1.00, "(a) 两个需求,两个来源", ha="center", va="top", fontsize=12, color=DARK)
ax.add_patch(plt.Rectangle((0.02, 0.62), 0.96, 0.28, fc="#eaf2fa", ec=BLUE, lw=1.8))
ax.text(0.05, 0.855, "需求 1:参数能改变输出", fontsize=10.2, color=BLUE, va="top")
ax.text(0.05, 0.80, "来源:漂移里的 $v_\\theta$\n"
                    "$\\mathrm{mean}_\\theta(x,\\sigma)=A(\\sigma)x+B(\\sigma)v_\\theta$\n"
                    "⇒ 输出随 $\\theta$ 移动,而且【可导】",
        fontsize=9.0, color=DARK, va="top")
ax.text(0.05, 0.655, "✓ 这就够了 —— 只要你不打算算概率", fontsize=9.0, color=BLUE, va="center")
ax.add_patch(plt.Rectangle((0.02, 0.22), 0.96, 0.30, fc="#eef7f1", ec=GREEN, lw=1.8))
ax.text(0.05, 0.495, "需求 2:输出要有可算的【概率】", fontsize=10.2, color=GREEN, va="top")
ax.text(0.05, 0.44, "来源:每步注入的噪声 $s\\xi$\n"
                    "$\\pi_\\theta(x'|x)=N(\\mathrm{mean}_\\theta,\\,s^2)$\n"
                    "⇒ 才有密度、才有 $\\log\\pi_\\theta$、才有 $\\rho$",
        fontsize=9.0, color=DARK, va="top")
ax.text(0.05, 0.255, "✓ 这只有噪声能给 —— 漂移给不了", fontsize=9.0, color=GREEN, va="center")
ax.text(0.5, 0.13,
        "两条路线的分岔:\n"
        "只利用需求 1:可微奖励反传 / DiffusionNFT 式监督目标 → 【不加噪声】\n"
        "需要需求 2:策略梯度(GRPO) → 【必须加噪声】",
        fontsize=9.0, color=DARK, ha="center", va="center",
        bbox=dict(fc="#fffbe6", ec="#e0c060"))
ax.set_xlim(0, 1); ax.set_ylim(0.04, 1.02)

# ---------------- (b) δ vs 高斯
ax2 = fig.add_axes([0.46, 0.17, 0.27, 0.66])
xs = np.linspace(-2.2, 2.2, 900)
ax2.plot(xs, gauss(xs, 0.45, 0.28), color=GREEN, lw=2.8, label="加了噪声:$N(\\mathrm{mean}_\\theta,s^2)$")
ax2.plot([0.0], [0.0], "o", ms=0)
ax2.vlines(0.0, 0, 3.1, color=BLUE, lw=3.0, label="确定性的 $\\delta$(漂移的输出)")
ax2.plot([0.0], [3.1], "^", ms=11, color=BLUE)
ax2.annotate("", xy=(0.45, 0.72), xytext=(0.0, 0.72),
             arrowprops=dict(arrowstyle="-|>", lw=2.2, color=DARK))
ax2.text(0.225, 0.78, "$\\theta$ 挪位置", fontsize=9.0, color=DARK, ha="center")
ax2.annotate("", xy=(0.17, 0.20), xytext=(0.73, 0.20),
             arrowprops=dict(arrowstyle="<|-|>", lw=2.0, color=GREEN))
ax2.text(0.45, 0.24, "宽度 $s$ ⇒ 概率的载体", fontsize=9.0, color=GREEN, ha="center")
ax2.text(-2.1, 2.75, "$\\delta$:θ 在里面(位置随 θ 平移)\n但密度不可用(δ 函数)\n⇒ 有\"改进方向\",没有\"概率\"",
         fontsize=8.8, color=BLUE, va="top")
ax2.text(-2.1, 1.55, "高斯:位置带 θ(可导)\n+ 宽度(可算密度)\n⇒ 两条需求都满足",
         fontsize=8.8, color=GREEN, va="top")
ax2.set_xlim(-2.2, 2.2); ax2.set_ylim(-0.05, 3.3)
ax2.set_yticks([]); ax2.set_xlabel("这一步的落点 $x'$", fontsize=10.5)
ax2.set_title("(b) 为什么\"有 θ\"不等于\"有概率\"", fontsize=11.3)
ax2.legend(fontsize=8.2, frameon=False, loc="upper right")
ax2.tick_params(labelsize=9)

# ---------------- (c) 路线对照
ax3 = fig.add_axes([0.755, 0.17, 0.23, 0.66]); ax3.axis("off")
ax3.text(0.5, 0.99, "(c) 所以路线才分岔", ha="center", va="top", fontsize=11.4, color=DARK)
rows = [["", "只用\n需求1", "需求\n1+2"],
        ["要噪声吗", "不要", "要"],
        ["可微奖励反传\n(DRaFT/ReFL)", "✓", "—"],
        ["DiffusionNFT\n(监督目标)", "✓", "—"],
        ["Flow-GRPO\n(策略梯度)", "✗", "✓"]]
tb = ax3.table(cellText=rows, cellLoc="center", bbox=[0.0, 0.48, 1.0, 0.46])
tb.auto_set_font_size(False); tb.set_fontsize(7.8)
for (r, c), cell in tb.get_celld().items():
    if r == 0:
        cell.set_facecolor("#eef2f7")
    if c == 1 and r > 0:
        cell.set_facecolor("#eaf2fa")
    if c == 2 and r > 0:
        cell.set_facecolor("#eef7f1")
ax3.text(0.0, 0.42,
         "一句话:\n"
         "$v_\\theta$ 给你\"往哪改\",\n"
         "$s\\xi$ 给你\"改多少才叫一次\n"
         "　　 可以计算概率的决策\"。\n\n"
         "只要前者 → 不加噪声\n"
         "(NFT 就这么做);\n"
         "要后者 → 必须加\n"
         "(GRPO 的代价)。",
         fontsize=8.4, color=DARK, va="top",
         bbox=dict(fc="#fffbe6", ec="#e0c060"))

fig.suptitle("θ 在漂移里 ⇒ 能【改变输出】;但\"输出有可算的概率\"只能靠噪声 —— 两件事,两条路线",
             fontsize=11.8, y=0.99)
fig.savefig(RF + "fg_theta_vs_noise.png", bbox_inches="tight")
print("wrote", RF + "fg_theta_vs_noise.png")
