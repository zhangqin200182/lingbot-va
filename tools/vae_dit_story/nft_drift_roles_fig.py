r"""漂移的读法:往高密度回一点 + 沿 v_θ 走一点?—— 对,但两项的【配比】随 σ 变化极大。

一步位移(生成方向,dσ<0):
  Δ = f·dσ + s·ξ ,   f = v_θ − (g²/2)∇log p
  ⇒ Δ = −|dσ|(1+nl²/2)·v_θ  +  |dσ|(g²/2)·∇log p  +  s·ξ
         └ 送粒子(沿 −v_θ)┘    └ 往高密度回(收宽度)┘

配比(除以 |dσ|,nl=0.5):
  σ=0.1 : 1.125 vs 0.014  → 补偿仅 1.2%   (几乎纯"按 −v_θ 走")
  σ=0.5 : 1.125 vs 0.125  → 10%
  σ=0.7 : 1.125 vs 0.292  → 20.6%
  σ=0.9 : 1.125 vs 1.125  → 50%
  σ=0.99: 1.125 vs 12.375 → 91.7%        (靠噪声端几乎全是"往回收")

核对:往高密度项正好抵消噪声加宽(高斯,一阶)
  σ=0.7: 噪声 +0.03070,补偿造成 −0.03025 ✓
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

fig = plt.figure(figsize=(15.2, 5.6), dpi=150)

# ---------------- (a) 三项分解
ax = fig.add_axes([0.03, 0.10, 0.35, 0.78]); ax.axis("off")
ax.text(0.5, 1.00, "(a) 一步位移 = 三项", ha="center", va="top", fontsize=12, color=DARK)
ax.plot([0.10, 0.92], [0.62, 0.62], color=GREY, lw=1.4)
ax.plot([0.10], [0.62], "o", ms=10, color=DARK)
ax.text(0.10, 0.68, "当前位置 $x$", fontsize=9.2, color=DARK, ha="center")
ax.add_patch(FancyArrowPatch((0.10, 0.52), (0.55, 0.52), arrowstyle="-|>", mutation_scale=13,
                             lw=2.8, color=BLUE))
ax.text(0.33, 0.44, "① 送粒子:$-|d\\sigma|(1+\\frac{nl^2}{2})v_\\theta$", fontsize=8.8, color=BLUE,
        ha="center", va="top")
ax.add_patch(FancyArrowPatch((0.55, 0.62), (0.68, 0.62), arrowstyle="-|>", mutation_scale=13,
                             lw=2.8, color=ORANGE, ls="--"))
ax.text(0.62, 0.70, "② 往高密度回:$+|d\\sigma|\\frac{g^2}{2}\\nabla\\log p$", fontsize=8.8,
        color=ORANGE, ha="center", va="bottom")
ax.plot([0.68], [0.62], "o", ms=10, color=GREEN)
ax.text(0.74, 0.62, "新中心", fontsize=9.2, color=GREEN, va="center")
ax.add_patch(FancyArrowPatch((0.68, 0.30), (0.68, 0.42), arrowstyle="<|-|>", mutation_scale=11,
                             lw=2.0, color=PURPLE))
ax.text(0.70, 0.36, "③ 再加宽 $s=\\sqrt{g^2|d\\sigma|}$\n(然后按这个高斯抽一次)", fontsize=8.8,
        color=PURPLE, va="center")
ax.text(0.02, 0.20,
        "角色分工:\n"
        "① 决定\"粒子往哪去\" —— 这是 $\\theta$ 影响学习的主通道\n"
        "② 决定\"刚被噪声撑开的分布收回来\" —— 保值分布\n"
        "③ 探索;幅度与 $\\theta$ 无关",
        fontsize=9.0, color=DARK, va="top", bbox=dict(fc="#fffbe6", ec="#e0c060"))
ax.set_xlim(0, 1); ax.set_ylim(0.05, 1.02)

# ---------------- (b) 配比随 σ
ax2 = fig.add_axes([0.43, 0.17, 0.28, 0.66])
sg = np.linspace(0.02, 0.995, 500); nl = 0.5
g2 = sg / (1 - sg) * nl ** 2
send = np.ones_like(sg) * (1 + nl ** 2 / 2)
back = g2 / 2
ax2.plot(sg, send, color=BLUE, lw=2.6, label="① 送粒子 $1+nl^2/2$(恒定)")
ax2.plot(sg, back, color=ORANGE, lw=2.6, label="② 往高密度回 $g^2/2$")
ax2.plot(sg, back / (send + back) * 10, color=GREEN, lw=2.0, ls="--",
         label="② 的占比 $\\times10$")
ax2.set_yscale("log")
ax2.set_xlabel("$\\sigma$", fontsize=10.5); ax2.set_ylabel("位移系数(对数)", fontsize=10.5)
ax2.set_title("(b) 靠数据端几乎纯\"按 $-v_\\theta$ 走\"\n靠噪声端一半以上是\"往回收\"", fontsize=11.2)
ax2.legend(fontsize=8.0, frameon=False, loc="upper left")
ax2.tick_params(labelsize=9)
for s, lab in [(0.1, "1.2%"), (0.5, "10%"), (0.7, "21%"), (0.9, "50%")]:
    ax2.annotate(lab, xy=(s, (s / (1 - s) * nl ** 2) / 2), xytext=(s, (s / (1 - s) * nl ** 2) / 2 * 1.8),
                 fontsize=8.2, color=ORANGE, ha="center")

# ---------------- (c) 表 + 抵消核对
ax3 = fig.add_axes([0.745, 0.17, 0.24, 0.66]); ax3.axis("off")
ax3.text(0.5, 0.99, "(c) 两项在干什么", ha="center", va="top", fontsize=11.4, color=DARK)
rows = [["项", "角色", "靠数据端", "靠噪声端"],
        ["① $-v_\\theta$", "送粒子\n(学习主通道)", "占 ~99%", "占少数"],
        ["② $\\nabla\\log p$", "收宽度\n(保值分布)", "占 ~1%", "占多数"]]
tb = ax3.table(cellText=rows, cellLoc="center", bbox=[0.0, 0.60, 1.0, 0.34])
tb.auto_set_font_size(False); tb.set_fontsize(7.8)
for (r, c), cell in tb.get_celld().items():
    if r == 0:
        cell.set_facecolor("#eef2f7")
    if c == 1 and r == 1:
        cell.set_facecolor("#eaf2fa")
    if c == 1 and r == 2:
        cell.set_facecolor("#fdf3e7")
ax3.text(0.0, 0.55,
         "核对(高斯,一阶):\n"
         "噪声把方差撑大 $+g^2|d\\sigma|$\n"
         "② 恰好把它收回 $\\approx-g^2|d\\sigma|$\n\n"
         "σ=0.7:噪声 +0.03070,\n"
         "　　　② 造成 −0.03025 ✓\n"
         "(差值是二阶项)",
         fontsize=8.4, color=DARK, va="top", bbox=dict(fc="#eef7f1", ec="#a9d5bb"))
ax3.text(0.0, 0.03,
         "所以你的读法对:\n\"往高密度回一点 +\n 按 $-v_\\theta$ 走一点\",\n只是\"一点\"的配比\n随 σ 变化很大。",
         fontsize=8.4, color=GREEN, va="bottom", bbox=dict(fc="#fffbe6", ec="#e0c060"))

fig.suptitle("漂移的读法:往高密度回一点 + 沿 $-v_\\theta$ 走一点 —— 对;但两项配比从 1% 变到 92%",
             fontsize=11.8, y=0.99)
fig.savefig(RF + "fg_drift_two_roles.png", bbox_inches="tight")
print("wrote", RF + "fg_drift_two_roles.png")
