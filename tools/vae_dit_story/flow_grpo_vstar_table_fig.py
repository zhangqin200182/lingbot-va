r"""把那张表画出来:同一个读数 (x_σ=0.1, σ=0.5) 上,两个候选数据点各自的 ε、速度、权重。

(a) 两条线都穿过同一个读数点 → 但斜率(=速度)方向相反
(b) 权重怎么来:先验 × 似然 → 归一化
(c) 加权平均得到场的输出 v*
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

RED, BLUE, GREEN, DARK, GREY, ORANGE = "#c0392b", "#2471a3", "#1e8449", "#2c3e50", "#95a5a6", "#b9770e"
RF = "/Users/kevin/code/lingbot-va/.claude/scratch/rf/"

XD, SD_ = 0.1, 0.5
X0 = np.array([1.0, -1.0])
PRIOR = 0.5
EPS = (XD - (1 - SD_) * X0) / SD_          # ε = (x_σ − (1−σ)x₀)/σ
LIK = np.exp(-EPS ** 2 / 2)                # N(ε;0,1),省掉公共常数 1/√(2π)
PROD = PRIOR * LIK
WT = PROD / PROD.sum()
VEL = EPS - X0                             # v = ε − x₀
VSTAR = float(WT @ VEL)
EX0 = float(WT @ X0)

fig = plt.figure(figsize=(15.0, 5.6), dpi=150)

# ---------------- (a) 两条线穿过同一点
ax = fig.add_axes([0.05, 0.14, 0.30, 0.70])
cols = [BLUE, RED]
for k in range(2):
    ax.plot([0, 1], [X0[k], EPS[k]], color=cols[k], lw=2.6, zorder=3)
    ax.plot([0], [X0[k]], "o", ms=11, color=cols[k], zorder=5)
    ax.text(-0.03, X0[k], f"$x_0={X0[k]:+.0f}$", fontsize=10.5, color=cols[k], ha="right", va="center")
    ax.plot([1], [EPS[k]], "o", ms=11, color=cols[k], zorder=5, alpha=0.55)
    ax.text(1.03, EPS[k], f"$\\epsilon={EPS[k]:+.2f}$", fontsize=10.5, color=cols[k], va="center")
    sx = 0.16 if k == 0 else 0.84
    ax.text(sx, (1 - sx) * X0[k] + sx * EPS[k] + (0.14 if k == 0 else -0.14),
            f"斜率 = $v={VEL[k]:+.2f}$", fontsize=10.5, color=cols[k],
            ha="left" if k == 0 else "right", va="bottom" if k == 0 else "top")
ax.plot([SD_], [XD], "o", ms=14, color=GREEN, zorder=7)
ax.axvline(SD_, color=GREY, ls="--", lw=1.1)
ax.text(SD_, XD + 0.13, f"同一个读数\n$x_\\sigma={XD},\\ \\sigma={SD_}$", fontsize=10.5,
        color=GREEN, ha="center")
ax.set_xlim(-0.28, 1.30); ax.set_ylim(-1.55, 1.55)
ax.set_xlabel("$\\sigma$", fontsize=11); ax.set_ylabel("$x$", fontsize=11)
ax.set_title("(a) 两条线都穿过同一个读数,但斜率相反", fontsize=11.5)
ax.tick_params(labelsize=9)

# ---------------- (b) 权重 = 先验 × 似然
ax2 = fig.add_axes([0.42, 0.14, 0.25, 0.70])
labels = ["先验\n$p(x_0)$", "似然\n$N(\\epsilon;0,1)$", "乘积\n(未归一化)", "归一化权重\n$w_k$"]
vals = np.array([[PRIOR, PRIOR], [LIK[0], LIK[1]], [PROD[0], PROD[1]], [WT[0], WT[1]]])
xpos = np.arange(4)
w = 0.34
for k, (c, name) in enumerate(zip(cols, ["$x_0=+1$", "$x_0=-1$"])):
    b = ax2.bar(xpos + (k - 0.5) * w, vals[:, k], width=w, color=c, alpha=0.85, label=name)
    for r, v in zip(b, vals[:, k]):
        ax2.text(r.get_x() + r.get_width() / 2, v + 0.015, f"{v:.4f}" if v < 0.7 else f"{v:.3f}",
                 ha="center", fontsize=8.8, color=DARK)
ax2.set_xticks(xpos); ax2.set_xticklabels(labels, fontsize=9.4)
ax2.set_ylim(0, 0.92); ax2.legend(fontsize=9.5, frameon=False)
ax2.set_title("(b) 权重从哪来:$0.5\\times0.7261$ 与 $0.5\\times0.4868$", fontsize=11.5)
ax2.tick_params(labelsize=9)
ax2.text(0.02, 0.86, f"两者相加 {PROD.sum():.4f}\n归一化 → {WT[0]:.4f} / {WT[1]:.4f}", fontsize=9.2,
         color=DARK, va="top", bbox=dict(fc="#fffbe6", ec="#e0c060"))

# ---------------- (c) 加权平均
ax3 = fig.add_axes([0.71, 0.14, 0.27, 0.70])
ax3.axhline(0, color=GREY, lw=0.9)
y0 = 0.0
for k in range(2):
    ax3.add_patch(FancyArrowPatch((0, 0.42 - 0.84 * k), (VEL[k] * 0.30, 0.42 - 0.84 * k),
                                  arrowstyle="-|>", mutation_scale=15, lw=3.0, color=cols[k]))
    ax3.text(VEL[k] * 0.30 + (0.04 if VEL[k] < 0 else -0.04), 0.52 - 0.84 * k,
             f"$v={VEL[k]:+.2f}$ × $w={WT[k]:.3f}$\n= {WT[k]*VEL[k]:+.4f}",
             fontsize=9.4, color=cols[k], ha="right" if VEL[k] < 0 else "left")
ax3.add_patch(FancyArrowPatch((0, -0.55), (VSTAR * 0.30, -0.55), arrowstyle="-|>",
                              mutation_scale=18, lw=3.6, color=GREEN, zorder=6))
ax3.text(VSTAR * 0.30, -0.66, f"$v^*=\\sum w_kv_k={VSTAR:+.4f}$", fontsize=11,
         color=GREEN, ha="center", va="top")
ax3.text(-0.71, -1.06,
         f"等价算法:$(x_\\sigma-E[x_0])/\\sigma$\n"
         f"$E[x_0]={EX0:+.4f}$ → $({XD}-{EX0:+.4f})/{SD_}={VSTAR:+.4f}$ ✓",
         fontsize=9.4, color=DARK, va="bottom",
         bbox=dict(fc="#eef7f1", ec="#a9d5bb"))
ax3.set_xlim(-0.75, 0.75); ax3.set_ylim(-1.18, 0.88)
ax3.set_yticks([]); ax3.set_xlabel("速度 $v$(每单位 $\\sigma$ 的位移)", fontsize=10)
ax3.set_title("(c) 加权求和 = 场的输出", fontsize=11.5)
ax3.tick_params(labelsize=9)

fig.suptitle("同一个读数、两个候选:反推 ε → 算速度 → 按 先验×似然 加权 → 相加", fontsize=13)
fig.savefig(RF + "fg_vstar_table.png", bbox_inches="tight")

print(f"读数 x_σ={XD}, σ={SD_};数据只有 {X0} 两点,先验各 {PRIOR}")
for k in range(2):
    print(f"  x0={X0[k]:+.1f}: ε=(x_σ-(1-σ)x0)/σ=({XD}-{1-SD_:.1f}*{X0[k]:+.1f})/{SD_}={EPS[k]:+.4f}"
          f" | 似然 N(ε;0,1)={LIK[k]:.4f} | 乘积={PROD[k]:.4f} | 权重={WT[k]:.4f} | v=ε-x0={VEL[k]:+.4f}")
print(f"  加权和 Σ w_k v_k     = {WT[0]:.4f}*({VEL[0]:+.2f}) + {WT[1]:.4f}*({VEL[1]:+.2f}) = {VSTAR:+.4f}")
print(f"  E[x0|x] = {EX0:+.4f} ; (x_σ-E[x0])/σ = ({XD}-{EX0:+.4f})/{SD_} = {(XD-EX0)/SD_:+.4f}")
print("wrote", RF + "fg_vstar_table.png")
