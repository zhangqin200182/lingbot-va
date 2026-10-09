r"""那两个积分是什么:全概率(消掉 x₀)与贝叶斯(条件期望)。

(a) 每个候选 x₀ 按先验撒一个高斯,叠加起来就是 p_σ(x);标出 x=0.1 处两个来源的贡献
(b) 联合贡献 (0.2897, 0.1942) ÷ p_σ(x)=0.4839 → 后验权重 (0.5987, 0.4013)  ← 这一步就是贝叶斯
(c) 用后验权重求期望 = 分子/分母,与前面表里的 E[x₀]=+0.1974 完全一致
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
from matplotlib.patches import Rectangle

RED, BLUE, GREEN, DARK, GREY, ORANGE = "#c0392b", "#2471a3", "#1e8449", "#2c3e50", "#95a5a6", "#b9770e"
RF = "/Users/kevin/code/lingbot-va/.claude/scratch/rf/"

MU = np.array([1.0, -1.0]); PR = np.array([0.5, 0.5]); SIG = 0.5
X = 0.1
SS = SIG ** 2
M = (1 - SIG) * MU


def gauss(x, m, s2):
    return np.exp(-(x - m) ** 2 / (2 * s2)) / np.sqrt(2 * np.pi * s2)


N_ = gauss(X, M, SS)                 # 两个成分在 x 处的密度
JOINT = PR * N_                      # 联合贡献(先验 × 似然)
PSIG = JOINT.sum()                   # 边缘密度 p_σ(x)
WT = JOINT / PSIG                    # 后验权重
EX0 = float(WT @ MU)

fig = plt.figure(figsize=(15.2, 6.3), dpi=150)

# ---------------- (a)
ax = fig.add_axes([0.05, 0.17, 0.30, 0.62])
xg = np.linspace(-2.0, 2.0, 700)
for k, (m, c) in enumerate(zip(M, [BLUE, RED])):
    ax.plot(xg, PR[k] * gauss(xg, m, SS), color=c, lw=1.8, ls="--", alpha=0.9,
            label=f"候选 $x_0={MU[k]:+.0f}$ 的贡献\n$p(x_0)\\cdot N(x;(1-\\sigma)x_0,\\sigma^2)$")
ax.plot(xg, PR[0] * gauss(xg, M[0], SS) + PR[1] * gauss(xg, M[1], SS),
        color=DARK, lw=2.6, label="两者相加 $=p_\\sigma(x)$")
ax.axvline(X, color=GREY, ls=":", lw=1.1)
for k, c in enumerate([BLUE, RED]):
    ax.add_patch(Rectangle((X, 0), 0.035, JOINT[k], fc=c, ec="none", alpha=0.55))
    ax.plot([X + 0.017], [JOINT[k]], "o", ms=8, color=c, zorder=6)
ax.annotate(f"$p_\\sigma(0.1)={PSIG:.4f}$", xy=(X, PSIG), xytext=(0.62, 0.62),
            fontsize=10, color=DARK, arrowprops=dict(arrowstyle="-|>", lw=1.4, color=DARK))
ax.text(0.42, 0.30, f"两段贡献:\n{JOINT[0]:.4f} 与 {JOINT[1]:.4f}\n相加 = {PSIG:.4f}", fontsize=9.4,
        color=DARK, bbox=dict(fc="#fffbe6", ec="#e0c060"))
ax.set_xlabel("$x$", fontsize=10.5); ax.set_ylabel("密度贡献", fontsize=10.5)
ax.set_title("(a) 第一个积分 = 全概率:\n把 $x_0$ 这个维度【加掉】→ 得到 $p_\\sigma(x)$", fontsize=11.2)
ax.legend(fontsize=7.8, frameon=False, loc="upper left")
ax.tick_params(labelsize=9)

# ---------------- (b)
ax2 = fig.add_axes([0.42, 0.17, 0.245, 0.62])
ax2.bar([0], [JOINT[0]], color=BLUE, width=0.5, label="来自 $x_0=+1$")
ax2.bar([0], [JOINT[1]], bottom=[JOINT[0]], color=RED, width=0.5, label="来自 $x_0=-1$")
ax2.text(0, JOINT[0] / 2, f"{JOINT[0]:.4f}", ha="center", va="center", fontsize=10, color="w")
ax2.text(0, JOINT[0] + JOINT[1] / 2, f"{JOINT[1]:.4f}", ha="center", va="center", fontsize=10, color="w")
ax2.text(0.06, PSIG + 0.045, f"联合 = {PSIG:.4f}\n$=p_\\sigma(x)$", ha="center", fontsize=9.6, color=DARK)
ax2.bar([0.9], [WT[0]], color=BLUE, width=0.5)
ax2.bar([0.9], [WT[1]], bottom=[WT[0]], color=RED, width=0.5)
ax2.text(0.9, WT[0] / 2, f"{WT[0]:.4f}", ha="center", va="center", fontsize=10, color="w")
ax2.text(0.9, WT[0] + WT[1] / 2, f"{WT[1]:.4f}", ha="center", va="center", fontsize=10, color="w")
ax2.text(0.9, 1.02, "归一化 = 1\n$=$ 后验权重", ha="center", fontsize=9.6, color=DARK)
ax2.annotate("", xy=(0.62, 0.5), xytext=(0.28, 0.5), arrowprops=dict(arrowstyle="-|>", lw=2, color=GREEN))
ax2.text(0.45, 0.55, "$\\div\\,p_\\sigma(x)$", ha="center", fontsize=10.5, color=GREEN)
ax2.set_xticks([0, 0.9]); ax2.set_xticklabels(["联合\n(先验×似然)", "条件\n$p(x_0|x_\\sigma)$"], fontsize=9.4)
ax2.set_ylim(0, 1.20); ax2.set_yticks([])
ax2.set_title("(b) 除以 $p_\\sigma(x)$ = 贝叶斯:\n$p(x_0|x)=\\dfrac{p(x_0)\\,N(x;(1-\\sigma)x_0,\\sigma^2)}{p_\\sigma(x)}$",
              fontsize=11.2)
ax2.legend(fontsize=8.4, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.02))

# ---------------- (c)
ax3 = fig.add_axes([0.715, 0.17, 0.265, 0.62]); ax3.axis("off")
ax3.text(0.5, 0.98, "(c) 期望 = 分子 ÷ 分母", ha="center", va="top", fontsize=11.5, color=DARK)
ax3.text(0.02, 0.80,
         f"分子 $\\int x_0\\,p(x_0)N\\,dx_0$\n"
         f"$= {JOINT[0]:.4f}(+1)+{JOINT[1]:.4f}(-1) = {JOINT[0]-JOINT[1]:+.4f}$",
         fontsize=9.8, color=DARK, va="top")
ax3.text(0.02, 0.60,
         f"分母 $\\int p(x_0)N\\,dx_0 = p_\\sigma(x) = {PSIG:.4f}$",
         fontsize=9.8, color=DARK, va="top")
ax3.text(0.02, 0.43,
         f"相除 $E[x_0|x_\\sigma] = {JOINT[0]-JOINT[1]:+.4f} / {PSIG:.4f} = {EX0:+.4f}$",
         fontsize=10.4, color=GREEN, va="top")
ax3.text(0.02, 0.26,
         f"等价写法:$\\sum_k w_k\\mu_k = {WT[0]:.4f}(+1)+{WT[1]:.4f}(-1) = {EX0:+.4f}$",
         fontsize=9.6, color=DARK, va="top")
ax3.text(0.02, 0.08,
         "与前面那张表的 $E[x_0]=+0.1974$ 完全一致 ✓\n权重也一致:$0.5987/0.4013$ ✓",
         fontsize=9.6, color=GREEN, va="top",
         bbox=dict(fc="#eef7f1", ec="#a9d5bb"))

fig.suptitle("那两个积分是什么:全概率(消掉 $x_0$)与贝叶斯(条件期望)", fontsize=13, y=0.99)
fig.savefig(RF + "fg_marginal_bayes.png", bbox_inches="tight")

print(f"σ={SIG}, x={X};成分均值 (1−σ)μ = {M}")
print(f"  两个成分密度 N(x;m,σ²) = {N_[0]:.4f}, {N_[1]:.4f}")
print(f"  联合(先验×似然)     = {JOINT[0]:.4f}, {JOINT[1]:.4f}   相加 = {PSIG:.4f} = p_σ(x)")
print(f"  后验权重             = {WT[0]:.4f}, {WT[1]:.4f}")
print(f"  分子 = Σ 联合·μ      = {JOINT[0]-JOINT[1]:+.4f}")
print(f"  期望 = 分子/分母     = {(JOINT[0]-JOINT[1])/PSIG:+.4f}")
print(f"  Σ w_k μ_k            = {EX0:+.4f}   (应相同)")
print("wrote", RF + "fg_marginal_bayes.png")
