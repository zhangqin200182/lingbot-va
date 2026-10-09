r"""后验均值 ≠ 后验众数。

(a) 两原子 toy:后验是"两点分布" → 众数 = +1(是数据点),均值 = +0.1974(不是数据点)
(b) 连续数据:后验是双峰曲线 → 众数在两个峰上,均值落在两峰【之间的低谷】里
(c) 为什么公式里必须是均值:L2 损失的最优解就是均值(一行推导)
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
SIG = 0.5

fig = plt.figure(figsize=(15.2, 5.6), dpi=150)

# ---------------- (a) 两原子:后验是两点分布
ax = fig.add_axes([0.055, 0.17, 0.27, 0.66])
MU = np.array([1.0, -1.0]); W = np.array([0.5987, 0.4013])
ax.bar(MU, W, width=0.22, color=[BLUE, RED], alpha=0.9)
for m, w in zip(MU, W):
    ax.text(m, w + 0.02, f"{w:.4f}", ha="center", fontsize=10.5, color=DARK)
ax.plot([MU[0]], [W[0] + 0.10], "v", ms=12, color=ORANGE)
ax.text(MU[0], W[0] + 0.13, "众数(最可能)\n$=+1$,是数据点", ha="center", fontsize=9.6, color=ORANGE, va="bottom")
mean = float(W @ MU)
ax.plot([mean], [-0.055], "*", ms=20, color=GREEN, clip_on=False, zorder=6)
ax.annotate(f"均值 $E[x_0|x_\\sigma]={mean:+.4f}$\n(不是数据点,落在两点之间)",
            xy=(mean, 0.0), xytext=(0.0, 0.30), fontsize=9.4, color=GREEN, ha="center",
            arrowprops=dict(arrowstyle="-|>", lw=1.6, color=GREEN))
ax.set_xlim(-1.6, 1.6); ax.set_ylim(0, 0.92)
ax.set_xticks([-1, 0, 1]); ax.set_xlabel("$x_0$(候选数据点)", fontsize=10.5)
ax.set_ylabel("后验权重 $p(x_0|x_\\sigma)$", fontsize=10.5)
ax.set_title("(a) 数据是离散两点时:\n众数是某个点,均值是加权重心", fontsize=11.2)
ax.tick_params(labelsize=9)

# ---------------- (b) 连续数据:众数在峰上,均值在谷里
ax2 = fig.add_axes([0.40, 0.17, 0.29, 0.66])
x0 = np.linspace(-2.2, 2.2, 900)
pdata = 0.5 * np.exp(-(x0 - 1) ** 2 / (2 * 0.22 ** 2)) / (0.22 * np.sqrt(2 * np.pi)) \
      + 0.5 * np.exp(-(x0 + 1) ** 2 / (2 * 0.22 ** 2)) / (0.22 * np.sqrt(2 * np.pi))
X = 0.1
post = pdata * np.exp(-(X - (1 - SIG) * x0) ** 2 / (2 * SIG ** 2))
post = post / np.trapz(post, x0)
ax2.plot(x0, post, color=DARK, lw=2.4)
i_mode = int(np.argmax(post))
mode = x0[i_mode]
mean = float(np.trapz(x0 * post, x0))
ax2.plot([mode], [post[i_mode]], "v", ms=13, color=ORANGE, zorder=6)
ax2.text(mode, post[i_mode] + 0.10, f"众数\n$={mode:+.2f}$", ha="center", fontsize=9.6, color=ORANGE)
ax2.plot([mean], [np.interp(mean, x0, post)], "*", ms=20, color=GREEN, zorder=6)
ax2.annotate(f"均值 $={mean:+.4f}$\n落在两个峰【之间的低谷】",
             xy=(mean, np.interp(mean, x0, post)), xytext=(mean - 0.15, post[i_mode] * 0.72),
             fontsize=9.8, color=GREEN, ha="center",
             arrowprops=dict(arrowstyle="-|>", lw=1.6, color=GREEN))
ax2.axhline(0, color=GREY, lw=0.8)
ax2.set_xlabel("$x_0$", fontsize=10.5); ax2.set_ylabel("后验密度 $p(x_0|x_\\sigma)$", fontsize=10.5)
ax2.set_title("(b) 数据连续时:均值可能落在后验低谷里", fontsize=11.2)
ax2.tick_params(labelsize=9)

# ---------------- (c) 为什么公式里必须是均值
ax3 = fig.add_axes([0.735, 0.17, 0.245, 0.66]); ax3.axis("off")
ax3.text(0.5, 0.97, "(c) 为什么必须是均值,不是众数", ha="center", va="top", fontsize=11.5, color=DARK)
ax3.text(0.02, 0.80,
         "流匹配的损失是平方误差:\n"
         "$L(a)=\\mathbb{E}_{x_0|x_\\sigma}\\!\\left[(a-x_0)^2\\right]$\n"
         "对 $a$ 求导置零:\n"
         "$2\\,\\mathbb{E}[a-x_0]=0$\n"
         "$\\Rightarrow a^*=\\mathbb{E}[x_0|x_\\sigma]$   ← 均值",
         fontsize=10, color=DARK, va="top")
ax3.text(0.02, 0.40,
         "众数 $\\arg\\max p(x_0|x_\\sigma)$ 不满足\n这个式子(它最小化的是 0-1 型损失),\n"
         "所以它不会出现在\nTweedie / $v$ 的公式里。",
         fontsize=10, color=ORANGE, va="top")
ax3.text(0.02, 0.13,
         "术语:$\\mathbb{E}[x_0|x_\\sigma]$ 在扩散文献里\n就叫\"去噪估计\" $\\hat{x}_0$。",
         fontsize=9.6, color=GREEN, va="top", bbox=dict(fc="#eef7f1", ec="#a9d5bb"))

fig.suptitle("我说错了:$E[x_0|x_\\sigma]$ 是后验【均值】,不是\"最可能来自哪个数据点\"(那是众数)", fontsize=13, y=0.995)
fig.savefig(RF + "fg_mean_vs_mode.png", bbox_inches="tight")
print(f"(a) 两原子:众数=+1(权重 {W[0]:.4f}),均值={mean:+.4f}")
print(f"(b) 连续:众数={mode:+.3f} (后验密度 {post[i_mode]:.4f}),均值={mean:+.4f} (后验密度 {np.interp(mean,x0,post):.4f})")
print("wrote", RF + "fg_mean_vs_mode.png")
