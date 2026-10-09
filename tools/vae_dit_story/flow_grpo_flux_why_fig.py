r"""通量的意义:把"密度变了"归因到"谁在搬"。

取 σ=0.5(Gaussian 数据 N(0,1) 时 V(σ)=(1−σ)²+σ² 在 σ=0.5 取极小)
→ 真值密度在这一刻【应该完全不变】:D_ODE(x) ≡ 0

(a) 三条"密度变化率"曲线:
      不补偿:+(g²/2)p''(红,中心降、尾巴升 → 会糊掉)
      补偿后:+(g²/2)p'' − (g²/2)p'' = 0(绿,与真值重合)
(b) 逐点对账:噪声贡献 vs 偏移贡献,互为反号
(c) 意义总结
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

RED, BLUE, GREEN, DARK, GREY, ORANGE = "#c0392b", "#2471a3", "#1e8449", "#2c3e50", "#95a5a6", "#b9770e"
RF = "/Users/kevin/code/lingbot-va/.claude/scratch/rf/"

V, NL = 0.5, 0.5
G2 = NL ** 2                      # g² = 0.25 → g²/2 = 0.125
p = lambda x: np.exp(-x ** 2 / (2 * V)) / np.sqrt(2 * np.pi * V)
d2p = lambda x: (x ** 2 / V ** 2 - 1 / V) * p(x)      # p''
noise = lambda x: 0.5 * G2 * d2p(x)                    # +(g²/2) p''
off = lambda x: -0.5 * G2 * d2p(x)                     # 偏移项贡献

fig = plt.figure(figsize=(15.2, 5.4), dpi=150)
xs = np.linspace(-2.4, 2.4, 700)

# ---------------- (a)
ax = fig.add_axes([0.05, 0.17, 0.31, 0.66])
ax.plot(xs, np.zeros_like(xs), color=DARK, lw=2.6, ls="--", label="真值 $\\partial_\\sigma p$(应处处为 0)")
ax.plot(xs, noise(xs), color=RED, lw=2.6, label="不补偿:$+(g^2/2)p''$")
ax.plot(xs, noise(xs) + off(xs), color=GREEN, lw=2.6, label="补偿后:$+(g^2/2)p''-(g^2/2)p''=0$")
ax.axhline(0, color=GREY, lw=0.8)
ax.annotate("中心密度会降", xy=(0, noise(0)), xytext=(-2.3, -0.165),
            fontsize=9.4, color=RED, ha="left", arrowprops=dict(arrowstyle="-|>", lw=1.2, color=RED))
ax.annotate("尾巴密度会升\n→ 分布会糊掉", xy=(1.55, noise(1.55)), xytext=(0.72, 0.085),
            fontsize=9.4, color=RED, arrowprops=dict(arrowstyle="-|>", lw=1.2, color=RED))
ax.set_xlim(-2.4, 2.4); ax.set_ylim(-0.19, 0.115)
ax.set_xlabel("$x$", fontsize=10.5); ax.set_ylabel("密度变化率 $\\partial_\\sigma p$", fontsize=10.5)
ax.set_title("(a) σ=0.5:真值密度【本该完全不变】\n不补偿却会变,补偿后才回到 0", fontsize=11.3)
ax.legend(fontsize=8.8, frameon=False, loc="upper left")
ax.tick_params(labelsize=9)

# ---------------- (b)
ax2 = fig.add_axes([0.415, 0.17, 0.27, 0.66])
xbar = np.array([-1.5, -0.8, 0.0, 0.8, 1.5])
w = 0.32
ax2.bar(xbar - w / 2, noise(xbar), width=w, color=RED, label="噪声贡献 $+(g^2/2)p''$")
ax2.bar(xbar + w / 2, off(xbar), width=w, color=GREEN, label="偏移贡献 $-(g^2/2)p''$")
ax2.axhline(0, color=GREY, lw=0.9)
for x0 in xbar:
    ax2.text(x0, max(abs(noise(x0)), abs(off(x0))) + 0.012, "和=0", fontsize=8.6,
             color=DARK, ha="center")
ax2.set_xlim(-2.1, 2.1); ax2.set_ylim(-0.20, 0.20)
ax2.set_xlabel("$x$", fontsize=10.5); ax2.set_ylabel("对密度变化的贡献", fontsize=10.5)
ax2.set_title("(b) 逐点对账:两股力量\n在每个位置都精确反号", fontsize=11.3)
ax2.legend(fontsize=8.6, frameon=False, loc="upper left")
ax2.tick_params(labelsize=9)

# ---------------- (c)
ax3 = fig.add_axes([0.72, 0.17, 0.26, 0.66]); ax3.axis("off")
ax3.text(0.5, 0.99, "(c) 那通量的意义是什么", ha="center", va="top", fontsize=11.8, color=DARK)
ax3.text(0.02, 0.88,
         "只看密度曲线:你能看到\n\"变了 / 没变\",但看不到【为什么】。\n\n"
         "写成通量:变化被拆成\n\"噪声贡献多少\"和\"偏移贡献多少\",\n"
         "于是\"抵消\"成了可以【逐点对账】的等式。\n\n"
         "而且一般分布(多峰、不对称、峰在移动)\n"
         "用\"一个方差\"描述不了,\n"
         "必须逐点跟踪整条曲线 —— 通量就是\n那个一般写法。",
         fontsize=9.2, color=DARK, va="top")
ax3.text(0.02, 0.10,
         "★ 实用意义:\n正因为能对账,才敢说\n\"改了采样方式,没改分布\"\n"
         "—— 这是把 flow 模型当 RL 策略\n优化的安全前提。",
         fontsize=9.2, color=GREEN, va="bottom", bbox=dict(fc="#eef7f1", ec="#a9d5bb"))

fig.suptitle("通量的意义:把「密度变了」翻译成「谁在搬、搬了多少」—— 这样才能验证两股力量是否抵消", fontsize=12.4, y=0.99)
fig.savefig(RF + "fg_flux_why.png", bbox_inches="tight")
print("σ=0.5, V=0.5, g²/2=0.125")
for x0 in [0.0, 0.8, 1.5]:
    print(f"  x={x0}: p={p(x0):.4f}  p''={d2p(x0):+.4f}  噪声贡献={noise(x0):+.4f}  "
          f"偏移贡献={off(x0):+.4f}  和={noise(x0)+off(x0):+.4f}")
print("wrote", RF + "fg_flux_why.png")
