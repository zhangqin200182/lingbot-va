r"""澄清:p 就是概率密度。

(a) 概率密度的含义:面积 = 概率质量;N 个样本落在条带里的个数 ≈ N·p·dx(这就是"数格子")
(b) 这一路上出现过的所有 p,各是什么
(c) 两个视角的桥:∫h(x)p(x)dx = E[h(x)] —— Step 3 的证明就是这座桥
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

fig = plt.figure(figsize=(15.2, 5.6), dpi=150)

# ---------------- (a) 面积 = 概率质量
ax = fig.add_axes([0.05, 0.18, 0.30, 0.66])
S = 0.5
V = (1 - S) ** 2 + S ** 2
P = lambda x: np.exp(-x ** 2 / (2 * V)) / np.sqrt(2 * np.pi * V)
xs = np.linspace(-2.6, 2.6, 700)
ax.plot(xs, P(xs), color=DARK, lw=2.6)
a, b = -0.05, 0.05
xg = np.linspace(a, b, 100)
ax.fill_between(xg, 0, P(xg), color=BLUE, alpha=0.35)
ax.annotate(f"面积 $=\\int_{a}^{b}p_\\sigma(x)dx$\n$\\approx {P(0.0)*(b-a):.4f}$(概率质量)",
            xy=(0.0, P(0.0) * 0.6), xytext=(-2.5, 0.95),
            fontsize=9.6, color=BLUE, arrowprops=dict(arrowstyle="-|>", lw=1.4, color=BLUE))
rng = np.random.default_rng(0)
smp = rng.normal(scale=np.sqrt(V), size=10000)
inside = smp[(smp >= a) & (smp <= b)]
ax.plot(inside[:60], -0.028 * np.ones(min(60, len(inside))), "|", ms=7, color=RED, alpha=0.85)
ax.plot(smp[:400], -0.055 * np.ones(400), "|", ms=5, color=GREY, alpha=0.5)
ax.text(0.0, -0.085, f"10000 个样本里有 {len(inside)} 个落进条带\n(≈ $10000\\times$面积 $=564$)",
        fontsize=9.0, color=RED, ha="left", va="top")
ax.set_ylim(-0.20, 1.10)
ax.set_xlabel("$x$", fontsize=10.5); ax.set_ylabel("$p_\\sigma(x)$", fontsize=10.5)
ax.set_title("(a) 概率密度:高度是密度,面积才是概率\n$N$ 个样本的个数 $\\approx N\\cdot p\\cdot dx$", fontsize=11.3)
ax.tick_params(labelsize=9)

# ---------------- (b) 所有 p 的清单
ax2 = fig.add_axes([0.40, 0.18, 0.30, 0.66]); ax2.axis("off")
ax2.text(0.5, 0.99, "(b) 这一路上出现过的 p", ha="center", va="top", fontsize=11.8, color=DARK)
rows = [["记号", "是什么(对谁取)"],
        ["$p_{data}(x_0)$", "数据的分布(干净样本)"],
        ["$p_\\sigma(x)$", "噪声水平 $\\sigma$ 处的边缘密度\n【= FP 方程里的 p】"],
        ["$p(x_0,x_\\sigma{=}x)$", "联合密度(先验×似然)"],
        ["$p(x_0\\,|\\,x_\\sigma{=}x)$", "后验:\"哪个数据点\""],
        ["$\\pi_\\theta(x'|x)$", "SDE 每一步的条件高斯(RL 的策略)"]]
tb = ax2.table(cellText=rows, cellLoc="left", bbox=[0.0, 0.34, 1.0, 0.62])
tb.auto_set_font_size(False); tb.set_fontsize(9.0)
for (r, c), cell in tb.get_celld().items():
    cell.set_edgecolor("#d5dbe2")
    if r == 0:
        cell.set_facecolor("#eef2f7"); cell.set_text_props(weight="bold")
    if r == 2:
        cell.set_facecolor("#eef7f1")
ax2.text(0.0, 0.28,
         "它们不是同一个函数,但由同一套东西连起来:\n"
         "$p_\\sigma(x)=\\int p_{data}(x_0)\\,N(x;(1-\\sigma)x_0,\\sigma^2)dx_0$",
         fontsize=9.2, color=DARK, va="top",
         bbox=dict(fc="#fffbe6", ec="#e0c060"))
ax2.text(0.0, 0.03,
         "FP 方程里的 $p$ = $p_\\sigma(x)$,\n就是你在图里看到的那一族钟形曲线。",
         fontsize=9.6, color=GREEN, va="bottom", bbox=dict(fc="#eef7f1", ec="#a9d5bb"))

# ---------------- (c) 两个视角的桥
ax3 = fig.add_axes([0.725, 0.18, 0.255, 0.66]); ax3.axis("off")
ax3.text(0.5, 0.99, "(c) 两个视角的桥", ha="center", va="top", fontsize=11.5, color=DARK)
ax3.text(0.02, 0.86,
         "粒子视角说:$N$ 个粒子各自走 $dx=f\\,d\\sigma$\n"
         "密度视角说:密度按 $\\partial_\\sigma p=-\\partial_x(fp)$ 演化\n\n"
         "两者怎么对应?用任意函数 $h$ 当探针:\n"
         "$\\int h(x)\\,p(x)\\,dx=\\mathbb{E}[h(x)]$\n"
         "左边是密度视角,右边是粒子视角\n—— 同一个量的两种写法。",
         fontsize=9.2, color=DARK, va="top")
ax3.text(0.02, 0.34,
         "Step 3 证的就是这件事:\n"
         "对【任意】$h$,两个视角给出的\n期望变化率都是 $\\mathbb{E}[\\nabla h\\cdot v^*]$,\n"
         "起点也一样 → 分布处处相同。\n\n"
         "所以 Step 3 的证明\n= \"粒子描述\"与\"密度描述\"对齐的证明。",
         fontsize=9.2, color=GREEN, va="top", bbox=dict(fc="#eef7f1", ec="#a9d5bb"))

fig.suptitle("澄清:这里的 $p$ 就是概率密度 —— 而且是前面一直在画的 $p_\\sigma(x)$", fontsize=12.8, y=0.99)
fig.savefig(RF + "fg_p_is_probability.png", bbox_inches="tight")

print(f"σ={S}: V={V}, p_σ(0)={P(0.0):.4f}")
print(f"条带 [{-0.05},{0.05}] 的面积 ≈ {P(0.0)*0.1:.4f} → 10000 个样本里期望 {P(0.0)*0.1*10000:.0f} 个")
print(f"实际落进条带的样本数:{len(inside)}")
print("wrote", RF + "fg_p_is_probability.png")
