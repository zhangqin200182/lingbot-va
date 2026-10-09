r"""密度层面的相消,拆到最细。

1 维写法(去掉 ∇· 的向量干扰):
    ∂_σ p = −∂_x(f p) + (g²/2)·p''          ← Fokker–Planck
    关键恒等式:p·(log p)' = p'   ⟹   ∂_x(p·(log p)') = p''

(a) 三项曲线:p'' 与 −p'' 逐点互为镜像(噪声抹平 vs 偏移拉回)
(b) 代数链,一行一行
(c) 恒等式的数值核对(高斯)
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

P = lambda x: np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi)         # p = N(0,1)
D1 = lambda x: -x * P(x)                                       # p'
D2 = lambda x: (x ** 2 - 1) * P(x)                             # p''
LOGP1 = lambda x: -x                                           # (log p)'
PLOGP1 = lambda x: P(x) * LOGP1(x)                             # p*(log p)'

fig = plt.figure(figsize=(15.0, 5.3), dpi=150)

# ---------------- (a)
ax = fig.add_axes([0.055, 0.16, 0.315, 0.68])
xs = np.linspace(-3.6, 3.6, 800)
ax.plot(xs, P(xs), color=DARK, lw=2.6, label="$p_\\sigma(x)$")
ax.plot(xs, D2(xs) * 0.5, color=RED, lw=2.2, label="噪声项 $(g^2/2)\\,p''$(抹平)")
ax.plot(xs, -D2(xs) * 0.5, color=GREEN, lw=2.2, label="偏移项 $-(g^2/2)\\,p''$(拉回)")
ax.axhline(0, color=GREY, lw=0.8)
for x0 in [-1.0, 0.0, 1.0]:
    ax.plot([x0, x0], [0, D2(x0) * 0.5], color=RED, lw=1.0, ls=":", alpha=0.8)
ax.annotate("峰处 $p''<0$:\n噪声把峰压下去", xy=(0, D2(0) * 0.5), xytext=(-3.4, 0.30),
            fontsize=9.2, color=RED, arrowprops=dict(arrowstyle="-|>", lw=1.2, color=RED))
ax.annotate("两侧 $p''>0$:\n噪声把谷填起来", xy=(2.0, D2(2.0) * 0.5), xytext=(1.1, 0.30),
            fontsize=9.2, color=RED, arrowprops=dict(arrowstyle="-|>", lw=1.2, color=RED))
ax.text(2.1, -0.30, "两条曲线【逐点互为反号】\n加起来处处为 0", fontsize=9.4, color=GREEN)
ax.set_xlim(-3.6, 3.6); ax.set_ylim(-0.42, 0.46)
ax.set_xlabel("$x$", fontsize=10.5); ax.set_ylabel("密度 / 变化项", fontsize=10.5)
ax.set_title("(a) 噪声抹平 = 把峰压低、把谷填高\n偏移项正好反过来,逐点抵消", fontsize=11.5)
ax.legend(fontsize=8.8, frameon=False, loc="lower right")
ax.tick_params(labelsize=9)

# ---------------- (b)
ax2 = fig.add_axes([0.415, 0.16, 0.30, 0.68]); ax2.axis("off")
ax2.text(0.5, 0.99, "(b) 一行一行推", ha="center", va="top", fontsize=12, color=DARK)
chain = [
    ("① Fokker–Planck(1 维)", "$\\partial_\\sigma p=-\\partial_x(fp)+\\frac{g^2}{2}p''$", DARK),
    ("② 代入漂移", "$f=v^*+\\frac{g^2}{2}(\\log p)'$", DARK),
    ("③ 乘上 $p$", "$fp=v^*p+\\frac{g^2}{2}\\,p(\\log p)'$", BLUE),
    ("④ 关键恒等式", "$p(\\log p)'=p\\cdot\\frac{p'}{p}=p'$", RED),
    ("⑤ 求散度", "$\\partial_x(p(\\log p)')=\\partial_x p'=p''$", RED),
    ("⑥ 代回 ①", "$\\partial_\\sigma p=-\\partial_x(v^*p)-\\frac{g^2}{2}p''+\\frac{g^2}{2}p''$", GREEN),
    ("⑦ 两项相消", "$\\partial_\\sigma p=-\\partial_x(v^*p)$", GREEN),
]
y = 0.90
for name, expr, col in chain:
    ax2.text(0.0, y, name, fontsize=9.6, color=col, va="top")
    ax2.text(0.06, y - 0.055, expr, fontsize=10.4, color=col, va="top")
    y -= 0.112
ax2.text(0.0, 0.015,
         "⑦ 与 ODE 的密度方程【完全相同】\n⇒ $p_\\sigma$ 在每个时刻都不变。",
         fontsize=9.8, color=GREEN, va="bottom", bbox=dict(fc="#eef7f1", ec="#a9d5bb"))

# ---------------- (c)
ax3 = fig.add_axes([0.755, 0.16, 0.225, 0.68]); ax3.axis("off")
ax3.text(0.5, 0.99, "(c) 恒等式数值核对", ha="center", va="top", fontsize=11.5, color=DARK)
rows = [["$x$", "$p(\\log p)'$", "$p'$"]]
for x0 in [-1.5, -0.5, 0.0, 0.8, 2.0]:
    rows.append([f"{x0:+.1f}", f"{PLOGP1(x0):+.5f}", f"{D1(x0):+.5f}"])
tb = ax3.table(cellText=rows, cellLoc="center", bbox=[0.0, 0.60, 1.0, 0.35])
tb.auto_set_font_size(False); tb.set_fontsize(8.8)
for (r, c), cell in tb.get_celld().items():
    if r == 0:
        cell.set_facecolor("#eef2f7")
rows2 = [["$x$", "$\\partial_x[p(\\log p)']$", "$p''$"]]
for x0 in [-1.5, -0.5, 0.0, 0.8, 2.0]:
    g = (PLOGP1(x0 + 1e-5) - PLOGP1(x0 - 1e-5)) / 2e-5
    rows2.append([f"{x0:+.1f}", f"{g:+.5f}", f"{D2(x0):+.5f}"])
tb2 = ax3.table(cellText=rows2, cellLoc="center", bbox=[0.0, 0.16, 1.0, 0.35])
tb2.auto_set_font_size(False); tb2.set_fontsize(8.8)
for (r, c), cell in tb2.get_celld().items():
    if r == 0:
        cell.set_facecolor("#fff4e5")
ax3.text(0.0, 0.10,
         "两列在每一行都相等 ⇒ 恒等式\n$p(\\log p)'=p'$ 与 $\\partial_x(p(\\log p)')=p''$ 成立。",
         fontsize=9.0, color=GREEN, va="top", bbox=dict(fc="#eef7f1", ec="#a9d5bb"))

fig.suptitle("密度层面的相消,拆到最细:关键只有一行 —— $p\\times(\\log p)'=p'$", fontsize=12.8, y=0.99)
fig.savefig(RF + "fg_fp_cancellation.png", bbox_inches="tight")

print("恒等式核对(p = N(0,1)):")
for x0 in [-1.5, -0.5, 0.0, 0.8, 2.0]:
    g = (PLOGP1(x0 + 1e-5) - PLOGP1(x0 - 1e-5)) / 2e-5
    print(f"  x={x0:+.1f}: p(logp)'={PLOGP1(x0):+.5f} vs p'={D1(x0):+.5f} | "
          f"∂x[p(logp)']={g:+.5f} vs p''={D2(x0):+.5f}")
print("wrote", RF + "fg_fp_cancellation.png")
