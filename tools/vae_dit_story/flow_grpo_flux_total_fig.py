r"""用户的理解验证:加噪声后"总通量没变",所以密度变化没变。

σ=0.5(高斯数据 V=0.5,此时 v*≡0,ODE 的密度本该完全不变)
  漂移通量(含偏移) = f·p = v*p + (g²/2)p'
  扩散通量(来自噪声) = −(g²/2)p'
  ────────────────────────────
  总通量 = v*p + (g²/2)p' − (g²/2)p' = v*p  ← 与 ODE 完全相同!

(a) 三条通量曲线:两股各自都很大,但处处反号,和恒为 0
(b) 从"总通量"到"密度变化":∂_σp = −∂_x F_total = 0
(c) 把用户的话改精确
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
G2H = 0.5 * NL ** 2                       # g²/2 = 0.125
p = lambda x: np.exp(-x ** 2 / (2 * V)) / np.sqrt(2 * np.pi * V)
dp = lambda x: -x / V * p(x)
vstar = lambda x: 0.0 * x                 # σ=0.5 时 v*≡0(高斯数据)
F_ode = lambda x: vstar(x) * p(x)         # ODE 的通量
F_drift = lambda x: vstar(x) * p(x) + G2H * dp(x)     # 含偏移的漂移通量
F_diff = lambda x: -G2H * dp(x)                       # 噪声带来的扩散通量
F_tot = lambda x: F_drift(x) + F_diff(x)

fig = plt.figure(figsize=(15.2, 5.4), dpi=150)
xs = np.linspace(-2.2, 2.2, 700)

# ---------------- (a)
ax = fig.add_axes([0.05, 0.17, 0.31, 0.66])
ax.plot(xs, F_ode(xs), color=DARK, lw=3.0, ls="--", label="ODE 的总通量 $v^*p\\equiv0$")
ax.plot(xs, F_drift(xs), color=RED, lw=2.4, label="漂移通量(含偏移)$=+(g^2/2)p'$")
ax.plot(xs, F_diff(xs), color=BLUE, lw=2.4, label="扩散通量(来自噪声)$=-(g^2/2)p'$")
ax.plot(xs, F_tot(xs), color=GREEN, lw=3.2, label="两股之和 $F_{\\rm total}\\equiv0$")
ax.axhline(0, color=GREY, lw=0.8)
ax.annotate("两股各自都很大\n(在 $|x|\\approx0.7$ 处最大)", xy=(0.7, F_drift(0.7)),
            xytext=(-2.1, 0.075), fontsize=9.2, color=DARK,
            arrowprops=dict(arrowstyle="-|>", lw=1.2, color=DARK))
ax.set_xlim(-2.2, 2.2); ax.set_ylim(-0.095, 0.115)
ax.set_xlabel("$x$", fontsize=10.5); ax.set_ylabel("通量 $F$", fontsize=10.5)
ax.set_title("(a) σ=0.5:两股通量【各自都不为零】\n但处处精确反号 → 总和恒为 0", fontsize=11.3)
ax.legend(fontsize=8.6, frameon=False, loc="lower right")
ax.tick_params(labelsize=9)

# ---------------- (b)
ax2 = fig.add_axes([0.415, 0.17, 0.29, 0.66])
ax2.plot(xs, F_tot(xs), color=GREEN, lw=2.6, label="总通量 $F_{\\rm total}$")
ax2.plot(xs, -np.gradient(F_tot(xs), xs), color="#7d3c98", lw=2.6,
         label="密度变化 $-\\partial_x F_{\\rm total}$")
ax2.axhline(0, color=GREY, lw=0.8)
ax2.text(0.0, 0.055, "两步:\n① 两股通量相加 → 得到总通量\n② 对总通量求位置导数、取负\n　 → 就是密度的变化率",
         fontsize=9.4, color=DARK, ha="center", va="bottom",
         bbox=dict(fc="#fffbe6", ec="#e0c060"))
ax2.text(0.0, -0.055, "这里两条曲线都是 0\n⇒ 密度不变 ⇒ 与 ODE 完全一致", fontsize=9.8, color=GREEN,
         ha="center", va="top")
ax2.set_xlim(-2.2, 2.2); ax2.set_ylim(-0.095, 0.115)
ax2.set_xlabel("$x$", fontsize=10.5); ax2.set_ylabel("通量 / 变化率", fontsize=10.5)
ax2.set_title("(b) 从通量到密度变化:\n$\\partial_\\sigma p=-\\partial_x F_{\\rm total}$", fontsize=11.3)
ax2.legend(fontsize=9, frameon=False, loc="lower left")
ax2.tick_params(labelsize=9)

# ---------------- (c)
ax3 = fig.add_axes([0.735, 0.17, 0.245, 0.66]); ax3.axis("off")
ax3.text(0.5, 0.99, "(c) 你的说法,改精确一点", ha="center", va="top", fontsize=11.6, color=DARK)
ax3.text(0.02, 0.87,
         "你说的:\n\"粒子方向变了,但总通量没变,\n所以密度变化没变\"\n\n"
         "【对】结论对,因果位置也很准:\n"
         "通量是【原因】,密度变化是【结果】。\n\n"
         "【改精确】不是\"方向变了但通量没变\",\n而是:\n"
         "【两股通量都变了】,\n但它们之和没变:\n"
         "　漂移(有方向)多了 $+(g^2/2)p'$\n"
         "　扩散(随机晃)带来 $-(g^2/2)p'$",
         fontsize=9.0, color=DARK, va="top")
ax3.text(0.02, 0.10,
         "更准确的一句话:\n个体的【路径】变随机了,\n但整体的【总通量】没变\n→ 分布没变。",
         fontsize=9.4, color=GREEN, va="bottom", bbox=dict(fc="#eef7f1", ec="#a9d5bb"))

fig.suptitle("验证你的理解:通量层面看,两股【各自都不是零】,但加起来与 ODE 的总通量完全相同", fontsize=12.4, y=0.99)
fig.savefig(RF + "fg_flux_total.png", bbox_inches="tight")
print("σ=0.5, v*≡0, g²/2=0.125")
for x0 in [0.0, 0.7, 1.0, 1.6]:
    print(f"  x={x0}: ODE通量={F_ode(x0):+.4f}  漂移通量={F_drift(x0):+.4f}  "
          f"扩散通量={F_diff(x0):+.4f}  总和={F_tot(x0):+.4f}")
print("wrote", RF + "fg_flux_total.png")
