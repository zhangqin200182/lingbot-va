r"""漂移是什么意思:高斯中心沿 v_θ 走一步?—— 基本是,但有三点要补。

一步(x' = mean + s·ξ):
  mean = x·(1 + g²dσ/(2σ)) + v_θ·dσ·(1 + g²(1−σ)/(2σ))
  ⇒ 中心位移 = dσ·[ (g²/(2σ))·x  + (1 + nl²/2)·v_θ ]
                 └ 补偿:往高密度收 ┘   └ 沿 v_θ 走(生成方向是 −v_θ)┘
  ⇒ 同时加宽 s = √(g²|dσ|)  ⇒ 一步 = 【平移】+【加宽】+【随机 ξ】

具体数(σ=0.7, nl=0.5, T=19):x 系数 −0.0219,v_θ 系数 −0.0592,宽度 s = 0.1752
  ⇒ 单步平移(0.05 量级)≪ 单步抖动(0.175)⇒ 单条轨迹看着抖,但平移是【系统性】的
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

# 数值(σ=0.7, nl=0.5, T=19)
sigma, nl, T = 0.7, 0.5, 19
dt = -1.0 / T
g2 = sigma / (1 - sigma) * nl ** 2
c_x = g2 * dt / (2 * sigma)
c_v = dt * (1 + nl ** 2 / 2)
s_noise = np.sqrt(g2 * abs(dt))
X0, V0 = 1.0, 0.5
d_comp, d_v = c_x * X0, c_v * V0
XNEW = X0 + d_comp + d_v

fig = plt.figure(figsize=(15.2, 5.6), dpi=150)

# ---------------- (a) 一步的几何(按比例)
ax = fig.add_axes([0.05, 0.17, 0.32, 0.66])
xs = np.linspace(X0 - 0.75, X0 + 0.75, 800)
ax.plot(xs, gauss(xs, X0, s_noise), color=GREY, lw=2.0, ls="--", label="这一步的抖动宽度 $s$")
ax.plot(xs, gauss(xs, XNEW, s_noise), color=GREEN, lw=2.6, label="下一步的中心(平移后)")
ax.plot([X0], [gauss(X0, X0, s_noise)], "o", ms=10, color=DARK, zorder=6)
ax.text(X0, gauss(X0, X0, s_noise) + 0.25, "当前 $x$", fontsize=9.4, color=DARK, ha="center")
ax.plot([XNEW], [gauss(XNEW, XNEW, s_noise)], "o", ms=10, color=GREEN, zorder=6)
ax.text(XNEW, gauss(XNEW, XNEW, s_noise) - 0.35, "新中心 $x+f\\,d\\sigma$", fontsize=9.4,
        color=GREEN, ha="center")
# 位移分解:单独用放大插图(位移比宽度小 ~3 倍,主图里画不下)
axi = ax.inset_axes([0.42, 0.30, 0.56, 0.38])
K = 14.0                      # 放大倍数(仅用于展示)
axi.plot([0], [0], "o", ms=8, color=DARK, zorder=6)
axi.text(0, 0.35, "起点 $x$", fontsize=8.4, color=DARK, ha="center")
axi.add_patch(FancyArrowPatch((0, 0), (d_comp * K, 0), arrowstyle="-|>",
                              mutation_scale=11, lw=2.4, color=ORANGE))
axi.text(d_comp * K / 2, -0.42, "补偿", fontsize=8.2, color=ORANGE, ha="center")
axi.add_patch(FancyArrowPatch((d_comp * K, 0), ((d_comp + d_v) * K, 0), arrowstyle="-|>",
                              mutation_scale=11, lw=2.4, color=BLUE))
axi.text(d_comp * K + d_v * K / 2, 0.38, "$v_\\theta$ 项", fontsize=8.2, color=BLUE, ha="center")
axi.set_xlim(-0.45, 1.15); axi.set_ylim(-0.85, 0.85)
axi.set_yticks([]); axi.tick_params(labelsize=7.4)
axi.set_title("位移分解(放大 %.0f×)" % K, fontsize=8.6, color=GREY)
for sp in ("top", "right", "left"):
    axi.spines[sp].set_visible(False)
ax.text(X0 - 0.72, 0.72, "半透明虚线 = 抖动宽度 $s$;两处几乎重合\n⇒ 单步平移是宽度的 %.0f%%"
        % (abs(d_comp + d_v) / s_noise * 100), fontsize=8.6, color=GREY, va="top")
ax.set_xlim(X0 - 0.78, X0 + 0.78); ax.set_ylim(-0.6, 3.0)
ax.set_xlabel("$x$(1 维示意,按真实比例)", fontsize=10.5)
ax.set_yticks([])
ax.set_title("(a) 一步 = 【平移】+【加宽】\n中心沿 $v_\\theta$ 走,补偿项再收一点", fontsize=11.3)
ax.legend(fontsize=8.4, frameon=False, loc="upper left")
ax.tick_params(labelsize=9)

# ---------------- (b) 系数表
ax2 = fig.add_axes([0.42, 0.17, 0.30, 0.66]); ax2.axis("off")
ax2.text(0.5, 1.00, "(b) 两个系数与噪声宽度(nl=0.5, T=19)", ha="center", va="top",
         fontsize=11.2, color=DARK)
rows = [["$\\sigma$", "$x$ 系数", "$v_\\theta$ 系数", "宽度 $s$"],
        ["0.3", "−0.0094", "−0.0592", "0.075"],
        ["0.5", "−0.0132", "−0.0592", "0.115"],
        ["0.7", "−0.0219", "−0.0592", "0.175"],
        ["0.9", "−0.0658", "−0.0592", "0.344"]]
tb = ax2.table(cellText=rows, cellLoc="center", bbox=[0.0, 0.52, 1.0, 0.40])
tb.auto_set_font_size(False); tb.set_fontsize(8.8)
for (r, c), cell in tb.get_celld().items():
    if r == 0:
        cell.set_facecolor("#eef2f7")
    if c == 2 and r > 0:
        cell.set_facecolor("#eaf2fa")
    if c == 1 and r > 0:
        cell.set_facecolor("#fdf3e7")
ax2.text(0.0, 0.46,
         "读法:\n"
         "· $v_\\theta$ 的系数 $=d\\sigma(1+nl^2/2)$:与 $\\sigma$ 无关\n"
         "· $x$ 的系数 $=g^2d\\sigma/(2\\sigma)$:随 $\\sigma\\to1$ 变大\n"
         "　(靠噪声端,补偿更用力)\n"
         "· 宽度 $s=\\sqrt{g^2|d\\sigma|}$:靠噪声端更大\n\n"
         "举例 $x=1,v_\\theta=0.5$:平移 −0.0515,宽度 0.175",
         fontsize=8.8, color=DARK, va="top",
         bbox=dict(fc="#fffbe6", ec="#e0c060"))

# ---------------- (c) 三点精确化
ax3 = fig.add_axes([0.75, 0.17, 0.24, 0.66]); ax3.axis("off")
ax3.text(0.5, 0.99, "(c) 三点要补精确", ha="center", va="top", fontsize=11.4, color=DARK)
ax3.text(0.0, 0.90,
         "① 不只是 $v_\\theta$:\n"
         "　位移 = $v_\\theta$ 项 + 补偿项\n"
         "　$d\\sigma](1+\\frac{nl^2}{2})v_\\theta+\\frac{g^2}{2\\sigma}x]$\n\n"
         "② 移动的是\"这一步落点分布的\n　中心\",不是\"粒子沿 $v_\\theta$ 直线飞\"\n"
         "　(严格说 SDE 描述粒子;\n　 每步高斯近似下才有\"中心\")\n\n"
         "③ 同时【加宽】$s$:\n"
         "　所以一步是\"平移 + 加宽\",\n"
         "　不是纯平移",
         fontsize=8.6, color=DARK, va="top")
ax3.text(0.0, 0.05,
         "生成方向(dσ<0):\n"
         "$-v_\\theta$ 指向数据;\n"
         "补偿项沿 $\\nabla\\log p$\n"
         "= 往高密度收(抵消加宽)。",
         fontsize=8.6, color=GREEN, va="bottom", bbox=dict(fc="#eef7f1", ec="#a9d5bb"))

fig.suptitle("漂移 = 速度;乘上步长 $d\\sigma$ 就是\"中心移动的位移向量\" —— 但位移里还有补偿项,且同时加宽",
             fontsize=11.8, y=0.99)
fig.savefig(RF + "fg_drift_meaning.png", bbox_inches="tight")
print(f"σ={sigma}, nl={nl}, T={T}: x 系数 {c_x:+.5f}, v 系数 {c_v:+.5f}, 宽度 {s_noise:.4f}")
print(f"  x={X0}, v_θ={V0} → 补偿位移 {d_comp:+.4f}, v_θ 位移 {d_v:+.4f}, 总平移 {d_comp+d_v:+.4f}")
print("wrote", RF + "fg_drift_meaning.png")
