r"""为什么加噪声就"产生了概率"?

核心机制(一维就看得清):
  确定性:x' = m_θ(x)                    → 输出是一个【点】→ 密度是 δ,不可用
  加噪声:x' = m_θ(x) + s·ξ ,  ξ~N(0,1)
        ⇕ 反过来解出"是哪个噪声造成了这个落点"
          ξ = (x' − m_θ(x))/s
        ⇒ π_θ(x'|x) = p_ξ( (x'−m_θ)/s ) / s = N(x'; m_θ, s²)

  "落在 x' 的概率密度" = "噪声取到 (x'−m)/s 这个值的概率密度"
  靠近中心 → 只需很小的噪声 → 落点多;远离中心 → 需要很大的噪声 → 落点少

  归一化自动成立:∫π dx' = ∫p_ξ dξ = 1(变量替换,雅可比 1/s 正好抵消)
  可导:θ 只出现在 m_θ 里 ⇒ ∇_θ log π = (ξ/s)·∂m/∂θ
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
phi = lambda z: np.exp(-z ** 2 / 2) / np.sqrt(2 * np.pi)

M, S = 0.0, 1.0                       # 中心与宽度(示意)
xi_show = np.array([-1.6, -0.7, 0.0, 0.6, 1.5])
x_show = M + S * xi_show

fig = plt.figure(figsize=(15.2, 5.8), dpi=150)

# ---------------- (a) 密度是怎么"搬"过去的
axL = fig.add_axes([0.06, 0.14, 0.20, 0.70])
z = np.linspace(-3.2, 3.2, 600)
axL.plot(phi(z), z, color=PURPLE, lw=2.6)
axL.axhline(0, color=GREY, lw=0.8)
for xi, xv in zip(xi_show, x_show):
    axL.plot(phi(xi), xi, "o", ms=8, color=PURPLE, zorder=6)
    axL.plot([0, phi(xi)], [xi, xi], color=PURPLE, lw=1.0, ls=":")
axL.set_xlim(-0.05, 0.62); axL.set_ylim(-3.2, 3.2)
axL.set_xlabel("密度", fontsize=9.6)
axL.set_ylabel("噪声 $\\xi = (x'-m_\\theta)/s$", fontsize=10)
axL.set_title("左侧:$\\xi\\sim N(0,1)$\n的密度(已知)", fontsize=10.6)
axL.tick_params(labelsize=8.6)

axM = fig.add_axes([0.28, 0.14, 0.09, 0.70]); axM.axis("off")
for xi, xv in zip(xi_show, x_show):
    axM.add_patch(FancyArrowPatch((0.05, xi / 3.4 * 0.5 + 0.5), (0.95, xv / 3.4 * 0.5 + 0.5),
                                  arrowstyle="-|>", mutation_scale=10, lw=1.6, color=GREY))
axM.text(0.5, 0.02, "$x'=m_\\theta+s\\xi$", fontsize=9.6, color=DARK, ha="center")
axM.text(0.5, 0.98, "仿射搬运\n(雅可比 $1/s$)", fontsize=9.0, color=DARK, ha="center", va="top")

axR = fig.add_axes([0.40, 0.14, 0.20, 0.70])
xr = np.linspace(-3.2, 3.2, 600)
axR.plot(xr, phi((xr - M) / S) / S, color=GREEN, lw=2.6)
axR.axhline(0, color=GREY, lw=0.8)
for xi, xv in zip(xi_show, x_show):
    axR.plot(xv, phi(xi) / S, "o", ms=8, color=GREEN, zorder=6)
axR.set_xlim(-3.2, 3.2); axR.set_ylim(-0.05, 0.62)
axR.set_xlabel("落点 $x'$", fontsize=10)
axR.set_ylabel("密度", fontsize=10)
axR.set_title("右侧:这就【是】$\\pi_\\theta(x'|x)$\n$=N(x';m_\\theta,s^2)$", fontsize=10.6)
axR.tick_params(labelsize=8.6)
axR.annotate("靠近中心:只需很小的噪声\n⇒ 落点多(密度高)", xy=(0.0, 0.399), xytext=(-3.1, 0.52),
             fontsize=8.6, color=GREEN, arrowprops=dict(arrowstyle="-|>", lw=1.2, color=GREEN))
axR.annotate("远离中心:需要很大的噪声\n⇒ 落点少(密度低)", xy=(2.0, 0.027), xytext=(-3.1, 0.15),
             fontsize=8.6, color=GREEN, arrowprops=dict(arrowstyle="-|>", lw=1.2, color=GREEN))

# ---------------- (b) 数字核对
ax2 = fig.add_axes([0.63, 0.14, 0.34, 0.70]); ax2.axis("off")
ax2.text(0.5, 1.00, "(b) 数字核对与可用性清单", ha="center", va="top", fontsize=11.4, color=DARK)
ax2.text(0.0, 0.92,
         "蒙特卡洛:抽 $\\xi\\sim N(0,1)$,令 $x'=m+s\\xi$,\n"
         "把 $x'$ 的直方图与 $N(m,s^2)$ 对比 —— 完全重合",
         fontsize=8.8, color=DARK, va="top")
rows = [["要求", "确定性 $\\delta$", "加噪声的高斯"],
        ["密度是有限数", "✗($\\infty$)", "✓"],
        ["归一化 $\\int\\pi=1$", "✗", "✓"],
        ["$\\log\\pi$ 能算", "✗", "✓"],
        ["比值 $\\rho$ 有意义", "✗ 0/0", "✓"],
        ["对 $\\theta$ 可导", "✗", "✓"],
        ["$\\nabla_\\theta\\log\\pi$", "✗", "$(\\xi/s)\\,\\partial m/\\partial\\theta$"]]
tb = ax2.table(cellText=rows, cellLoc="center", bbox=[0.0, 0.30, 1.0, 0.55])
tb.auto_set_font_size(False); tb.set_fontsize(8.2)
for (r, c), cell in tb.get_celld().items():
    if r == 0:
        cell.set_facecolor("#eef2f7")
    if c == 1 and r > 0:
        cell.set_facecolor("#fdecea")
    if c == 2 and r > 0:
        cell.set_facecolor("#eef7f1")
ax2.text(0.0, 0.25,
         "一句话:确定性映射把整个概率空间映到【一个点】,点集上的测度是 $\\delta$;\n"
         "噪声把像铺成【一整片】,密度才有地方可以定义。\n"
         "而\"落点的密度\" = \"反推出来的那个噪声值的密度\"除以 $s$。",
         fontsize=8.8, color=DARK, va="top", bbox=dict(fc="#fffbe6", ec="#e0c060"))

fig.suptitle("为什么加噪声就产生了概率:密度不是凭空出现的,而是从噪声的密度【搬】过来的",
             fontsize=12.2, y=0.99)
fig.savefig(RF + "fg_noise_creates_prob.png", bbox_inches="tight")

# 数值核对:反推噪声
rng = np.random.default_rng(0)
xi = rng.normal(size=200000)
xp = M + S * xi
bins = np.linspace(-3, 3, 61)
h, _ = np.histogram(xp, bins=bins, density=True)
ana = phi((bins[:-1] + bins[1:]) / 2)
print(f"直方图 vs 解析 N(0,1) 的最大偏差: {np.abs(h - ana).max():.4f}(离散化误差量级)")
print("归一化:∫N(x;0,1)dx =", f"{np.trapezoid(phi(np.linspace(-8,8,4001)), np.linspace(-8,8,4001)):.6f}")
print("wrote", RF + "fg_noise_creates_prob.png")
