r"""如果 ε 已经提供多样性,为什么还要每步注入 g·dW?

答案:注入噪声买的不是【多样性】,而是【可用的似然】。

  ODE:ε → x(T) 是【确定性映射】
       · "这一张图的概率" = δ(不可用)
       · 要算密度只能走变量替换:log p(x) = log p_noise(ε) − log|det ∂F_θ/∂ε|
         ⇒ 需要整条 T 步映射的雅可比行列式(neural-ODE 似然):昂贵、数值不稳
       · 而且只有"整条一个数",没有每步的概率

  SDE:每一步都是 θ 依赖的高斯
       · π_θ(x_{j+1}|x_j) = N(mean_θ(x_j), s_j²)      ⇒ 每步都有可算的对数密度
       · log π_θ(traj) = Σ_j log N(...)               ⇒ 能直接对 θ 求导
       · ∇_θ log π_θ = (ξ_j/s_j)·∂m_j/∂θ              ⇒ 才有策略梯度

⇒ 多样性:两者都有(靠 ε);差别只在【有没有可用的似然】。
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
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

RED, BLUE, GREEN, DARK, GREY, ORANGE, PURPLE = ("#c0392b", "#2471a3", "#1e8449",
                                                "#2c3e50", "#95a5a6", "#b9770e", "#7d3c98")
RF = "/Users/kevin/code/lingbot-va/.claude/scratch/rf/"

fig = plt.figure(figsize=(15.2, 6.0), dpi=150)

# ---------------- 上排:ODE
ax = fig.add_axes([0.035, 0.52, 0.44, 0.40]); ax.axis("off")
ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.02", fc="#f7f9fb", ec=GREY, lw=1.8))
ax.text(0.5, 0.94, "ODE:多样性 ✓,但【似然不可用】", ha="center", fontsize=11.4, color=BLUE)
ys = np.linspace(0.25, 0.72, 4)
for i, y0 in enumerate(ys):
    e = [-1.6, -0.5, 0.6, 1.6][i]
    ax.add_patch(FancyArrowPatch((0.10, y0), (0.44, y0 + 0.04 * e), arrowstyle="-|>",
                                 mutation_scale=12, lw=2.0, color=(BLUE if e > 0 else ORANGE)))
    ax.text(0.075, y0, f"$\\epsilon_{i+1}$", fontsize=9.0, color=DARK, ha="right", va="center")
    ax.text(0.47, y0 + 0.04 * e, f"$x^{{({i+1})}}$", fontsize=9.0, color=DARK, va="center")
ax.text(0.66, 0.62, "不同 $\\epsilon$ → 不同 $x$\n(多样性 ✓)", fontsize=9.4, color=GREEN, va="center")
ax.text(0.66, 0.34,
        "但给定 $\\epsilon$ 后 $x$ 是【确定的】\n"
        "⇒ 概率是 $\\delta$;要算密度只能:\n"
        "　$\\log p(x)=\\log p_{\\rm noise}(\\epsilon)-\\log|\\det\\partial F_\\theta/\\partial\\epsilon|$\n"
        "　= 整条 $T$ 步映射的雅可比 —— 昂贵、不稳\n"
        "⇒ 而且只有\"整条一个数\"",
        fontsize=8.6, color=RED, va="center")

# ---------------- 下排:SDE
ax2 = fig.add_axes([0.035, 0.06, 0.44, 0.40]); ax2.axis("off")
ax2.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.02", fc="#eef7f1", ec=GREEN, lw=1.8))
ax2.text(0.5, 0.94, "SDE:多样性 ✓ + 【每步似然可算】", ha="center", fontsize=11.4, color=GREEN)
xs = np.linspace(0.10, 0.50, 5)
for i in range(4):
    ax2.add_patch(FancyArrowPatch((xs[i], 0.45), (xs[i + 1], 0.45 + (0.14 if i % 2 else -0.12)),
                                  arrowstyle="-|>", mutation_scale=11, lw=2.0, color=GREEN))
    ax2.text((xs[i] + xs[i + 1]) / 2, 0.60 if i % 2 else 0.28,
             f"$N(m_\\theta,s_{i+1}^2)$", fontsize=8.4, color=GREEN, ha="center")
ax2.text(0.055, 0.45, "$\\epsilon$", fontsize=9.4, color=DARK, ha="right", va="center")
ax2.text(0.53, 0.45, "$x$", fontsize=9.4, color=DARK, va="center")
ax2.text(0.63, 0.62,
         "每步都是【θ 依赖的高斯】\n"
         "$\\pi_\\theta(x_{j+1}|x_j)=N(m_\\theta(x_j),s_j^2)$\n"
         "$\\Rightarrow\\log\\pi_\\theta=\\sum_j\\log N(\\cdot)$ 可算",
         fontsize=9.0, color=DARK, va="center")
ax2.text(0.63, 0.26,
         "$\\nabla_\\theta\\log\\pi_\\theta=(\\xi_j/s_j)\\,\\partial m_j/\\partial\\theta$\n"
         "⇒ 才有策略梯度\n"
         "(多样性仍然来自 $\\epsilon$;\n　 加噪买的是似然,不是多样性)",
         fontsize=8.6, color=GREEN, va="center")

# ---------------- 右栏:结论
ax3 = fig.add_axes([0.50, 0.06, 0.48, 0.86]); ax3.axis("off")
ax3.text(0.5, 1.00, "为什么会这样:两件事被混在一起了", ha="center", va="top", fontsize=12, color=DARK)
rows = [["", "ODE", "SDE"],
        ["不同 $\\epsilon$ → 不同输出", "✓", "✓"],
        ["给定 $\\epsilon$ 后输出是否确定", "确定", "随机"],
        ["\"这一张图\"的概率", "$\\delta$/雅可比", "每步高斯"],
        ["每步对数密度", "没有", "有"],
        ["对数似然比 $\\rho$", "构造不出", "可构造"],
        ["策略梯度", "无法估计", "可以"],
        ["额外收益", "—", "逐步信赖域(1.001¹⁹≈1.019)"]]
tb = ax3.table(cellText=rows, cellLoc="center", bbox=[0.0, 0.55, 1.0, 0.42])
tb.auto_set_font_size(False); tb.set_fontsize(9.0)
for (r, c), cell in tb.get_celld().items():
    if r == 0:
        cell.set_facecolor("#eef2f7")
    if c == 1 and r > 0:
        cell.set_facecolor("#f7f9fb" if r > 2 else "#eef7f1")
    if c == 2 and r > 0:
        cell.set_facecolor("#eef7f1" if r > 2 else "#eef7f1")
ax3.text(0.0, 0.47,
         "一个反过来的证据:\n"
         "如果加噪声只是为了\"多样性\",\n"
         "那就没必要费那么大劲把分布保回来\n"
         "(补偿项 $-(g^2/2)\\nabla\\log p$)。\n"
         "恰恰因为我们【必须】加噪声(为了似然),\n"
         "才不得不花力气让分布不变。",
         fontsize=9.4, color=RED, va="top", bbox=dict(fc="#fdecea", ec="#e6a9a0"))
ax3.text(0.0, 0.10,
         "一句话:多样性靠 $\\epsilon$ 就有;\n"
         "每步注入 $g\\,dW$ 买的是\"每步一个可算的概率\",\n"
         "由此才有 $\\rho$、才有策略梯度。",
         fontsize=9.6, color=GREEN, va="top", bbox=dict(fc="#eef7f1", ec="#a9d5bb"))

fig.suptitle("ε 已经带来多样性;每步注入 $g\\,dW$ 换来的不是多样性,而是【可用的似然】", fontsize=12.4, y=0.99)
fig.savefig(RF + "fg_why_inject_noise.png", bbox_inches="tight")
print("wrote", RF + "fg_why_inject_noise.png")
