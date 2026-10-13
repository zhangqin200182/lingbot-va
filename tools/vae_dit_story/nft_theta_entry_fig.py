r"""θ 是怎么进入"每步高斯"的?

一步的输出 = 【确定的均值(含 θ)】 + 【随机的噪声(与 θ 无关)】

  漂移: f = v_θ − (g²/2)∇log p_σ               ← ∇log p 也是用 v_θ 算出来的(!)
  而 ∇log p_σ(x) = −(x + (1−σ)v_θ)/σ
  ⇒ f = (g²/(2σ))·x + [1 + g²(1−σ)/(2σ)]·v_θ     ← 补偿项被吸收成 v_θ 的系数
  ⇒ mean = x + f·dσ = A(σ)·x + B(σ)·v_θ
        A(σ) = 1 + dσ·nl²/(2(1−σ))   (只依赖 σ)
        B    = dσ·(1 + nl²/2)        (与 σ 无关!)

  噪声: x' = mean + s·ξ ,   s = std_t·√|dσ|   ← 只依赖 σ 与 noise_level,不含 θ

⇒ θ 唯一入口 = v_θ(x,σ)(一条可导的确定项);随机性只是外面另加的一项
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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

RED, BLUE, GREEN, DARK, GREY, ORANGE, PURPLE = ("#c0392b", "#2471a3", "#1e8449",
                                                "#2c3e50", "#95a5a6", "#b9770e", "#7d3c98")
RF = "/Users/kevin/code/lingbot-va/.claude/scratch/rf/"
N = lambda x, m, s: np.exp(-(x - m) ** 2 / (2 * s ** 2)) / (s * np.sqrt(2 * np.pi))

fig = plt.figure(figsize=(15.2, 5.6), dpi=150)

# ---------------- (a) 一步的解剖
ax = fig.add_axes([0.055, 0.17, 0.30, 0.66])
xs = np.linspace(-1.2, 1.2, 600)
ax.plot(xs, N(xs, 0.0, 0.20), color=DARK, lw=2.6, label="当前 $\\theta$ 的这一步:$N(m_\\theta,s^2)$")
ax.plot(xs, N(xs, 0.35, 0.20), color=GREEN, lw=2.4, ls="--", label="换个 $\\theta$:中心移动,宽度不变")
rng = np.random.default_rng(1)
xi = rng.normal(size=3)
ax.plot(0.20 * xi, N(0.20 * xi, 0.0, 0.20), "o", ms=9, color=BLUE, zorder=6,
        label="实际抽到的 3 个落点 $m_\\theta+s\\xi$")
ax.axvline(0.0, color=DARK, ls=":", lw=1.2); ax.axvline(0.35, color=GREEN, ls=":", lw=1.2)
ax.annotate("", xy=(0.35, 1.35), xytext=(0.0, 1.35),
            arrowprops=dict(arrowstyle="-|>", lw=2.4, color=GREEN))
ax.text(0.175, 1.40, "$\\theta$ 只挪【中心】", fontsize=9.4, color=GREEN, ha="center")
ax.annotate("", xy=(-0.25, 0.75), xytext=(0.25, 0.75),
            arrowprops=dict(arrowstyle="<|-|>", lw=2.0, color=BLUE))
ax.text(0.0, 0.79, "宽度 $s$ 由 $\\sigma$、noise_level 定,与 $\\theta$ 无关", fontsize=8.8,
        color=BLUE, ha="center")
ax.set_xlim(-1.2, 1.2); ax.set_ylim(-0.05, 1.65)
ax.set_xlabel("这一步的落点 $x'$", fontsize=10.5); ax.set_ylabel("密度", fontsize=10.5)
ax.set_title("(a) 一步的解剖:确定性中心 + 随机噪声\n$\\theta$ 改中心,改不了宽度", fontsize=11.3)
ax.legend(fontsize=8.2, frameon=False, loc="upper left")
ax.tick_params(labelsize=9)

# ---------------- (b) θ 的唯一入口
ax2 = fig.add_axes([0.41, 0.17, 0.32, 0.66]); ax2.axis("off")
ax2.text(0.5, 1.00, "(b) $\\theta$ 的唯一入口:$v_\\theta$", ha="center", va="top",
         fontsize=11.8, color=DARK)
ax2.text(0.0, 0.90,
         "漂移:$f=v_\\theta-\\frac{g^2}{2}\\nabla\\log p_\\sigma$\n\n"
         "但补偿项里的 score 也是用 $v_\\theta$ 算出来的:\n"
         "$\\nabla\\log p_\\sigma(x)=-\\frac{x+(1-\\sigma)v_\\theta}{\\sigma}$\n"
         "(不需要第二个网络)\n\n"
         "代进去 ⇒ $f=\\frac{g^2}{2\\sigma}x+[1+\\frac{g^2(1-\\sigma)}{2\\sigma}]v_\\theta$",
         fontsize=9.4, color=DARK, va="top")
ax2.text(0.0, 0.42,
         "再乘 $d\\sigma$ 加回 $x$:\n"
         "$\\mathrm{mean}=A(\\sigma)\\,x+B\\,v_\\theta$\n\n"
         "$A(\\sigma)=1+\\frac{d\\sigma\\,nl^2}{2(1-\\sigma)}$(只依赖 $\\sigma$)\n"
         "$B=d\\sigma(1+\\frac{nl^2}{2})$(与 $\\sigma$ 无关!)\n\n"
         "⇒ 整个均值 = 【只依赖 σ 的系数】×$x$ + 【只依赖 σ 的系数】×$v_\\theta$",
         fontsize=9.4, color=GREEN, va="top", bbox=dict(fc="#eef7f1", ec=GREEN))
ax2.text(0.0, 0.06,
         "所以\"补偿项\"不是独立于 $\\theta$ 的另一个东西 ——\n它被吸收成了 $v_\\theta$ 的系数。",
         fontsize=9.2, color=RED, va="top", bbox=dict(fc="#fdecea", ec="#e6a9a0"))

# ---------------- (c) 谁依赖 θ
ax3 = fig.add_axes([0.755, 0.17, 0.235, 0.66]); ax3.axis("off")
ax3.text(0.5, 0.99, "(c) 谁依赖 $\\theta$", ha="center", va="top", fontsize=11.6, color=DARK)
rows = [["量", "依赖 $\\theta$?"],
        ["$v_\\theta(x,\\sigma)$", "是"],
        ["均值 $\\mathrm{mean}$", "是\n(只经 $v_\\theta$)"],
        ["$\\nabla\\log p_\\sigma$(补偿)", "是\n(由 $v_\\theta$ 算出)"],
        ["高斯宽度 $s$", "否\n(只依赖 σ)"],
        ["每步噪声 $\\xi$", "否\n(每步重抽)"],
        ["$\\log\\pi_\\theta$", "是\n(只经均值)"]]
tb = ax3.table(cellText=rows, cellLoc="center", bbox=[0.0, 0.44, 1.0, 0.52])
tb.auto_set_font_size(False); tb.set_fontsize(7.8)
for (r, c), cell in tb.get_celld().items():
    if r == 0:
        cell.set_facecolor("#eef2f7")
    if c == 1 and r > 0:
        t = cell.get_text().get_text()
        cell.set_facecolor("#eef7f1" if t.startswith("是") else "#fdecea")
ax3.text(0.0, 0.38,
         "两个直接后果:\n"
         "① $\\nabla_\\theta\\log\\pi_\\theta=(\\xi/s)\\,\\partial m/\\partial\\theta$\n"
         "　(方差与 θ 无关 ⇒ 梯度只经均值)\n"
         "② 逐步 KL 有闭式:\n"
         "　$\\|m_\\theta-m_{\\rm ref}\\|^2/(2s^2)$",
         fontsize=8.6, color=DARK, va="top",
         bbox=dict(fc="#fffbe6", ec="#e0c060"))
ax3.text(0.0, 0.03,
         "一句话:随机的是 $\\xi$,\n确定且可导的是 $v_\\theta$。",
         fontsize=8.8, color=GREEN, va="bottom", bbox=dict(fc="#eef7f1", ec="#a9d5bb"))

fig.suptitle("$\\theta$ 是怎么进入\"每步高斯\"的:只通过 $v_\\theta$ 影响【均值】;宽度与噪声都与 $\\theta$ 无关",
             fontsize=12.2, y=0.99)
fig.savefig(RF + "fg_theta_enters_step.png", bbox_inches="tight")
print("wrote", RF + "fg_theta_enters_step.png")
