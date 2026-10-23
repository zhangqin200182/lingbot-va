r"""x_t = (1−t)x₀ + t·ε 是什么分布?

  它是两个【独立】随机变量之和:
      (1−t)x₀        ← 被缩放后的数据分布(形状继承 p_data)
      t·ε            ← 协方差 t²I 的各向同性高斯
  ⇒ 卷积:p_t = Law((1−t)x₀) ⊛ N(0, t²I)
  ⇒ 积分形式:p_t(x) = ∫ p_data(x₀) · N(x; (1−t)x₀, t²I) dx₀
  ⇒ 等价说法:以每个 (1−t)x₀ 为中心、宽 t 的高斯、按 p_data 加权的高斯混合
            (即"用高斯核对 p_data 做缩放后的核密度")

  只有 p_data 本身是高斯时才仍是高斯(高斯对卷积稳定):
      mean = (1−t)μ , cov = (1−t)²Σ + t²I
  一般不是高斯:两原子数据 ⇒ 双峰(超额峰度≈−1.4,Shapiro p≈1e−45)
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

RED, BLUE, GREEN, DARK, GREY, ORANGE, PURPLE = ("#c0392b", "#2471a3", "#1e8449",
                                                "#2c3e50", "#95a5a6", "#b9770e", "#7d3c98")
RF = "/Users/kevin/code/lingbot-va/.claude/scratch/rf/"
N = lambda x, m, s: np.exp(-(x - m) ** 2 / (2 * s ** 2)) / (s * np.sqrt(2 * np.pi))

fig = plt.figure(figsize=(15.2, 5.7), dpi=150)

# ---------------- (a) 三种读法
ax = fig.add_axes([0.03, 0.08, 0.31, 0.82]); ax.axis("off")
ax.text(0.5, 1.00, "(a) 三种等价读法", ha="center", va="top", fontsize=12, color=DARK)
ax.text(0.0, 0.93,
         "$x_t=(1-t)x_0+t\\,\\epsilon$,$\\ x_0\\sim p_{\\rm data}$,$\\ \\epsilon\\sim N(0,I)$\n"
         "两者【独立】",
         fontsize=9.4, color=DARK, va="top")
ax.text(0.0, 0.80,
         "① 卷积(最本质)\n"
         "　 $p_t={\\rm Law}((1-t)x_0)\\;\\circledast\\;N(0,t^2I)$",
         fontsize=9.0, color=BLUE, va="top", bbox=dict(fc="#eaf2fa", ec=BLUE))
ax.text(0.0, 0.65,
         "② 高斯混合(最好算)\n"
         "　 $p_t(x)=\\int p_{\\rm data}(x_0)\\,N(x;(1-t)x_0,\\,t^2I)\\,dx_0$\n"
         "　 = 以每个 $(1-t)x_0$ 为中心、宽 $t$ 的高斯\n"
         "　　 按 $p_{\\rm data}$ 加权",
         fontsize=9.0, color=GREEN, va="top", bbox=dict(fc="#eef7f1", ec=GREEN))
ax.text(0.0, 0.44,
         "③ 重参数化(location-scale)\n"
         "　 $x_t=a_t x_0+b_t\\epsilon$,$\\ a_t=1-t$,$\\ b_t=t$",
         fontsize=9.0, color=PURPLE, va="top", bbox=dict(fc="#f4ecf7", ec=PURPLE))
ax.text(0.0, 0.30,
         "两端:\n"
         "　$t=0$:$p_0=p_{\\rm data}$(δ 混合)\n"
         "　$t=1$:$p_1=N(0,I)$ ✓ 标准高斯",
         fontsize=9.0, color=DARK, va="top")
ax.text(0.0, 0.15,
         "★ 一般【不是】高斯 ——\n"
         "形状由 $p_{\\rm data}$ 决定,只是被高斯核抹平。",
         fontsize=9.2, color=RED, va="top", bbox=dict(fc="#fdecea", ec="#e6a9a0"))
ax.set_xlim(0, 1); ax.set_ylim(0.02, 1.02)

# ---------------- (b) 两原子数据:双峰 → 变宽 → 高斯
ax2 = fig.add_axes([0.38, 0.16, 0.29, 0.68])
xs = np.linspace(-2.2, 2.2, 900)
w = 0.8
for t, col, lw in [(0.12, "#1a5276", 2.6), (0.30, BLUE, 2.4), (0.55, GREEN, 2.4),
                   (0.80, ORANGE, 2.4), (1.00, RED, 2.6)]:
    p = w * N(xs, (1 - t) * 1.0, max(t, 1e-3)) + (1 - w) * N(xs, -(1 - t) * 1.0, max(t, 1e-3))
    ax2.plot(xs, p, color=col, lw=lw, label=f"$t={t}$")
ax2.plot(xs, N(xs, 0, 1), color=RED, lw=1.2, ls=":")
ax2.set_xlabel("$x_t$", fontsize=10.4); ax2.set_ylabel("$p_t(x)$", fontsize=10.4)
ax2.set_title("(b) 数据是两原子(0.8/0.2)时:\n双峰 → 变宽 → 趋于标准高斯", fontsize=11.0)
ax2.legend(fontsize=8.2, frameon=False, loc="upper right")
ax2.tick_params(labelsize=9)
ax2.set_xlim(-2.2, 2.2); ax2.set_ylim(0, 4.3)
ax2.text(-2.1, 4.1, "t=0 时是 δ 混合(画不出来)\n两峰高度比 4:1 一直到 t→1 都保留",
         fontsize=8.2, color=GREY, va="top")

# ---------------- (c) 什么时候才是高斯
ax3 = fig.add_axes([0.71, 0.08, 0.27, 0.82]); ax3.axis("off")
ax3.text(0.5, 1.00, "(c) 什么时候才真的是高斯", ha="center", va="top", fontsize=11.4, color=DARK)
ax3.text(0.0, 0.92,
         "只有当 $p_{\\rm data}$ 本身是高斯:\n\n"
         "　$p_{\\rm data}=N(\\mu,\\Sigma)$\n"
         "　⇒ $x_t$ 仍是高斯(高斯对卷积稳定)\n"
         "　　mean $=(1-t)\\,\\mu$\n"
         "　　cov  $=(1-t)^2\\Sigma+t^2I$",
         fontsize=8.8, color=GREEN, va="top", bbox=dict(fc="#eef7f1", ec=GREEN))
ax3.text(0.0, 0.62,
         "数值核对($p_{\\rm data}=N(0,1)$):\n"
         "　t=0.2:实测 Var 0.6808 / 解析 0.6800\n"
         "　t=0.5:实测 0.4998 / 解析 0.5000\n"
         "　t=0.9:实测 0.8157 / 解析 0.8200",
         fontsize=8.4, color=DARK, va="top", bbox=dict(fc="#fffbe6", ec="#e0c060"))
ax3.text(0.0, 0.44,
         "两原子数据则明显【不是】高斯:\n"
         "　t=0.3:超额峰度 −1.428,Shapiro p≈8e−45\n"
         "　t=0.5:超额峰度 −0.500,p≈4e−10\n"
         "　t=0.8:峰度 +0.002,p≈0.28 ← 快接近高斯了",
         fontsize=8.4, color=RED, va="top", bbox=dict(fc="#fdecea", ec="#e6a9a0"))
ax3.text(0.0, 0.24,
         "⇒ 所以\"后验/边缘是高斯\"是常见误解;\n"
         "　 只有 $t\\to1$ 时才近似高斯(高斯项占主导)。",
         fontsize=8.6, color=DARK, va="top")
ax3.text(0.0, 0.10,
         "附:协方差恒等式\n"
         "$\\mathrm{Cov}(x_t)=(1-t)^2\\mathrm{Cov}(x_0)+t^2I$",
         fontsize=8.6, color=BLUE, va="bottom", bbox=dict(fc="#eaf2fa", ec=BLUE))

fig.suptitle("$x_t=(1-t)x_0+t\\epsilon$ 是什么分布?—— 缩放后的数据分布与各向同性高斯的【卷积】/ 高斯混合;一般不是高斯",
             fontsize=11.2, y=0.99)
fig.savefig(RF + "fg_xt_distribution.png", bbox_inches="tight")
print("wrote", RF + "fg_xt_distribution.png")
