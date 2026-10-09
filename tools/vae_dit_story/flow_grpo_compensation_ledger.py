r"""补偿是怎么发生的:一步的方差账 + 方差轨迹。

数据 = N(0,1) → p_σ = N(0, V(σ)),V(σ)=(1−σ)²+σ²。
(a) 三种系数 c 下的【方差轨迹】Var(σ),与解析 V(σ) 对比 → 偏差从哪里累积
(b) 一步的账(σ=0.5 → 0.49):噪声 +g²|dσ| 与偏移 −g²|dσ| 精确相消
(c) 一般密度层面的对消(不只是方差)
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
N = 200
SIGG = np.linspace(1.0, 0.0, N + 1)
SMAX = SIGG[1]; NL = 0.5
Vfun = lambda s: (1 - s) ** 2 + s ** 2


def vf(x, s):
    return (x - (1 - s) * x / Vfun(s)) / max(s, 1e-6)


def sc(x, s):
    return -x / Vfun(s)


def g_of(s):
    a = min(s, SMAX)
    return np.sqrt(a / (1 - a)) * NL


def traj_var(c, M=40000, gen=None):
    gen = gen or np.random.default_rng(4)
    x = gen.normal(size=M)
    out = [x.var()]
    for i in range(N):
        s, sn = SIGG[i], SIGG[i + 1]
        d = sn - s
        gg = g_of(s)
        x = x + (vf(x, s) - c * 0.5 * gg ** 2 * sc(x, s)) * d + gg * np.sqrt(-d) * gen.normal(size=M)
        out.append(x.var())
    return np.array(out)


fig = plt.figure(figsize=(15.0, 5.4), dpi=150)

# ---------------- (a) 方差轨迹
ax = fig.add_axes([0.055, 0.16, 0.31, 0.68])
keep = SIGG <= 0.92
ax.plot(SIGG[keep], [Vfun(s) for s in SIGG[keep]], "k--", lw=2.4, label="解析 $V(\\sigma)$(真值)")
for c, col in zip([0.0, 0.5, 1.0, 1.5], [RED, ORANGE, GREEN, PURPLE]):
    ax.plot(SIGG[keep], traj_var(c)[keep], color=col, lw=1.9,
            label=f"$c={c}$" + ("(正确)" if c == 1.0 else ""))
ax.set_xlabel("$\\sigma$", fontsize=10.5); ax.set_ylabel("样本方差 Var($x_\\sigma$)", fontsize=10.5)
ax.set_title("(a) 方差轨迹(σ≤0.92):$c=1$ 紧贴真值\n$c<1$ 越走越宽,$c>1$ 越走越窄", fontsize=11.5)
ax.legend(fontsize=8.8, frameon=False, loc="upper right")
ax.tick_params(labelsize=9)

# ---------------- (b) 一步的账
ax2 = fig.add_axes([0.42, 0.16, 0.28, 0.68]); ax2.axis("off")
S0, D = 0.5, -0.01
V0 = Vfun(S0); g0 = g_of(S0); a_off = (g0 ** 2 / 2) / V0
noise_gain = g0 ** 2 * abs(D)
off_loss = 2 * a_off * V0 * abs(D)
ax2.text(0.5, 0.99, "(b) 一步的账(σ: 0.5 → 0.49)", ha="center", va="top", fontsize=11.5, color=DARK)
steps = [
    ("起点方差", f"$V={V0:.4f}$", DARK),
    (f"这一步注入的噪声 $g^2|d\\sigma|$", f"$+{noise_gain:.4f}$", RED),
    (f"偏移项把方差收紧 $2\\Delta a V|d\\sigma|$", f"$-{off_loss:.4f}$", GREEN),
    ("净变化", f"${noise_gain-off_loss:+.4f}$", BLUE),
    ("解析要求 $V'(\\sigma)d\\sigma=(4\\sigma-2)d\\sigma$", f"${(4*S0-2)*D:+.4f}$", DARK),
]
y = 0.86
for name, val, col in steps:
    ax2.text(0.02, y, name, fontsize=9.4, color=col, va="top")
    ax2.text(0.98, y, val, fontsize=10, color=col, va="top", ha="right")
    y -= 0.11
ax2.text(0.02, y - 0.02,
         "其中 $\\Delta a=\\frac{g^2}{2V}$ 就是恒等式里那一项\n"
         "贡献的收紧量。\n\n"
         "⇒ 噪声撑大多少,偏移就压回多少,\n步内精确相消(净 0 = 解析要求 0)。",
         fontsize=9.2, color=DARK, va="top",
         bbox=dict(fc="#eef7f1", ec="#a9d5bb"))

# ---------------- (c) 一般密度层面
ax3 = fig.add_axes([0.735, 0.16, 0.245, 0.68]); ax3.axis("off")
ax3.text(0.5, 0.99, "(c) 不只是方差:整个密度", ha="center", va="top", fontsize=11.5, color=DARK)
ax3.text(0.02, 0.86,
         "密度演化(Fokker–Planck):\n"
         "$\\partial_\\sigma p=-\\nabla\\!\\cdot\\!(f p)+\\frac{g^2}{2}\\Delta p$\n"
         "　　　　　　└ 漂移搬运 ┘　└ 噪声抹平 ┘\n\n"
         "代入 $f=v^*+\\frac{g^2}{2}\\nabla\\log p$:\n"
         "$-\\nabla\\!\\cdot\\!(fp)=-\\nabla\\!\\cdot\\!(v^*p)-\\frac{g^2}{2}\\Delta p$\n"
         "(用了 $\\Delta p=\\nabla\\!\\cdot\\!(p\\nabla\\log p)$)\n\n"
         "代回去:$\\partial_\\sigma p=-\\nabla\\!\\cdot\\!(v^*p)$\n"
         "→ 与 ODE 的密度方程【完全相同】\n"
         "→ 所以每个 σ 的分布都不变。",
         fontsize=9.0, color=DARK, va="top")
ax3.text(0.02, 0.10,
         "一句话:补偿发生在【同一步里】,\n不是先走偏再修回来。",
         fontsize=9.4, color=GREEN, va="top", bbox=dict(fc="#eef7f1", ec="#a9d5bb"))

fig.suptitle("补偿是怎么发生的:噪声在同一步里把分布撑开,偏移项在同一步里把它压回 —— 精确相消", fontsize=12.6, y=0.99)
fig.savefig(RF + "fg_compensation_ledger.png", bbox_inches="tight")

print(f"一步的账(σ={S0} → {S0+D},g={g0:.4f},V={V0:.4f}):")
print(f"  噪声贡献        +{noise_gain:.6f}")
print(f"  偏移贡献        -{off_loss:.6f}   (Δa = g²/(2V) = {a_off:.4f})")
print(f"  净变化          {noise_gain-off_loss:+.6f}")
print(f"  解析要求        {(4*S0-2)*D:+.6f}")
print("方差轨迹终点值:", ", ".join(f"c={c}:{traj_var(c)[-1]:.4f}" for c in [0.0, 0.5, 1.0, 1.5]))
print("wrote", RF + "fg_compensation_ledger.png")
