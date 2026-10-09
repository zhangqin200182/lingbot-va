r"""Step 4 在干什么:一族"等价 SDE"。

命题:对【任意】噪声强度 g(σ),只要漂移取 f = v* ∓ (g²/2)∇log p_σ,
      边缘分布 p_σ 全部保持不变;g=0 退回 ODE(Step 3)。

(a) 同一族 p_σ 下,三条不同 noise_level 的轨迹(0 / 0.5 / 1.0),g(σ)=√(σ/(1−σ))·nl
(b) σ=0 终点直方图:三档都复现同一个 p_data   ← 数值验证 f 的公式
(c) 数值核对表 + 为什么是这一项(g 的选择在数值上也重要)
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
MU = np.array([1.0, -1.0]); PR = np.array([0.5, 0.5])
NSTEP = 100
SIGG = np.linspace(1.0, 0.0, NSTEP + 1)
SMAX = SIGG[1]
LEVELS = [0.0, 0.5, 1.0]
COLS = [BLUE, ORANGE, RED]


def p_sig(x, s):
    m = (1 - s) * MU
    return np.sum([PR[k] * np.exp(-(x - m[k]) ** 2 / (2 * s ** 2)) / np.sqrt(2 * np.pi * s ** 2)
                   for k in range(2)], axis=0)


def wts(x, s):
    """x: (M,) → (M,2) 后验权重"""
    m = (1 - s) * MU
    lw = np.log(PR)[None, :] - 0.5 * np.log(2 * np.pi * s ** 2) - (x[:, None] - m[None, :]) ** 2 / (2 * s ** 2)
    lw = lw - lw.max(axis=1, keepdims=True)
    w = np.exp(lw)
    return w / w.sum(axis=1, keepdims=True)


def v_field(x, s):
    return (x - wts(x, s) @ MU) / max(s, 1e-6)


def score(x, s):
    m = (1 - s) * MU
    return np.sum(wts(x, s) * (-(x[:, None] - m[None, :]) / s ** 2), axis=1)


def g_of(s, nl):
    a = min(s, SMAX)
    return np.sqrt(a / (1 - a)) * nl


def sde(x0, nl, gen):
    """x0: (M,) → 轨迹 (NSTEP+1, M)"""
    x = x0.copy()
    tr = [x.copy()]
    for i in range(NSTEP):
        s, sn = SIGG[i], SIGG[i + 1]
        d = sn - s
        g = g_of(s, nl)
        drift = v_field(x, s) - 0.5 * g ** 2 * score(x, s)     # 生成方向:减号
        x = x + drift * d + g * np.sqrt(-d) * gen.normal(size=x.shape)
        tr.append(x.copy())
    return np.array(tr)


fig = plt.figure(figsize=(15.2, 5.7), dpi=150)

# ---------------- (a) 三条轨迹(同一个起点)
ax = fig.add_axes([0.05, 0.16, 0.30, 0.68])
xs = np.linspace(-2.4, 2.4, 400)
for s in [1.0, 0.6, 0.25, 0.0]:
    d = p_sig(xs, s); sc = 0.34 / d.max()
    ax.plot(s + d * sc, xs, color="#8a6d3b", lw=1.3, alpha=0.85)
    ax.fill_betweenx(xs, s, s + d * sc, color="#f0c987", alpha=0.30)
E0 = np.array([-0.55])
for lvl, c in zip(LEVELS, COLS):
    t = sde(E0, lvl, np.random.default_rng(7))
    ax.plot(SIGG, t[:, 0], color=c, lw=2.4 if lvl == 0 else 1.6, alpha=0.95,
            label=("noise_level $=0$(ODE)" if lvl == 0 else f"noise_level $={lvl}$"))
ax.plot([1], [E0[0]], "o", ms=9, color=DARK, zorder=6)
ax.text(1.02, E0[0] + 0.07, "同一个 $\\epsilon$", fontsize=9.4, color=DARK)
ax.set_xlim(1.32, -0.05); ax.set_ylim(-2.5, 2.4)
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlabel("$\\sigma$(1 → 0,生成方向)", fontsize=10.5)
ax.set_title("(a) noise_level 是旋钮\n越大越随机,但分布不变", fontsize=11.5)
ax.legend(fontsize=8.8, frameon=False, loc="lower left")

# ---------------- (b) 终点分布
ax2 = fig.add_axes([0.42, 0.16, 0.29, 0.68])
M = 4000
stats = []
for lvl, c in zip(LEVELS, COLS):
    gen = np.random.default_rng(42)
    ends = sde(gen.normal(size=M), lvl, gen)[-1]
    ax2.hist(ends, bins=np.linspace(-2.4, 2.4, 90), density=True, histtype="step", lw=1.9,
             color=c, label=("noise_level $=0$" if lvl == 0 else f"noise_level $={lvl}$"))
    stats.append((lvl, ends.mean(), ends.std(), float((ends > 0).mean()), float(np.abs(ends).max())))
ax2.plot(xs, p_sig(xs, 0.0), "k--", lw=2.4, label="真值 $p_{data}$")
ax2.set_xlim(-2.3, 2.3)
ax2.set_xlabel("$x$(σ=0 的终点)", fontsize=10.5); ax2.set_ylabel("密度", fontsize=10.5)
ax2.set_title("(b) 三档噪声的终点分布都等于 $p_{data}$", fontsize=11.5)
ax2.legend(fontsize=8.8, frameon=False, loc="upper center")
ax2.tick_params(labelsize=9)

# ---------------- (c) 表格 + 直观
ax3 = fig.add_axes([0.745, 0.16, 0.235, 0.68]); ax3.axis("off")
ax3.text(0.5, 0.99, "(c) 数值核对(σ=0 终点)", ha="center", va="top", fontsize=11.2, color=DARK)
tbl = [["noise\nlevel", "均值", "std", "P(+1)"]]
for lvl, m, s, p, mx in stats:
    tbl.append([f"{lvl:g}", f"{m:+.3f}", f"{s:.3f}", f"{p:.3f}"])
tbl.append(["真值", "+0.000", "1.000", "0.500"])
t = ax3.table(cellText=tbl, cellLoc="center", colWidths=[0.24, 0.26, 0.24, 0.26], bbox=[0.0, 0.66, 1.0, 0.30])
t.auto_set_font_size(False); t.set_fontsize(8.4); t.scale(1.0, 1.42)
for (r, c), cell in t.get_celld().items():
    if r == 0:
        cell.set_facecolor("#eef2f7")
    if r == len(tbl) - 1:
        cell.set_facecolor("#eef7f1")
ax3.text(0.0, 0.60,
         "为什么是这个式子:\n"
         "噪声把密度【抹平】:$\\frac{g^2}{2}\\Delta p$\n"
         "漂移必须把它【拉回】:$\\frac{g^2}{2}\\nabla\\log p$\n"
         "而 $\\Delta p=\\nabla\\!\\cdot\\!(p\\nabla\\log p)$\n"
         "两项精确抵消 → $p_\\sigma$ 不变",
         fontsize=8.8, color=DARK, va="top",
         bbox=dict(fc="#fffbe6", ec="#e0c060"))
ax3.text(0.0, 0.30,
         "$g=0$ → 退回 Step 3 的 ODE ✓\n$g$ 自由选,漂移被 $g$ 唯一决定。\n\n"
         "注:论文取 $g(\\sigma)=\\sqrt{\\frac{\\sigma}{1-\\sigma}}\\cdot$nl,\n"
         "它在 σ→0 自动衰减到 0;若改用常数 $g$,\n"
         "末尾 score 变尖会把数值积分炸掉(实测 |x|→∞)。",
         fontsize=8.4, color=GREEN, va="top")

fig.suptitle("Step 4 在干什么:找【一族】随机采样器 —— 任选噪声 $g$,只要漂移补偿 $\\frac{g^2}{2}\\nabla\\log p$,分布就不变",
             fontsize=12.4, y=0.995)
fig.savefig(RF + "fg_equivalent_sde_family.png", bbox_inches="tight")

print(f"σ=0 终点统计(每档 {M} 条,{NSTEP} 步):")
for lvl, m, s, p, mx in stats:
    print(f"  noise_level={lvl:<4}: 均值 {m:+.4f}  std {s:.4f}  P(+1)={p:.4f}  |x|max={mx:.2f}")
print("  真值            : 均值 +0.0000  std 1.0000  P(+1)=0.5000")
print("wrote", RF + "fg_equivalent_sde_family.png")
