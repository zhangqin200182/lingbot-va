r"""两张图:
Fig A  什么是 ODE、什么是 SDE(定义级),以及 Step 3 到底在证什么
Fig B  偏移量不能乱加:补偿系数 c 取错时分布会被抹宽/压窄
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

# 数据 = N(0,1)(解析可算);p_σ = N(0, (1−σ)²+σ²)
N = 200
SIGG = np.linspace(1.0, 0.0, N + 1)
SMAX = SIGG[1]
NL = 0.5
V = lambda s: (1 - s) ** 2 + s ** 2


def vf(x, s):
    return (x - (1 - s) * x / V(s)) / max(s, 1e-6)


def sc(x, s):
    return -x / V(s)


def g_of(s):
    a = min(s, SMAX)
    return np.sqrt(a / (1 - a)) * NL


def run(c, x0, gen, nsteps=N):
    x = x0.copy()
    tr = [x.copy()]
    for i in range(nsteps):
        s, sn = SIGG[i], SIGG[i + 1]
        d = sn - s
        gg = g_of(s)
        x = x + (vf(x, s) - c * 0.5 * gg ** 2 * sc(x, s)) * d + gg * np.sqrt(-d) * gen.normal(size=x.shape)
        tr.append(x.copy())
    return np.array(tr)


# ============================================================ Fig A
fig, axes = plt.subplots(1, 2, figsize=(14.6, 5.2), dpi=150,
                         gridspec_kw=dict(width_ratios=[1.05, 1.0], wspace=0.22))
ax = axes[0]
xs = np.linspace(-3.4, 3.4, 400)
for s in [1.0, 0.6, 0.25, 0.0]:
    sd = np.sqrt(V(s))
    dens = np.exp(-xs ** 2 / (2 * sd ** 2)) / (sd * np.sqrt(2 * np.pi))
    sc_ = 0.30 / dens.max()
    ax.plot(s + dens * sc_, xs, color="#8a6d3b", lw=1.3, alpha=0.85)
    ax.fill_betweenx(xs, s, s + dens * sc_, color="#f0c987", alpha=0.30)
E0 = np.array([-1.1])
t_ode = run(0.0, E0, np.random.default_rng(1))
ax.plot(SIGG, t_ode[:, 0], color=BLUE, lw=3.0, label="ODE:$dx=v\\,d\\sigma$(轨迹唯一)")
for k, c in enumerate([PURPLE, ORANGE, RED]):
    t = run(1.0, E0, np.random.default_rng(20 + k))
    ax.plot(SIGG, t[:, 0], color=c, lw=1.4, alpha=0.9,
            label=("SDE:$dx=f\\,d\\sigma+g\\,dW$(每次不同)" if k == 0 else None))
ax.plot([1], [E0[0]], "o", ms=9, color=DARK, zorder=6)
ax.text(1.02, E0[0] - 0.12, "同一个起点 $\\epsilon$", fontsize=9.4, color=DARK)
ax.set_xlim(-0.05, 1.34); ax.set_ylim(-3.6, 3.4)
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlabel("$\\sigma$(1 = 纯噪声 → 0 = 数据)", fontsize=10.5)
ax.set_title("(a) 同一个起点:ODE 一条路,SDE 每次不同", fontsize=11.8)
ax.legend(fontsize=8.8, frameon=False, loc="lower left")

ax2 = axes[1]; ax2.axis("off")
rows = [["", "ODE(常微分方程)", "SDE(随机微分方程)"],
        ["方程", "$dx=v(x,\\sigma)\\,d\\sigma$", "$dx=f(x,\\sigma)\\,d\\sigma+g(\\sigma)\\,dW$"],
        ["有没有随机", "没有", "有:$dW\\sim N(0,d\\sigma)$"],
        ["给定起点", "轨迹【唯一】", "每次都不一样"],
        ["叫什么", "速度场 $v$", "漂移 $f$ + 扩散 $g$"],
        ["生活例子", "抛体轨迹", "花粉在水里的布朗运动"],
        ["本文用例", "Step 3 的采样器", "Step 4 找的那一族"]]
tb = ax2.table(cellText=rows, cellLoc="left", bbox=[0.0, 0.22, 1.0, 0.72])
tb.auto_set_font_size(False); tb.set_fontsize(9.4)
for (r, c), cell in tb.get_celld().items():
    cell.set_edgecolor("#d5dbe2")
    if r == 0:
        cell.set_facecolor("#eef2f7"); cell.set_text_props(weight="bold")
    if c == 0:
        cell.set_facecolor("#fafbfc")
ax2.text(0.0, 0.16, "Step 3 在证什么:", fontsize=11, color=DARK, va="top", fontweight="bold")
ax2.text(0.0, 0.07,
         "训练只保证【每个点】上 $v$ 准(局部);"
         "采样是【整体】行为(全局)。\n"
         "Step 3 证明:按 ODE 走,每个 $\\sigma$ 时刻的分布确实等于 $p_\\sigma$。",
         fontsize=9.6, color=DARK, va="top",
         bbox=dict(fc="#eef7f1", ec="#a9d5bb"))
fig.suptitle("先补定义:ODE 是「确定性速度场」,SDE 是「速度场 + 每步随机晃动」", fontsize=13, y=0.99)
fig.savefig(RF + "fg_ode_vs_sde_basic.png", bbox_inches="tight")

# ============================================================ Fig B
fig2, axes2 = plt.subplots(1, 3, figsize=(15.2, 5.0), dpi=150,
                           gridspec_kw=dict(width_ratios=[1.0, 1.0, 0.85], wspace=0.28))
CS = [0.0, 0.5, 1.0, 1.5]
COLS = [RED, ORANGE, GREEN, PURPLE]
M = 20000
res = {}
for c, col in zip(CS, COLS):
    gen = np.random.default_rng(3)
    e = run(c, gen.normal(size=M), gen)[-1]
    res[c] = e

ax = axes2[0]
for c, col in zip(CS, COLS):
    ax.hist(res[c], bins=np.linspace(-6, 6, 160), density=True, histtype="step", lw=1.9, color=col,
            label=f"$c={c}$" + ("(正确)" if c == 1.0 else ""))
ax.plot(xs, np.exp(-xs ** 2 / 2) / np.sqrt(2 * np.pi), "k--", lw=2.2, label="真值 $N(0,1)$")
ax.set_xlim(-5, 5)
ax.set_xlabel("$x$(σ=0 终点)", fontsize=10.5); ax.set_ylabel("密度", fontsize=10.5)
ax.set_title("(a) 补偿系数 $c$ 取错 → 终点分布变形", fontsize=11.5)
ax.legend(fontsize=9, frameon=False)
ax.tick_params(labelsize=9)

ax2b = axes2[1]
csvals = np.round(np.arange(0.0, 2.01, 0.1), 2)
stds = []
for c in csvals:
    gen = np.random.default_rng(7)
    stds.append(run(c, gen.normal(size=6000), gen)[-1].std())
ax2b.plot(csvals, stds, "o-", color=BLUE, lw=2.0, ms=4.5)
ax2b.axhline(1.0, color=GREEN, ls="--", lw=1.6)
ax2b.axvline(1.0, color=GREEN, ls=":", lw=1.6)
ax2b.text(1.02, 1.42, "$c=1$:精确补偿\n$\\Rightarrow$ std = 真值", fontsize=9.6, color=GREEN)
ax2b.text(0.03, 1.62, "补偿不足 → 分布被抹宽", fontsize=9.4, color=RED)
ax2b.text(1.55, 0.80, "补偿过度 → 分布被压窄", fontsize=9.4, color=PURPLE)
ax2b.set_xlabel("补偿系数 $c$(漂移里用 $c\\cdot\\frac{g^2}{2}\\nabla\\log p$)", fontsize=10.2)
ax2b.set_ylabel("终点 std(真值 = 1.000)", fontsize=10.2)
ax2b.set_title("(b) std 随 $c$ 单调变化,只有 $c=1$ 落在真值上", fontsize=11.5)
ax2b.tick_params(labelsize=9)

ax3 = axes2[2]; ax3.axis("off")
tbl = [["$c$", "std", "结论"]]
for c in [0.0, 0.5, 1.0, 1.5, 2.0]:
    s = res[c].std() if c in res else run(c, np.random.default_rng(5).normal(size=6000), np.random.default_rng(5))[-1].std()
    tbl.append([f"{c:g}", f"{s:.4f}", "✓ 正确" if c == 1.0 else ("被抹宽" if s > 1.005 else "被压窄")])
tb2 = ax3.table(cellText=tbl, cellLoc="center", bbox=[0.0, 0.62, 1.0, 0.34])
tb2.auto_set_font_size(False); tb2.set_fontsize(9.2)
for (r, c_), cell in tb2.get_celld().items():
    if r == 0:
        cell.set_facecolor("#eef2f7")
    if r == 3:
        cell.set_facecolor("#eef7f1")
ax3.text(0.0, 0.55,
         "结论:那一项不是「随手加的偏移」,\n而是噪声的【精确补偿量】。\n\n"
         "少补($c<1$):噪声把分布抹宽\n(不补时 std $1.645$,宽了 64%)\n"
         "多补($c>1$):把分布压窄($c=2$ 时 0.72)",
         fontsize=9.4, color=DARK, va="top",
         bbox=dict(fc="#fdecea", ec="#e6a9a0"))
fig2.suptitle("回到你的担心:偏移会不会把目的地搞错?—— 补偿量错就会,补偿对了就不会", fontsize=12.6, y=0.99)
fig2.savefig(RF + "fg_compensation_coef.png", bbox_inches="tight")

print("Fig B 数值(c 为补偿系数,真值 std = 1.0000):")
for c in [0.0, 0.5, 1.0, 1.5, 2.0]:
    e = res[c] if c in res else run(c, np.random.default_rng(5).normal(size=6000), np.random.default_rng(5))[-1]
    print(f"  c={c:<4}: 均值 {e.mean():+.4f}  std {e.std():.4f}")
print("  std 曲线采样点:", ", ".join(f"{c:g}:{s:.3f}" for c, s in list(zip(csvals, stds))[::5]))
print("wrote", RF + "fg_ode_vs_sde_basic.png"); print("wrote", RF + "fg_compensation_coef.png")
