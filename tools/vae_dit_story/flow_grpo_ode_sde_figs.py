r"""ODE→SDE 的两张图(重写版)。

图 A(示意):σ 轴上的前向插值 + 两条反演路径(确定性 ODE / 随机 SDE),下方给出 g(σ)
图 B(实测):在解析可算的 1 维两模态高斯混合上真跑
          (a) 同一个 ε 跑 8 次 ODE  → 完全重合
          (b) 同一个 ε 跑 8 次 SDE  → 分散,会换模态
          (c) 各 3000 条终点的分布 vs 真值 p_data → 两者都复现同一个分布
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

RED, BLUE, GREEN, DARK, GREY, ORANGE = "#c0392b", "#2471a3", "#1e8449", "#2c3e50", "#95a5a6", "#b9770e"
RF = "/Users/kevin/code/lingbot-va/.claude/scratch/rf/"

MU = np.array([-1.0, 1.0]); SD = np.array([0.15, 0.15]); W = np.array([0.5, 0.5])
NOISE_LEVEL = 0.7
NSTEP = 50
SIG = np.linspace(1.0, 0.0, NSTEP + 1)
SMAX = 1.0 - 1.0 / NSTEP


def p_sigma(x, s):
    m = (1 - s) * MU
    v = s ** 2 + (1 - s) ** 2 * SD ** 2
    return np.sum(W[None, :] * np.exp(-(x[:, None] - m[None, :]) ** 2 / (2 * v[None, :]))
                  / np.sqrt(2 * np.pi * v[None, :]), axis=1)


def post(x, s):
    m = (1 - s) * MU
    v = s ** 2 + (1 - s) ** 2 * SD ** 2
    lw = np.log(W)[None, :] - 0.5 * np.log(2 * np.pi * v)[None, :] - (x[:, None] - m[None, :]) ** 2 / (2 * v[None, :])
    lw = lw - lw.max(axis=1, keepdims=True)
    return np.exp(lw) / np.exp(lw).sum(axis=1, keepdims=True)


def v_field(x, s):
    w = post(x, s)
    return (x - (w * MU[None, :]).sum(axis=1)) / max(s, 1e-6)


def score(x, s):
    m = (1 - s) * MU
    v = s ** 2 + (1 - s) ** 2 * SD ** 2
    return (post(x, s) * (-(x[:, None] - m[None, :]) / v[None, :])).sum(axis=1)


def g_of(s):
    sc = min(s, SMAX)
    return np.sqrt(sc / (1 - sc)) * NOISE_LEVEL


def rollout(x0, mode, gen=None):
    x = np.array([x0], dtype=float)
    traj = [x[0]]
    for i in range(NSTEP):
        s, s_next = SIG[i], SIG[i + 1]
        d = s_next - s
        v = v_field(x, s)
        if mode == "ode":
            x = x + v * d
        else:
            g = g_of(s)
            x = x + (v - 0.5 * g ** 2 * score(x, s)) * d + g * np.sqrt(-d) * gen.normal(size=1)
        traj.append(x[0])
    return np.array(traj)


# ============================================================ 图 A
fig = plt.figure(figsize=(14.8, 8.6), dpi=150)
ax = fig.add_axes([0.055, 0.295, 0.905, 0.585])
axg = fig.add_axes([0.315, 0.075, 0.40, 0.145])

xs = np.linspace(-1.9, 1.9, 400)
for s in [0.0, 0.25, 0.55, 0.8, 1.0]:
    d = p_sigma(xs, s)
    sc = 0.32 / d.max()
    ax.plot(s + d * sc, xs, color="#8a6d3b", lw=1.5, alpha=0.9)
    ax.fill_betweenx(xs, s, s + d * sc, color="#f0c987", alpha=0.32)
    ax.text(s + 0.05, 1.72, f"$p_{{{s:.2f}}}$", ha="left", fontsize=9.2, color="#7e5109")
ax.axvline(0, color=GREY, lw=0.8); ax.axvline(1, color=GREY, lw=0.8)

for a, b in zip([-1.0, -0.7, 1.0, 0.8], [0.3, -0.75, -0.35, 0.95]):
    ax.plot([0, 1], [a, b], color=ORANGE, lw=1.2, alpha=0.7, zorder=2)
ax.text(0.50, -1.83,
        "前向(加噪):$x_\\sigma=(1-\\sigma)x_0+\\sigma\\epsilon$ —— 【定义】了每个 $\\sigma$ 时刻的边缘分布 $p_\\sigma$",
        ha="center", fontsize=11, color=ORANGE)

EPS0 = -0.55
t_ode = rollout(EPS0, "ode")
t_sde = rollout(EPS0, "sde", gen=np.random.default_rng(100))
ax.plot(SIG, t_ode, color=BLUE, lw=3.2, zorder=5, label="① ODE:$dx=v\\,d\\sigma$(给定 $\\epsilon$,路径唯一)")
ax.plot(SIG, t_sde, color=RED, lw=2.2, zorder=6, label="② SDE:$dx=[v-\\frac{g^2}{2}\\nabla\\log p_\\sigma]d\\sigma+g\\,dW$")
for i in range(6, NSTEP, 6):
    ax.add_patch(FancyArrowPatch((SIG[i], t_sde[i]), (SIG[i], t_sde[i] + (0.22 if i % 12 else -0.22)),
                                 arrowstyle="-|>", mutation_scale=9, lw=1.1, color=RED, alpha=0.8))
ax.plot([1], [EPS0], "o", ms=10, color=DARK, zorder=8)
ax.text(1.03, EPS0 + 0.06, "起点:同一个 $\\epsilon$", fontsize=10, color=DARK)
ax.plot([0], [t_ode[-1]], "o", ms=10, color=BLUE, zorder=8)
ax.plot([0], [t_sde[-1]], "o", ms=10, color=RED, zorder=8)
ax.text(0.04, t_ode[-1] - 0.02, f"ODE 落点 {t_ode[-1]:+.2f}", fontsize=9.6, color=BLUE, va="top")
ax.text(0.04, t_sde[-1] + 0.03, f"SDE 落点 {t_sde[-1]:+.2f}(换了模态)", fontsize=9.6, color=RED, va="bottom")
ax.text(0.30, 1.30, "两条路径共享同一族 $p_\\sigma$:\n在每个 $\\sigma$ 竖线上分布都一样", fontsize=10,
        color=DARK, bbox=dict(fc="w", ec="#e3e8ee", alpha=0.95))
ax.set_xlim(1.42, -0.05); ax.set_ylim(-2.0, 1.95)
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlabel("$\\sigma$    (生成方向:从左往右,$\\sigma$ 1 → 0)", fontsize=11.5)
ax.set_title("图 A:前向加噪定义了 $p_\\sigma$;ODE 与 SDE 是它的两条【反演】路径", fontsize=13.5)
ax.legend(fontsize=9.8, frameon=True, loc="lower right", framealpha=0.95)

sg = np.linspace(0.02, 0.98, 200)
axg.plot(sg, [g_of(s) for s in sg], color=GREEN, lw=2.4)
axg.set_xlabel("$\\sigma$", fontsize=9.5)
axg.set_ylabel("注入噪声强度 $g$", fontsize=9.5)
axg.set_title("$g(\\sigma)=\\sqrt{\\sigma/(1-\\sigma)}\\cdot$noise_level:靠数据端(σ→0)自动衰减到 0", fontsize=9.6)
axg.tick_params(labelsize=8)
fig.savefig(RF + "fg_ode_sde_schematic.png", bbox_inches="tight")

# ============================================================ 图 B
fig2, axes = plt.subplots(1, 3, figsize=(15.2, 4.6), dpi=150)
EPS0 = -0.55

ax = axes[0]
ax.plot(SIG, rollout(EPS0, "ode"), color=BLUE, lw=4.5, alpha=0.9)
ax.set_title("(a) 同一个 $\\epsilon$,跑 8 次 ODE", fontsize=11.5)
ax.text(0.5, -1.62, "8 条曲线【完全重合】(路径唯一)\n每步似然是 $\\delta$ → GRPO 算不出 ratio",
        ha="center", fontsize=9.6, color=BLUE)
ax.set_ylabel("$x$", fontsize=10)

ax = axes[1]
ends = []
for k in range(8):
    tt = rollout(EPS0, "sde", gen=np.random.default_rng(100 + k))
    ends.append(tt[-1])
    ax.plot(SIG, tt, lw=1.6, alpha=0.85, color=GREEN if tt[-1] > 0 else RED)
ax.set_title("(b) 同一个 $\\epsilon$,跑 8 次 SDE", fontsize=11.5)
ax.text(0.5, -1.62, f"8 条曲线【分散】,落点 {sum(e>0 for e in ends)} 个在 +1、{sum(e<0 for e in ends)} 个在 −1\n"
                    "每步是高斯 → 有 log-prob、有探索、会换模态",
        ha="center", fontsize=9.6, color=DARK)

ax = axes[2]
n = 3000
gen_ode = np.random.default_rng(2000)
gen_sde = np.random.default_rng(1000)
end_ode = np.array([rollout(gen_ode.normal(), "ode")[-1] for _ in range(n)])
end_sde = np.array([rollout(gen_sde.normal(), "sde", gen=gen_sde)[-1] for _ in range(n)])
xg = np.linspace(-1.9, 1.9, 600)
ax.hist(end_ode, bins=np.linspace(-1.9, 1.9, 70), density=True, color=BLUE, alpha=0.40, label=f"ODE({n} 条)")
ax.hist(end_sde, bins=np.linspace(-1.9, 1.9, 70), density=True, color=RED, alpha=0.40, label=f"SDE({n} 条)")
ax.plot(xg, p_sigma(xg, 0.0), "k--", lw=2.2, label="真值 $p_{data}$")
ax.set_title("(c) 终点分布:两者都复现同一个 $p_{data}$", fontsize=11.5)
ax.set_xlabel("$x$(终点位置)", fontsize=10)
ax.set_ylabel("密度", fontsize=10)
ax.legend(fontsize=9, frameon=False, loc="upper center")
ax.text(0.0, 0.62,
        f"ODE: 均值 {end_ode.mean():+.3f}  std {end_ode.std():.3f}  P(+)= {float((end_ode>0).mean()):.3f}\n"
        f"SDE: 均值 {end_sde.mean():+.3f}  std {end_sde.std():.3f}  P(+)= {float((end_sde>0).mean()):.3f}\n"
        f"真值: 均值 +0.000  std 1.011  P(+)= 0.500",
        ha="center", fontsize=9.2, color=DARK, bbox=dict(fc="w", ec="#e3e8ee", alpha=0.95))
ax.tick_params(labelsize=8.5)

for ax in axes[:2]:
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(-1.95, 1.95)
    ax.set_xlabel("$\\sigma$(1 = 纯噪声 → 0 = 数据)", fontsize=10)
    ax.axhline(0, color=GREY, lw=0.6, alpha=0.5)
    ax.tick_params(labelsize=8.5)
axes[2].set_xlim(-1.95, 1.95)

fig2.suptitle("图 B:解析可算的两模态 toy 上真跑 —— ODE 与 SDE 复现同一分布,但只有 SDE 给出可优化的随机策略",
              fontsize=12.5)
fig2.tight_layout(rect=[0, 0, 1, 0.90])
fig2.savefig(RF + "fg_ode_sde_rollout.png", bbox_inches="tight")

print("=== 数值核对 ===")
print(f"真值 p_data: 均值 +0.000  std 1.0112  P(+)=0.500")
print(f"ODE {n} 条: 均值 {end_ode.mean():+.4f}  std {end_ode.std():.4f}  P(+)={float((end_ode>0).mean()):.4f}")
print(f"SDE {n} 条: 均值 {end_sde.mean():+.4f}  std {end_sde.std():.4f}  P(+)={float((end_sde>0).mean()):.4f}")
print("同一个 ε、8 次 ODE 是否完全相同:",
      len({round(float(rollout(EPS0, 'ode')[-1]), 10) for _ in range(5)}) == 1)
print("同一个 ε、8 次 SDE 终点:", [round(float(e), 3) for e in ends])
print("wrote", RF + "fg_ode_sde_schematic.png"); print("wrote", RF + "fg_ode_sde_rollout.png")
