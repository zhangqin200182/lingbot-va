r"""Step 3 在干什么:证明 ODE dx/dσ = v 的边缘分布就是 p_σ。

(a) 概念图:插值路径是直线,ODE 轨迹是曲线 —— 【轨迹不同,但每条竖线上的分布相同】
(b) 数值验证:σ=0.5 处 ODE 点的直方图 vs 解析 p_σ
(c) 数值验证:σ=0 处(采样终点)直方图 vs 真值 p_data  ← 这才是"能拿来采样"的依据
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
MU = np.array([1.0, -1.0]); PR = np.array([0.5, 0.5]); NSTEP = 200
SIG = np.linspace(1.0, 0.0, NSTEP + 1)


def p_sig(x, s):
    m = (1 - s) * MU
    return np.sum([PR[k] * np.exp(-(x - m[k]) ** 2 / (2 * s ** 2)) / np.sqrt(2 * np.pi * s ** 2)
                   for k in range(2)], axis=0)


def wts(x, s):
    m = (1 - s) * MU
    lw = np.log(PR) - 0.5 * np.log(2 * np.pi * s ** 2) - (x - m) ** 2 / (2 * s ** 2)
    lw -= lw.max()
    w = np.exp(lw)
    return w / w.sum()


def v_field(x, s):
    w = wts(x, s)
    return (x - float(w @ MU)) / max(s, 1e-6)


def ode_traj(x0):
    x = x0; tr = [x]
    for i in range(NSTEP):
        s, sn = SIG[i], SIG[i + 1]
        x = x + v_field(x, s) * (sn - s)
        tr.append(x)
    return np.array(tr)


rng = np.random.default_rng(0)
NT = 20000
starts = rng.normal(size=NT)
trajs = np.array([ode_traj(s) for s in starts[:20]])
alls = np.array([ode_traj(s) for s in starts])

fig = plt.figure(figsize=(15.0, 5.6), dpi=150)

# ---------------- (a)
ax = fig.add_axes([0.05, 0.16, 0.31, 0.68])
xs = np.linspace(-2.2, 2.2, 400)
for s in [1.0, 0.7, 0.45, 0.2, 0.0]:
    d = p_sig(xs, s); sc = 0.30 / d.max()
    ax.plot(s + d * sc, xs, color="#8a6d3b", lw=1.3, alpha=0.85)
    ax.fill_betweenx(xs, s, s + d * sc, color="#f0c987", alpha=0.30)
for k in range(6):
    e = rng.normal()
    x0 = rng.normal()
    ax.plot([0, 1], [x0, e], color=GREY, lw=1.0, alpha=0.75)
ax.plot(SIG, trajs[0], color=BLUE, lw=2.2)
for k in range(1, 8):
    ax.plot(SIG, trajs[k], color=BLUE, lw=1.4, alpha=0.85)
ax.text(0.50, -2.35, "灰线 = 插值路径(直线,用真实 $(x_0,\\epsilon)$ 画)\n"
                     "蓝线 = ODE 轨迹(曲线,只知道 $v$)—— 两者【轨迹不同】",
        ha="center", fontsize=9.8, color=DARK)
ax.text(0.02, 2.05, "每条竖线上的分布相同", fontsize=10, color=GREEN)
ax.set_xlim(-0.05, 1.42); ax.set_ylim(-2.6, 2.3)
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlabel("$\\sigma$(1 → 0)", fontsize=10.5)
ax.set_title("(a) 轨迹不同,分布相同", fontsize=11.5)

# ---------------- (b)(c)
for k, (S, lab) in enumerate([(0.5, "(b) σ=0.5:轨迹中段"), (0.0, "(c) σ=0:采样终点")]):
    ax2 = fig.add_axes([0.43 + k * 0.29, 0.16, 0.245, 0.68])
    idx = int(np.argmin(np.abs(SIG - S)))
    pts = alls[:, idx]
    ax2.hist(pts, bins=np.linspace(-2.4, 2.4, 120), density=True, color=BLUE, alpha=0.45,
             label=f"ODE 终点({NT} 条)")
    ax2.plot(xs, p_sig(xs, S), "k--", lw=2.0, label="解析 $p_\\sigma$")
    ax2.set_xlim(-2.3, 2.3)
    ax2.set_xlabel("$x$", fontsize=10.5)
    if k == 0:
        ax2.set_ylabel("密度", fontsize=10.5)
        ax2.legend(fontsize=8.6, frameon=False)
    ax2.set_title(lab, fontsize=11.5)
    ax2.tick_params(labelsize=9)
    m_ode, s_ode = pts.mean(), pts.std()
    if S > 0:
        grid = np.linspace(-6, 6, 4000); pdf = p_sig(grid, S)
        m_th = float(np.trapz(grid * pdf, grid) / np.trapz(pdf, grid))
        s_th = float(np.sqrt(np.trapz((grid - m_th) ** 2 * pdf, grid) / np.trapz(pdf, grid)))
    else:
        m_th, s_th = 0.0, 1.0
    ax2.text(0.5, 0.88, f"ODE:均值 {m_ode:+.3f} std {s_ode:.3f}\n解析:均值 {m_th:+.3f} std {s_th:.3f}",
             transform=ax2.transAxes, fontsize=9.2, color=DARK, ha="center", va="top",
             bbox=dict(fc="#eef7f1", ec="#a9d5bb"))

fig.suptitle("Step 3 在证什么:ODE 采样器的每条竖线分布都等于 $p_\\sigma$(所以 $\\sigma=0$ 时就是数据分布)",
             fontsize=13, y=0.995)
fig.savefig(RF + "fg_ode_distribution.png", bbox_inches="tight")

print("数值核对(20000 条 ODE 轨迹):")
for S in [0.9, 0.7, 0.5, 0.3, 0.1, 0.0]:
    idx = int(np.argmin(np.abs(SIG - S)))
    pts = alls[:, idx]
    if S > 0:
        grid = np.linspace(-6, 6, 4000); pdf = p_sig(grid, S)
        m_th = float(np.trapz(grid * pdf, grid) / np.trapz(pdf, grid))
        s_th = float(np.sqrt(np.trapz((grid - m_th) ** 2 * pdf, grid) / np.trapz(pdf, grid)))
    else:
        m_th, s_th = 0.0, 1.0
    print(f"  σ={S:.1f}: ODE 均值 {pts.mean():+.4f} std {pts.std():.4f} | 解析 均值 {m_th:+.4f} std {s_th:.4f}")
print(f"  σ=0 落到 +1 模态的比例: {float((alls[:, -1] > 0).mean()):.4f}(真值 0.5)")
print("wrote", RF + "fg_ode_distribution.png")
