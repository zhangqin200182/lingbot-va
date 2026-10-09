r"""专门讲那一步:为什么 v* 是条件期望,以及 (x − E[x0])/σ 是怎么化简出来的。

图 (a) 一个 (x0, ε) 对:速度 ε−x0 与 (x−x0)/σ 是同一个数(数值代入验证)
图 (b) 同一个 (x, σ) 上,两个数据点给出【方向相反】的两个速度 → 必须取后验加权平均
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

fig = plt.figure(figsize=(14.6, 6.6), dpi=150)

# ---------------- (a) 一个对:ε−x0 与 (x−x0)/σ 同一个数
ax = fig.add_axes([0.055, 0.12, 0.42, 0.72])
x0a, epsa, sa = 1.0, -0.5, 0.25
xA = (1 - sa) * x0a + sa * epsa
sg = np.linspace(0, 1, 2)
ax.plot(sg, [(1 - s) * x0a + s * epsa for s in sg], color=DARK, lw=2.4, zorder=3)
ax.plot([0], [x0a], "o", ms=12, color=BLUE, zorder=5)
ax.text(0.02, x0a + 0.06, f"$x_0={x0a:+.2f}$(数据)", fontsize=10.5, color=BLUE)
ax.plot([1], [epsa], "o", ms=12, color=RED, zorder=5)
ax.text(1.02, epsa - 0.12, f"$\\epsilon={epsa:+.2f}$(噪声)", fontsize=10.5, color=RED, ha="right")
ax.axvline(sa, color=GREY, ls="--", lw=1.2)
ax.plot([sa], [xA], "o", ms=12, color=GREEN, zorder=6)
ax.text(sa + 0.02, xA + 0.10, f"$x_\\sigma={xA:.3f}$", fontsize=10.5, color=GREEN)
vA = epsa - x0a
ax.add_patch(FancyArrowPatch((sa, xA), (sa + 0.16, xA + vA * 0.16), arrowstyle="-|>",
                             mutation_scale=16, lw=2.6, color="#7e5109"))
ax.text(sa + 0.19, xA + vA * 0.16, "$v=\\epsilon-x_0=-1.5$\n(每单位 $\\sigma$ 的位移)", fontsize=9.6,
        color="#7e5109", va="center")
ax.text(0.5, -0.93,
        f"$\\epsilon-x_0=({epsa:+.2f})-({x0a:+.2f})=-1.50$\n"
        f"$(x_\\sigma-x_0)/\\sigma=({xA:.3f}-{x0a:+.2f})/{sa}=({-0.375:.3f})/{sa}=-1.50$",
        ha="center", fontsize=11, color=DARK,
        bbox=dict(fc="#fffbe6", ec="#e0c060", alpha=1.0))
ax.set_xlim(-0.06, 1.14); ax.set_ylim(-1.05, 1.35)
ax.set_xlabel("$\\sigma$", fontsize=11); ax.set_ylabel("$x$", fontsize=11)
ax.set_title("(a) 一个 $(x_0,\\epsilon)$ 对:两个式子是【同一个数】", fontsize=12)
ax.tick_params(labelsize=9)

# ---------------- (b) 同一个 (x,σ) 上两个数据点 → 方向相反的两个速度
ax2 = fig.add_axes([0.555, 0.12, 0.42, 0.72])
XD, SD_ = 0.1, 0.5
pts = [1.0, -1.0]
prior = 0.5
eps = [(XD - (1 - SD_) * p) / SD_ for p in pts]          # ε = (x − (1−σ)x0)/σ
lik = [np.exp(-e ** 2 / 2) for e in eps]                  # N(ε;0,1)
w = np.array([prior * l for l in lik]); w = w / w.sum()
v = np.array([e - p for e, p in zip(eps, pts)])           # ε − x0
v_star = float(w @ v)
ex0 = float(w @ np.array(pts))
ax2.axhline(0, color=GREY, lw=0.8, alpha=0.6)
ax2.plot([XD], [0], "o", ms=13, color=GREEN, zorder=6)
ax2.text(XD, 0.14, f"当前点 $x_\\sigma={XD}$", fontsize=11, color=GREEN, ha="center")
for p, e, vv, ww in zip(pts, eps, v, w):
    ax2.plot([p], [0], "o", ms=11, color=BLUE, zorder=5)
    ax2.text(p, -0.20, f"$x_0={p:+.0f}$", fontsize=10, color=BLUE, ha="center")
    ax2.add_patch(FancyArrowPatch((XD, 0.30 if vv < 0 else -0.30),
                                  (XD + vv * 0.28, 0.30 if vv < 0 else -0.30),
                                  arrowstyle="-|>", mutation_scale=15, lw=3.0,
                                  color=RED if vv < 0 else ORANGE))
    ax2.text(XD + vv * 0.28 + (0.04 if vv < 0 else -0.04), 0.36 if vv < 0 else -0.36,
             f"$\\epsilon={e:+.2f}\\Rightarrow v={vv:+.2f}$\n权重 ${ww:.3f}$",
             fontsize=9.4, color=RED if vv < 0 else ORANGE,
             ha="right" if vv < 0 else "left", va="bottom" if vv < 0 else "top")
ax2.add_patch(FancyArrowPatch((XD, -0.02), (XD + v_star * 0.28, -0.02), arrowstyle="-|>",
                              mutation_scale=17, lw=3.4, color=GREEN, zorder=7))
ax2.text(XD + v_star * 0.28, -0.10, f"加权平均 $v^*={v_star:+.3f}$", fontsize=10.5,
         color=GREEN, ha="left", va="top")
ax2.text(XD - 1.85, 0.86,
         "同一个 $(x_\\sigma,\\sigma)$ 上,两个数据点给出**方向相反**的速度\n"
         "而一个确定性场只能输出一个数 → 只能取【后验加权平均】\n"
         f"权重 $\\propto$ 先验 $\\times$ 似然:$0.5\\cdot{lik[0]:.3f}$ 与 $0.5\\cdot{lik[1]:.3f}$ → ${w[0]:.3f}/{w[1]:.3f}$\n"
         f"两种算法一致:$\\sum w_k v_k={v_star:+.3f}$ 与 $(x_\\sigma-E[x_0])/\\sigma=({XD}-{ex0:+.3f})/{SD_}={((XD-ex0)/SD_):+.3f}$",
         fontsize=9.6, color=DARK, va="top",
         bbox=dict(fc="#eef7f1", ec="#a9d5bb", alpha=1.0))
ax2.set_xlim(-1.95, 1.95); ax2.set_ylim(-0.95, 0.95)
ax2.set_yticks([]); ax2.set_xlabel("$x$", fontsize=11)
ax2.set_title("(b) 同一个 $(x_\\sigma,\\sigma)$ 上,必须取加权平均", fontsize=12)
ax2.tick_params(labelsize=9)

fig.suptitle("那一步在做什么:$v^*$ 为什么是条件期望,以及它为什么等于 $(x_\\sigma-E[x_0|x_\\sigma])/\\sigma$",
             fontsize=13)
fig.savefig(RF + "fg_vstar_step.png", bbox_inches="tight")

print("(a) 验证: ε−x0 =", epsa - x0a, " ; (x−x0)/σ =", (xA - x0a) / sa)
print(f"(b) x_σ={XD}, σ={SD_}")
for p, e, vv, ww in zip(pts, eps, v, w):
    print(f"    x0={p:+.1f}: ε={e:+.4f}  似然={np.exp(-e**2/2):.4f}  权重={ww:.4f}  速度 v={vv:+.4f}")
print(f"    加权平均 v* = Σ w_k v_k = {v_star:+.4f}")
print(f"    E[x0|x] = {ex0:+.4f} ; (x−E[x0])/σ = {(XD-ex0)/SD_:+.4f}   ← 与上式一致")
print("wrote", RF + "fg_vstar_step.png")
