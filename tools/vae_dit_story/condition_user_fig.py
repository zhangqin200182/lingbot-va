r"""在"点看到不同 x₀ 密度"的模型里,条件 c 是什么?

  你的模型:固定 (x_t, t),这个点"看到"若干候选 x₀,各有权重
       w(x₀|x_t) = 先验(x₀) × 高斯(x_t;(1−t)x₀,t²I)
                    └ ① ┘      └ ② 几何项 ┘

  ★ 条件只改 ①,不改 ②:
       无条件: 先验 = p_data(x₀)
       有条件: 先验 = p_data(x₀|c)
       高斯项【完全相同】—— 因为几何(离 x_t 多近)与语言无关!

  ⇒ 同一个点、同一组高斯,换先验表 ⇒ 后验均值变 ⇒ 下一步方向变
  数值:固定 x_t=0.4, t=0.5
       E[x₀|x_t]   = +0.452  → 位移 (E−x)/t = +0.104
       E[x₀|x_t,c] = +0.812  → 位移 (E−x)/t = +0.823   (差 8 倍)

  ⇒ 条件的主导权随 t 衰减:
       t=0.9: |ΔE|=0.602   t=0.5: 0.360   t=0.15: 0.006   t=0.05: 0.000
     因为 t 小时高斯项极尖 ⇒ 权重被"离得最近"独占 ⇒ 先验(语言)几乎无用
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

x0 = np.array([-1.0, -0.2, 0.6, 1.0])
prior_u = np.array([0.25, 0.25, 0.25, 0.25])
prior_c = np.array([0.05, 0.05, 0.30, 0.60])


def weights(xt, t, prior):
    g = np.exp(-(xt - (1 - t) * x0) ** 2 / (2 * t ** 2))
    w = prior * g
    return w / w.sum(), g


xt, t0 = 0.4, 0.5
wu, g = weights(xt, t0, prior_u)
wc, _ = weights(xt, t0, prior_c)
Eu, Ec = wu @ x0, wc @ x0

fig = plt.figure(figsize=(15.2, 5.7), dpi=150)

# ---------------- (a) 固定点:两套权重
ax = fig.add_axes([0.05, 0.16, 0.40, 0.68])
idx = np.arange(len(x0))
ax.bar(idx - 0.19, wu, width=0.36, color=BLUE, label="无条件:权重 = $p_{\\rm data}$ × 高斯")
ax.bar(idx + 0.19, wc, width=0.36, color=ORANGE, label="有条件:权重 = $p_{\\rm data}(\\cdot|c)$ × 高斯")
ax.plot(idx, g / g.sum(), "k*", ms=13, label="几何项(高斯),两条件【完全相同】", zorder=6)
ax.set_xticks(idx); ax.set_xticklabels([f"$x_0^{{({i+1})}}$={v:+.1f}" for i, v in enumerate(x0)], fontsize=9)
ax.set_ylabel("归一化权重", fontsize=10.2)
ax.set_title(f"(a) 固定点 $x_t={xt}$,时刻 $t={t0}$:\n条件只换【先验表】,几何项不动", fontsize=11.2)
ax.legend(fontsize=8.0, frameon=False, loc="center left")
ax.tick_params(labelsize=9)
ax.axhline(0, color=GREY, lw=0.8)
ax.text(0.5, 0.99, f"$E[x_0|x_t]$: {Eu:+.3f} → {Ec:+.3f}\n"
                   f"位移 $(E-x_t)/t$: {((Eu-xt)/t0):+.3f} → {((Ec-xt)/t0):+.3f}$(差 8 倍)",
        fontsize=9.0, color=DARK, transform=ax.transAxes, ha="center", va="top",
        bbox=dict(fc="#fffbe6", ec="#e0c060"))

# ---------------- (b) 条件主导权随 t
ax2 = fig.add_axes([0.52, 0.16, 0.24, 0.68])
ts = np.linspace(0.03, 0.97, 300)
dE = []
for tt in ts:
    a, _ = weights(xt, tt, prior_u)
    b, _ = weights(xt, tt, prior_c)
    dE.append(abs(b @ x0 - a @ x0))
ax2.plot(ts, dE, color=GREEN, lw=2.8)
for tt, lab in [(0.9, "t=0.9"), (0.5, "t=0.5"), (0.15, "t=0.15 ≈ 0")]:
    a, _ = weights(xt, tt, prior_u); b, _ = weights(xt, tt, prior_c)
    ax2.plot([tt], [abs(b @ x0 - a @ x0)], "o", ms=7, color=RED)
    ax2.annotate(lab, xy=(tt, abs(b @ x0 - a @ x0)), xytext=(tt - 0.05, abs(b @ x0 - a @ x0) + 0.055),
                 fontsize=8.4, color=RED)
ax2.set_xlabel("噪声水平 $t$", fontsize=10.2)
ax2.set_ylabel("$|E[x_0|x_t,c]-E[x_0|x_t]|$", fontsize=10.0)
ax2.set_title("(b) 条件的\"话语权\"随 $t$ 衰减\n(同一个点,先验差被高斯项压掉)", fontsize=11.0)
ax2.tick_params(labelsize=9)
ax2.text(0.97, 0.06, "高噪声:语言主导结构\n低噪声:采样路径主导细节", fontsize=8.4,
         color=DARK, va="bottom", ha="right", transform=ax2.transAxes,
         bbox=dict(fc="#eef7f1", ec="#a9d5bb"))

# ---------------- (c) 结论
ax3 = fig.add_axes([0.80, 0.10, 0.185, 0.80]); ax3.axis("off")
ax3.text(0.5, 1.00, "(c) 用你的语言说", ha="center", va="top", fontsize=11.4, color=DARK)
ax3.text(0.0, 0.92,
         "你的模型只需\n把一个符号换掉:\n\n"
         "$p(x_0|x_t)$\n"
         "　↓\n"
         "$p(x_0|x_t,c)$\n\n"
         "而 $c$ 只动了\n分子里的【先验】。",
         fontsize=9.0, color=BLUE, va="top", bbox=dict(fc="#eaf2fa", ec=BLUE))
ax3.text(0.0, 0.60,
         "于是:\n\n"
         "· 遵从性 = 候选集合\n"
         "　【收缩】(猫/狗/房子 →\n"
         "　 只剩红立方体)\n\n"
         "· 多样性 = 候选集合\n"
         "　不是一个点(集合内\n"
         "　仍有多个盆地)",
         fontsize=8.8, color=GREEN, va="top", bbox=dict(fc="#eef7f1", ec=GREEN))
ax3.text(0.0, 0.20,
         "所以 c 不是\n\"在空间里加一支箭头\",\n而是\n\"换掉这块空间上\n哪些 x₀ 有资格\n当候选、各占多重\"。",
         fontsize=8.6, color=RED, va="top", bbox=dict(fc="#fdecea", ec="#e6a9a0"))

fig.suptitle("在\"点看到不同 x₀ 密度\"的模型里,条件 c = 只换【先验表】;几何项(高斯)与 c 无关 —— 所以同一个点、不同方向",
             fontsize=11.0, y=0.99)
fig.savefig(RF + "fg_condition_in_user_model.png", bbox_inches="tight")
print("wrote", RF + "fg_condition_in_user_model.png")
