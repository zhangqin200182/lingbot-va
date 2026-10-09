r"""组内基线到底扣掉了什么、扣不掉什么。

把奖励拆成三段:
  R_i = 难度(prompt 固有,全组共享) + Δ_i(这个样本的真本事) + η_i(这条轨迹的运气)
  R̄   = 难度 + mean(Δ) + mean(η)
  A_i ∝ R_i − R̄ = (Δ_i − mean Δ)  +  (η_i − mean η)
                  └ 保留:要的信号 ┘   └ 残留:个体运气,扣不掉 ┘

(a) 三段分解:减掉组内平均之后剩下什么
(b) 同样的 Δ、不同的 η(两次采样)⇒ 优势不同 ⇒ 残留运气 = 估计方差
(c) 为什么无偏 + 残留方差随 G 的变化 + 代码里的降方差措施
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

DIFF = 5.0                                   # prompt 固有难度(全组共享)
D = np.array([1.0, -1.0, 0.5, -0.5])         # 真本事 Δ
ETA1 = np.array([0.2, 0.3, -0.1, 0.1])       # 第一次采样的个体运气
ETA2 = np.array([-0.4, 0.5, 0.2, -0.3])      # 第二次采样的个体运气
R1 = DIFF + D + ETA1
R2 = DIFF + D + ETA2
A1 = (R1 - R1.mean()) / (R1.std() + 1e-4)
A2 = (R2 - R2.mean()) / (R2.std() + 1e-4)

fig = plt.figure(figsize=(15.2, 5.6), dpi=150)

# ---------------- (a) 三段分解
ax = fig.add_axes([0.05, 0.17, 0.33, 0.66])
x = np.arange(4)
ax.bar(x, DIFF, width=0.5, color=GREY, label="难度(prompt 固有,全批共享)")
ax.bar(x, D, bottom=DIFF, width=0.5, color=GREEN, label="$\\Delta_i$ 这个样本的真本事")
ax.bar(x, ETA1, bottom=DIFF + D, width=0.5, color=ORANGE, label="$\\eta_i$ 这条轨迹的运气")
ax.axhline(R1.mean(), color=RED, ls="--", lw=2.0)
ax.text(3.45, R1.mean() + 0.06, f"组内平均 $\\bar R={R1.mean():.3f}$", fontsize=9.2, color=RED, ha="right")
for i in x:
    ax.text(i, R1[i] + 0.07, f"{R1[i]:.2f}", ha="center", fontsize=9.0, color=DARK)
ax.set_xticks(x); ax.set_xticklabels([f"样本{i+1}" for i in x], fontsize=9.4)
ax.set_ylim(4.2, 7.0)
ax.set_ylabel("奖励 $R_i$", fontsize=10.5)
ax.set_title("(a) 奖励 = 难度 + 真本事 + 运气\n减掉组内平均 ⇒ 扣掉难度与共同运气", fontsize=11.3)
ax.legend(fontsize=8.4, frameon=False, loc="lower right")
ax.tick_params(labelsize=9)

# ---------------- (b) 残留的个体运气
ax2 = fig.add_axes([0.44, 0.17, 0.30, 0.66])
ax2.bar(x - 0.19, A1, width=0.36, color=BLUE, label="第 1 次采样(运气 $\\eta_1$)")
ax2.bar(x + 0.19, A2, width=0.36, color=ORANGE, label="第 2 次采样(同样的 $\\Delta$,不同运气)")
ax2.axhline(0, color=GREY, lw=0.9)
for i in x:
    ax2.text(i - 0.19, A1[i] + (0.07 if A1[i] > 0 else -0.19), f"{A1[i]:+.2f}",
             ha="center", fontsize=8.6, color=BLUE)
    ax2.text(i + 0.19, A2[i] + (0.07 if A2[i] > 0 else -0.19), f"{A2[i]:+.2f}",
             ha="center", fontsize=8.6, color=ORANGE)
ax2.set_xticks(x); ax2.set_xticklabels([f"样本{i+1}" for i in x], fontsize=9.4)
ax2.set_ylim(-2.1, 2.4)
ax2.set_ylabel("优势 $A_i$", fontsize=10.5)
ax2.set_title("(b) 但【个体运气】扣不掉:\n同样的真本事,两次采样优势不同", fontsize=11.3)
ax2.legend(fontsize=8.4, frameon=False, loc="upper left")
ax2.tick_params(labelsize=9)
ax2.text(1.5, -1.95, "残留的 $\\eta_i-\\overline{\\eta}$ 就是估计的【方差】,不是 bug", fontsize=9.0,
         color=RED, ha="center", bbox=dict(fc="#fdecea", ec="#e6a9a0"))

# ---------------- (c) 数学
ax3 = fig.add_axes([0.775, 0.17, 0.22, 0.66]); ax3.axis("off")
ax3.text(0.5, 0.99, "(c) 两层数学", ha="center", va="top", fontsize=11.6, color=DARK)
ax3.text(0.0, 0.88,
         "① 为什么减掉它【不引入偏差】\n"
         "策略梯度里减去任意与\n\"选了哪个动作\"无关的 $b$:\n"
         "$\\mathbb{E}[\\,b\\,\\nabla_\\theta\\log\\pi\\,]=0$\n"
         "⇒ 只降方差,不改期望\n"
         "(严格无偏版是留一均值\n"
         "$\\bar R_{-i}$;代码用含自身的\n"
         "组均值,偏差 $O(1/G)$)",
         fontsize=8.8, color=DARK, va="top")
ax3.text(0.0, 0.42,
         "② 两者的精度完全不同\n"
         "共同运气(可扣干净):\n"
         "$\\mathrm{Var}(\\bar\\eta)=\\sigma^2/G$:4→$0.25\\sigma^2$,24→$0.042\\sigma^2$\n"
         "个体运气(扣不掉):\n"
         "$\\mathrm{Var}(\\eta_i-\\bar\\eta)=\\sigma^2(1-\\frac{1}{G})$:几乎就是 $\\sigma^2$\n"
         "⇒ 只靠加大 $G$ 消不掉它",
         fontsize=8.8, color=DARK, va="top")
ax3.text(0.0, 0.16,
         "代码里的降方差措施:\n"
         "· per-prompt 统计【跨 update 累积】\n"
         "· 记 zero_std_ratio 监控退化组\n"
         "· adv_clip ±5、clip 1e-3、KL",
         fontsize=8.6, color=GREEN, va="top", bbox=dict(fc="#eef7f1", ec="#a9d5bb"))

fig.suptitle("组内基线扣掉了什么:难度 + 整批共享的偏移;扣不掉的是【每个样本自己的运气】(那是估计方差)", fontsize=12.2, y=0.99)
fig.savefig(RF + "fg_baseline_removes.png", bbox_inches="tight")
print(f"难度={DIFF}, Δ={D.tolist()}")
print("第1次 R =", np.round(R1, 3).tolist(), f"(均值 {R1.mean():.4f}) → A =", np.round(A1, 3).tolist())
print("第2次 R =", np.round(R2, 3).tolist(), f"(均值 {R2.mean():.4f}) → A =", np.round(A2, 3).tolist())
print("两次优势差异(残留运气的影响):", np.round(A1 - A2, 3).tolist())
print(f"残留方差系数 1-1/G: G=4 → {1-1/4:.3f}; G=24 → {1-1/24:.3f}")
print("wrote", RF + "fg_baseline_removes.png")
