r"""RL 改 v,怎么处理每步的漂移(补偿)与噪声?

关键事实(从代码符号化推导):
  mean = [1 + dσ·nl²/(2(1−σ))]·x  +  dσ·(1+nl²/2)·v_θ
                                        └ 与 σ 无关的常数! ┘
⇒ 补偿项没有变成"v 旁边外加的一项",而是被吸收进 v 的系数里
⇒ θ 只通过这个系数进入 mean,改动按固定倍数传递,不存在"被漂移稀释"

(a) 系数分解:θ 只出现在 v 的系数上,且该系数与 σ 无关
(b) 噪声带来的"整组运气"由组内基线自动扣掉(差一个常数 → 优势完全相同)
(c) 三层保护
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

fig = plt.figure(figsize=(15.2, 5.6), dpi=150)

# ---------------- (a) 系数分解
ax = fig.add_axes([0.055, 0.17, 0.315, 0.66])
nls = np.array([0.5, 0.7, 1.0])
amp = 1 + nls ** 2 / 2
b = ax.bar(range(3), amp, width=0.5, color=[BLUE, GREEN, RED], alpha=0.9)
for r, v in zip(b, amp):
    ax.text(r.get_x() + r.get_width() / 2, v + 0.02, f"{v:.3f}×", ha="center", fontsize=10.5, color=DARK)
ax.axhline(1.0, color=GREY, ls="--", lw=1.4)
ax.text(2.35, 1.02, "无补偿(纯 ODE)", fontsize=9.0, color=GREY, ha="right")
ax.set_xticks(range(3)); ax.set_xticklabels([f"noise_level\n{n}" for n in nls], fontsize=9.4)
ax.set_ylim(0.9, 1.65)
ax.set_ylabel("$v_\\theta$ 的系数 $/\\,d\\sigma$", fontsize=10.5)
ax.set_title("(a) 补偿被吸收进 $v$ 的系数:\n$B=d\\sigma\\,(1+\\mathrm{nl}^2/2)$,与 $\\sigma$ 无关", fontsize=11.3)
ax.tick_params(labelsize=9)
ax.text(-0.45, 1.55,
        "代码里的那一行:\n"
        "$\\mathrm{mean}=x\\,(1+\\frac{g^2d\\sigma}{2\\sigma})+v_\\theta\\,d\\sigma\\,(1+\\frac{g^2(1-\\sigma)}{2\\sigma})$\n"
        "代入 $g^2=\\dfrac{\\sigma}{1-\\sigma}\\mathrm{nl}^2$ ⇒ 第二项系数 = $d\\sigma(1+\\mathrm{nl}^2/2)$\n"
        "⇒ $\\theta$ 只出现在 $v_\\theta$ 的系数上,而且是常数",
        fontsize=8.8, color=DARK, va="top", bbox=dict(fc="#fffbe6", ec="#e0c060"))

# ---------------- (b) 组内基线扣掉运气
ax2 = fig.add_axes([0.425, 0.17, 0.30, 0.66])
G1 = np.array([0.80, 0.20, 0.90, 0.50])          # 这一组"运气好"
G2 = G1 - 0.30                                    # 同样 4 个样本,整体运气差 0.3
A1 = (G1 - G1.mean()) / (G1.std() + 1e-4)
A2 = (G2 - G2.mean()) / (G2.std() + 1e-4)
x = np.arange(4)
ax2.bar(x - 0.19, A1, width=0.36, color=BLUE, label=f"组 1(均值 {G1.mean():.2f})")
ax2.bar(x + 0.19, A2, width=0.36, color=ORANGE, label=f"组 2(均值 {G2.mean():.2f},整体差 0.3)")
ax2.axhline(0, color=GREY, lw=0.9)
for i in x:
    ax2.text(i, A1[i] + (0.06 if A1[i] > 0 else -0.16), f"{A1[i]:+.2f}", ha="center", fontsize=9.0, color=BLUE)
ax2.set_xticks(x); ax2.set_xticklabels([f"样本{i+1}" for i in x], fontsize=9.4)
ax2.set_ylim(-1.75, 1.75)
ax2.set_ylabel("优势 $A_i$", fontsize=10.5)
ax2.set_title("(b) 噪声带来的\"整组运气\"被自动扣掉:\n两组奖励只差一个常数 → 优势【完全相同】", fontsize=11.3)
ax2.legend(fontsize=8.8, frameon=False, loc="upper left")
ax2.tick_params(labelsize=9)
ax2.text(1.5, -1.62, "组内平均 = 这一组的共同运气(含噪声);减掉它,剩下的才是\"这个样本本身的好坏\"",
         fontsize=8.8, color=GREEN, ha="center", bbox=dict(fc="#eef7f1", ec="#a9d5bb"))

# ---------------- (c) 三层保护
ax3 = fig.add_axes([0.75, 0.17, 0.245, 0.66]); ax3.axis("off")
ax3.text(0.5, 0.99, "(c) 谁在处理什么", ha="center", va="top", fontsize=11.6, color=DARK)
ax3.text(0.0, 0.88,
         "漂移(补偿)项:\n"
         "· 不需要\"克服\"——它不进 $\\theta$ 的依赖关系,\n"
         "　只是把 $v$ 的系数乘一个常数\n"
         "· 它的作用是让\"加噪声不改分布\",\n"
         "　保证 RL 的起点是合法的原模型\n\n"
         "噪声:\n"
         "· 不需要\"克服\"——它是【无偏探索】\n"
         "　$\\nabla_\\theta\\log\\pi=(\\xi/s)\\partial m/\\partial\\theta$,$\\mathbb{E}[\\xi]=0$\n"
         "· 它造成的\"运气差异\"由【组内基线】扣掉\n"
         "　$A_i=(R_i-\\bar R)/\\mathrm{std}$",
         fontsize=9.0, color=DARK, va="top")
ax3.text(0.0, 0.24,
         "三层保护:\n"
         "① 组内基线  → 扣掉共同运气\n"
         "② clip 1e-3 → 单步只许动 0.1%\n"
         "③ KL(β)     → 整体别跑离原模型",
         fontsize=9.0, color=GREEN, va="top", bbox=dict(fc="#eef7f1", ec="#a9d5bb"))

fig.suptitle("改 $v$ 会不会被漂移和噪声淹没?—— 补偿变成 $v$ 的常数系数;噪声是无偏探索,运气由组内基线扣掉",
             fontsize=12.2, y=0.99)
fig.savefig(RF + "fg_v_vs_drift_noise.png", bbox_inches="tight")
print(f"放大因子 1+nl²/2: nl=0.5 → {1+0.25/2:.4f}; nl=0.7 → {1+0.49/2:.4f}; nl=1.0 → {1+1/2:.4f}")
print("组1 奖励", G1, "→ 优势", np.round(A1, 4))
print("组2 奖励", G2, "→ 优势", np.round(A2, 4))
print("两组优势是否完全相同:", np.allclose(A1, A2))
print("wrote", RF + "fg_v_vs_drift_noise.png")
