r"""拿到 x₀ 之后为什么又加噪?—— 因为"采样"和"训练"是两件相反的事。

  采样(部署时干的):  ε ──去噪──▶ x₀        目的:造出一张图
  训练(学的时候干的): x₀ ──加噪──▶ x_t      目的:出题,教网络"速度场长什么样"

网络的【输入永远是一个带噪点】(x_t, t):
  · 它在部署时从 t=1 走到 t=0,从来看不到"干净图作输入"
  · "速度"也需要同一路径上的两个点才能定义 ⇒ 干净图本身没有速度

而"加噪"是【自己出题】:
  拿一张 x₀,【重新抽】一个 ε_new ⇒ 立刻知道正确答案 v = α̇x₀ + σ̇ε_new
  一张图 → 无穷多组 (x_t, v)  ⇒ 不需要保存轨迹,还能当数据增广
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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

RED, BLUE, GREEN, DARK, GREY, ORANGE, PURPLE = ("#c0392b", "#2471a3", "#1e8449",
                                                "#2c3e50", "#95a5a6", "#b9770e", "#7d3c98")
RF = "/Users/kevin/code/lingbot-va/.claude/scratch/rf/"

fig = plt.figure(figsize=(15.2, 5.8), dpi=150)

# ---------------- (a) 两件相反的事
ax = fig.add_axes([0.03, 0.55, 0.94, 0.40]); ax.axis("off")
ax.text(0.5, 1.02, "(a) 采样 与 训练:方向相反,目的不同", ha="center", va="top",
        fontsize=12, color=DARK)
ax.add_patch(FancyArrowPatch((0.08, 0.70), (0.92, 0.70), arrowstyle="-|>", mutation_scale=16,
                             lw=3.0, color=RED))
ax.text(0.06, 0.79, "采样(部署:造图)", fontsize=10.4, color=RED, ha="left")
for x, lab in [(0.10, "$\\epsilon$"), (0.30, ""), (0.50, ""), (0.70, ""), (0.90, "$x_0$")]:
    ax.plot([x], [0.70], "o", ms=8, color=RED)
ax.text(0.50, 0.61, "去噪:从纯噪声一步步走到干净图", fontsize=9.4, color=RED, ha="center")
ax.add_patch(FancyArrowPatch((0.92, 0.26), (0.08, 0.26), arrowstyle="-|>", mutation_scale=16,
                             lw=3.0, color=GREEN))
ax.text(0.94, 0.35, "训练(学习:出题)", fontsize=10.4, color=GREEN, ha="right")
for x in [0.90, 0.70, 0.50]:
    ax.plot([x], [0.26], "o", ms=8, color=GREEN)
ax.plot([0.10], [0.26], "o", ms=8, color=ORANGE)
ax.text(0.90, 0.175, "$x_0$", fontsize=9.4, color=GREEN, ha="center")
ax.text(0.44, 0.175, "加噪:造出各种噪声水平的【考题】", fontsize=9.4, color=GREEN, ha="center")
ax.text(0.10, 0.175, "$\\epsilon_{\\rm new}$", fontsize=9.4, color=ORANGE, ha="center")
ax.text(0.5, 0.045, "一个在\"用\"网络,一个在\"教\"网络 —— 所以都要动噪声,但方向相反",
        fontsize=9.6, color=DARK, ha="center", bbox=dict(fc="#fffbe6", ec="#e0c060"))
ax.set_xlim(0, 1); ax.set_ylim(0.0, 1.05)

# ---------------- (b) 一张图 → 无穷考题
ax2 = fig.add_axes([0.04, 0.08, 0.28, 0.42])
x0 = 1.0
rng = np.random.default_rng(4)
for k in range(3):
    e = rng.normal() * 1.6
    ts = np.linspace(0, 1, 200)
    ax2.plot(ts, (1 - ts) * x0 + ts * e, lw=2.0, color=[BLUE, GREEN, PURPLE][k], alpha=0.9)
    ax2.plot([1], [e], "o", ms=9, color=[BLUE, GREEN, PURPLE][k])
    ax2.text(1.02, e, f"$\\epsilon_{{{k+1}}}$", fontsize=9.0, color=[BLUE, GREEN, PURPLE][k], va="center")
    for t in np.linspace(0.2, 0.9, 5):
        ax2.plot([t], [(1 - t) * x0 + t * e], "o", ms=4.5, color=[BLUE, GREEN, PURPLE][k], alpha=0.7)
ax2.plot([0], [x0], "o", ms=12, color=ORANGE, zorder=8)
ax2.text(0.02, x0 - 0.55, "同一张 $x_0$", fontsize=9.4, color=ORANGE)
ax2.set_xlim(-0.05, 1.35); ax2.set_ylim(-3.2, 3.2)
ax2.set_xlabel("噪声水平 $t$", fontsize=9.8); ax2.set_ylabel("$x$", fontsize=9.8)
ax2.set_title("(b) 重新抽 $\\epsilon$ ⇒ 一张图出无数道题\n(每个点都自带正确答案 $v$)", fontsize=10.8)
ax2.tick_params(labelsize=8.6)

# ---------------- (c) 三个要点
ax3 = fig.add_axes([0.37, 0.08, 0.61, 0.42]); ax3.axis("off")
ax3.text(0.0, 0.99,
         "① 为什么必须加噪?\n"
         "　网络的输入【永远是带噪点】$(x_t,t)$:部署时它从 $t=1$ 走到 $t=0$,从来看不到\"干净图当输入\";\n"
         "　而且\"速度\"要同一路径上的两个点才能定义 ⇒ 干净图本身没有速度。\n\n"
         "② 为什么用【新】的 $\\epsilon$,不用采样时那个?\n"
         "　两者都合法(都来自 $x_t=\\alpha_t x_0+\\sigma_t\\epsilon$ 这一族)。重新抽的好处:一张 $x_0$ 出【无数道题】\n"
         "　(不同噪声方向 × 不同水平)⇒ 数据增广;且不必保存轨迹 ⇒ 省显存。\n\n"
         "③ 奖励在这一步参与吗?\n"
         "　完全不参与。④ 只负责\"把图变成考题\";奖励在 ⑥ 里当权重:$r$ 大就多学,小就反着学。",
         fontsize=9.2, color=DARK, va="top")
fig.suptitle("拿到 $x_0$ 后为什么又加噪:采样是\"把噪声变成图\",训练是\"把图变成考题\" —— 同一个插值族,两个方向",
             fontsize=12.0, y=0.985)
fig.savefig(RF + "fg_why_renoise.png", bbox_inches="tight")
print("wrote", RF + "fg_why_renoise.png")
