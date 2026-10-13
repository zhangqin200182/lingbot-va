r"""三栏对照:"改哪里"。

  预训练      :前向加噪 → FM 回归 λ‖v_θ − v‖²
  Flow-GRPO   :改【采样器】(ODE→SDE)⇒ 拿到每步似然 ⇒ 优势 × 噪声方向(策略梯度)
  DiffusionNFT:改【训练目标】(前向 FM 回归 + 奖励加权的正/负分支)⇒ 不需要似然

共同点:三者都要"生成候选 + 打分";区别在【奖励从哪个口子进去】。
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
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

RED, BLUE, GREEN, DARK, GREY, ORANGE, PURPLE = ("#c0392b", "#2471a3", "#1e8449",
                                                "#2c3e50", "#95a5a6", "#b9770e", "#7d3c98")
RF = "/Users/kevin/code/lingbot-va/.claude/scratch/rf/"


def box(ax, x, y, w, h, text, fc="w", ec="#c8d4e0", fs=8.6, tc=DARK, lw=1.4, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc,
            fontweight=("bold" if bold else "normal"))


fig = plt.figure(figsize=(15.2, 6.2), dpi=150)
ax = fig.add_axes([0.02, 0.03, 0.96, 0.90]); ax.axis("off")
ax.set_xlim(0, 3); ax.set_ylim(0, 1)

cols = [
    (0.0, "① 预训练(基线)", GREY, "#f7f9fb",
     [("前向加噪\n$x_t=\\alpha_t x_0+\\sigma_t\\epsilon$", "w", "#c8d4e0"),
      ("FM 回归\n$\\|v_\\theta-v\\|^2$", "w", "#c8d4e0"),
      ("更新 $\\theta$", "w", "#c8d4e0")],
     "没有奖励;只学\"数据长什么样\""),
    (1.0, "② Flow-GRPO:改【采样器】", RED, "#fdecea",
     [("采样:ODE → 【SDE】\n(每步注入 $g\\,dW$)", "#fdecea", RED),
      ("记下【每步 $\\log\\pi$】\n+ 组内优势 $A$", "#fdecea", RED),
      ("策略梯度\n$-A\\cdot\\log\\pi$(ratio/clip/KL)", "#fdecea", RED)],
     "随机性进【采样】⇒ 才有似然 ⇒ 梯度 = 优势 × 噪声方向"),
    (2.0, "③ DiffusionNFT:改【训练目标】", GREEN, "#eef7f1",
     [("采样:照旧\n(任意 solver,可用确定性 ODE)", "#eef7f1", GREEN),
      ("只留【干净样本】$x_0$\n前向加噪 $x_t$(不记似然)", "#eef7f1", GREEN),
      ("奖励加权的 FM 回归\n$r\\|v^+_\\theta-v\\|^2+(1-r)\\|v^-_\\theta-v\\|^2$", "#eef7f1", GREEN)],
     "奖励进【监督目标】⇒ 不需要似然、不挑 solver"),
]
for x0, title, col, bg, items, note in cols:
    ax.add_patch(FancyBboxPatch((x0 + 0.02, 0.10), 0.92, 0.84, boxstyle="round,pad=0.02",
                                fc=bg, ec=col, lw=2.0, alpha=0.55))
    ax.text(x0 + 0.48, 0.925, title, ha="center", va="center", fontsize=11.6, color=col, fontweight="bold")
    yy = 0.76
    for k, (txt, fc, ec) in enumerate(items):
        box(ax, x0 + 0.09, yy - 0.135, 0.78, 0.135, txt, fc, ec, fs=8.6,
            lw=(2.2 if ec in (RED, GREEN) else 1.3))
        if k < 2:
            ax.add_patch(FancyArrowPatch((x0 + 0.48, yy - 0.14), (x0 + 0.48, yy - 0.175),
                                         arrowstyle="-|>", mutation_scale=12, lw=1.8, color=DARK))
        yy -= 0.215
    ax.text(x0 + 0.48, 0.145, note, ha="center", va="center", fontsize=8.6, color=col)

ax.annotate("", xy=(1.98, 0.02), xytext=(1.02, 0.02),
            arrowprops=dict(arrowstyle="-|>", lw=2.2, color=DARK))
ax.text(1.5, -0.02, "共同点:都要\"生成候选 + 打分\";区别只在【奖励从哪个口子进去】",
        ha="center", va="top", fontsize=9.4, color=DARK)
fig.savefig(RF + "fg_nft_where_to_change.png", bbox_inches="tight")
print("wrote", RF + "fg_nft_where_to_change.png")
