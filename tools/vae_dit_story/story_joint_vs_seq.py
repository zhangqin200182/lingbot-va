r"""Sequential (two-step) vs joint denoising, in the gravity-field language.

(A) two fields, one-way: the action field's LENS is one SAMPLE from the video field
(B) one field in the product space (video x action): the correlation is in its geometry
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
from matplotlib.patches import FancyArrowPatch, Ellipse

RED, BLUE, GREEN, DARK, GREY, ORANGE = "#c0392b", "#2471a3", "#1e8449", "#2c3e50", "#95a5a6", "#b9770e"
RF = "/Users/kevin/code/lingbot-va/.claude/scratch/rf/"

fig = plt.figure(figsize=(15.2, 6.2), dpi=150)
gs = fig.add_gridspec(1, 3, wspace=0.14, left=0.03, right=0.985, top=0.78, bottom=0.17)
axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

VPTS = [(-1.35, 0.75), (0.95, 1.05), (1.45, -0.85), (-0.95, -1.25), (0.15, 0.05)]
APTS = [(-1.45, -0.25), (-0.55, 1.35), (0.85, 0.15), (1.55, 1.25), (0.05, -1.45)]
START = np.array([0.15, 0.05])


def draw(ax, pts, rings_at, lens_color, title, sub):
    for (x, y) in pts:
        ax.plot(x, y, "o", ms=5.5, color=DARK, mec="w", mew=0.6, zorder=5)
    for m in rings_at:
        for r in [0.45, 0.78, 1.12]:
            ax.add_patch(Ellipse(m, r * 2, r * 2, fc="none", ec="#8a6d3b", lw=0.5, alpha=0.5))
    ax.add_patch(Ellipse((0, 0), 4.5, 3.7, fc=lens_color, ec=lens_color, lw=1.8, ls="--", alpha=0.13))
    ax.plot(*START, marker="*", ms=16, color="#111", mec="w", mew=0.9, zorder=7)
    ax.text(-2.25, 1.98, title, fontsize=12, color=DARK)
    ax.text(-2.25, 1.70, sub, fontsize=9.0, color=lens_color)
    ax.set_xlim(-2.35, 2.35); ax.set_ylim(-2.05, 2.25)
    ax.set_xticks([]); ax.set_yticks([])


# ---------------- (A) 两步走
draw(axes[0], VPTS, [(-1.35, 0.75), (0.15, 0.05)], ORANGE,
     "视频场  $p(v|h)$", "透镜 = 历史(确定)")
pick = np.array([-1.05, 0.62])
axes[0].add_patch(FancyArrowPatch(START, pick, arrowstyle="-|>", mutation_scale=17, lw=2.6, color=GREEN, zorder=6))
axes[0].plot(*pick, "o", ms=12, color=GREEN, mec="w", mew=1.4, zorder=8)
axes[0].text(pick[0], pick[1] - 0.42, "采样出的 $\\hat v$\n(只取一个)", fontsize=9.0, color=GREEN, ha="center")

draw(axes[1], APTS, [(-1.45, -0.25), (0.05, -1.45)], RED,
     "动作场  $p(a|h,\\hat v)$", "透镜 = 历史 + $\\hat v$  (随机)")
pick2 = np.array([-0.35, -1.15])
axes[1].add_patch(FancyArrowPatch(START, pick2, arrowstyle="-|>", mutation_scale=17, lw=2.6, color=GREEN, zorder=6))
axes[1].plot(*pick2, "o", ms=12, color=GREEN, mec="w", mew=1.4, zorder=8)
axes[1].text(pick2[0], pick2[1] - 0.45, "动作输出 $\\hat a$", fontsize=9.0, color=GREEN, ha="center")

# ---------------- (B) 联合
JS = [(-1.25, -1.25), (1.25, 1.25), (1.35, -1.15), (-1.15, 1.35)]
draw(axes[2], JS, JS, GREY, "乘积空间:一个联合场", "相关性写在几何里(v 与 a 对齐的模式)")
path = np.array([[0.0, 0.0], [0.28, 0.18], [0.8, 0.68], [1.1, 1.02], [1.25, 1.25]])
axes[2].plot(path[:, 0], path[:, 1], "-o", color=GREEN, lw=2.4, ms=4.5, zorder=7)
axes[2].plot(0, 0, marker="X", ms=10, color="#111", mec="w", mew=0.9, zorder=8)
axes[2].text(0.1, -0.35, "纯噪声起点", fontsize=8.6, color="#5d6d7e")
axes[2].set_xlabel("视频方向的 1 维投影  →", fontsize=9.4)
axes[2].set_ylabel("动作方向的 1 维投影  →", fontsize=9.4)
axes[2].annotate("两个模态每一步\n互相看,一起落到\n一致的一对 $(\\hat v,\\hat a)$",
                 xy=(1.25, 1.25), xytext=(-2.2, 0.35), fontsize=9.0, color=GREEN,
                 arrowprops=dict(arrowstyle="->", lw=1.3, color=GREEN))

fig.patches.append(FancyArrowPatch((0.3245, 0.47), (0.3595, 0.47), transform=fig.transFigure,
                                   arrowstyle="-|>", mutation_scale=20, lw=3.2, color=GREEN))
fig.text(0.20, 0.885, "(A) 两步走:两个场 + 一个随机透镜", ha="center", fontsize=12.5,
         color=RED, fontweight="bold")
fig.text(0.79, 0.885, "(B) 联合去噪:乘积空间里只有一个场", ha="center", fontsize=12.5,
         color=BLUE, fontweight="bold")
fig.text(0.012, 0.09,
         "目标分布完全相同:$p(v,a|h)=p(v|h)\\cdot p(a|h,v)$ 是恒等式。(A) 用祖先采样做这个分解 —— 先把 $\\hat v$ 采出来(一次\"承诺\"),"
         "它再当动作场的透镜;训练时动作看到的却是【真值 $v$】→ 存在 gap。\n"
         "(B) 直接采联合:两个模态在每一步互相看,训练和推理天然一致;代价是跨模态 $\\sigma$ 组合空间巨大、每步的序列长一倍,"
         "而且不能给两个模态配完全不同的噪声调度(现在动作 shift=0.05、视频 shift=5.0)。",
         ha="left", fontsize=9.6, color="#34495e")
fig.savefig(RF + "story_joint_vs_seq.png", bbox_inches="tight")
print("wrote", RF + "story_joint_vs_seq.png")
