"""Two intuitive figures for the closing section.

Fig 1  the two gravity fields side by side, and the ONE arrow that connects them
Fig 2  why the action field's lens is a random variable (one sample from the video field)
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
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse, Circle

RED, BLUE, GREEN, DARK, GREY, ORANGE = "#c0392b", "#2471a3", "#1e8449", "#2c3e50", "#95a5a6", "#b9770e"
RF = "/Users/kevin/code/lingbot-va/.claude/scratch/rf/"


def field(ax, sources, weights, color, title, lens_label, lens_color, out_label, rng):
    """schematic: point sources + level sets of their weighted sum + one reading + the pull"""
    xs = np.linspace(-2.6, 2.6, 200)
    ys = np.linspace(-2.6, 2.6, 200)
    GX, GY = np.meshgrid(xs, ys)
    rho = np.zeros_like(GX)
    for (sx, sy), w in zip(sources, weights):
        rho += w * np.exp(-((GX - sx) ** 2 + (GY - sy) ** 2) / (2 * 0.55 ** 2))
    ax.contourf(GX, GY, rho, levels=10 ** np.linspace(-2.6, 0, 11), cmap="YlOrBr", alpha=0.55)
    ax.contour(GX, GY, rho, levels=10 ** np.linspace(-2.6, 0, 11), colors="#8a6d3b",
               linewidths=0.4, alpha=0.5)
    for (sx, sy), w in zip(sources, weights):
        ax.plot(sx, sy, "o", ms=4 + 5 * w / max(weights), color=color, mec="w", mew=0.6,
                alpha=0.95, zorder=4)
    ax.add_patch(Ellipse((0, 0), 4.6, 4.2, fc=lens_color, ec=lens_color, lw=2.0,
                         ls="--", alpha=0.17, zorder=1))
    ax.text(-2.45, 2.3, lens_label, fontsize=8.6, color=lens_color, va="top",
            bbox=dict(fc="w", ec=lens_color, lw=0.9, alpha=0.92))
    # the reading and the pull
    rd = np.array([0.15, -0.35])
    w = np.array([wi * np.exp(-((rd[0] - sx) ** 2 + (rd[1] - sy) ** 2) / (2 * 0.55 ** 2))
                  for (sx, sy), wi in zip(sources, weights)])
    w = w / w.sum()
    mu = np.array([np.sum(w * np.array([s[0] for s in sources])),
                   np.sum(w * np.array([s[1] for s in sources]))])
    ax.plot(*rd, marker="*", ms=16, color="#111", mec="w", mew=0.9, zorder=6)
    ax.annotate("", xy=mu, xytext=rd, arrowprops=dict(arrowstyle="-|>", lw=3.0, color=GREEN,
                                                      mutation_scale=18), zorder=6)
    ax.text(rd[0] - 2.4, rd[1] - 0.55, "$x_\\sigma$ (读数)\n绿箭头 = 输出 $v$", fontsize=8.4, color=GREEN)
    ax.set_xlim(-2.6, 2.6); ax.set_ylim(-2.6, 2.6); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=11.5)
    ax.text(0, -2.95, out_label, ha="center", fontsize=9.4, color=color)


rng = np.random.default_rng(3)
V_SRC = [(-1.4, 0.9), (0.9, 1.2), (1.5, -0.9), (-1.0, -1.3), (0.2, 0.1)]
A_SRC = [(-1.5, -0.2), (-0.6, 1.4), (0.8, 0.2), (1.6, 1.3), (0.1, -1.5)]

fig = plt.figure(figsize=(14.6, 7.4), dpi=150)
axL = fig.add_axes([0.035, 0.155, 0.37, 0.635])
axR = fig.add_axes([0.615, 0.155, 0.37, 0.635])

field(axL, V_SRC, [1.0] * 5, RED,
      "视频场:$p(v\\,|\\,h)$  ——  世界模型",
      "透镜 = 历史 $h$\n(真实观测视频 + 过去已执行动作)", ORANGE,
      "去噪 → 未来一个 chunk 的视频 $\\hat v$", rng)
field(axR, A_SRC, [0.55, 0.5, 1.0, 0.7, 0.6], BLUE,
      "动作场:$p(a\\,|\\,h,\\hat v)$  ——  策略 / 逆动力学",
      "透镜 = 历史 $h$  +  本轮预测的视频 $\\hat v$\n(后者是一次采样 → 透镜是随机的)",
      RED, "去噪 → 动作 chunk $\\hat a$", rng)

# the one arrow that connects them
fig.patches.append(FancyArrowPatch((0.415, 0.52), (0.605, 0.52), transform=fig.transFigure,
                                   arrowstyle="-|>", mutation_scale=26, lw=4.0, color=GREEN))
fig.text(0.510, 0.555, "唯一一条\n同 chunk 通路", ha="center", fontsize=9.6, color=GREEN,
         fontweight="bold")
fig.patches.append(FancyArrowPatch((0.605, 0.30), (0.415, 0.30), transform=fig.transFigure,
                                   arrowstyle="-|>", mutation_scale=22, lw=2.6, color=RED,
                                   linestyle="--"))
fig.text(0.510, 0.255, "被掩码禁止\n$2c+1 < 2c$ 不成立", ha="center", fontsize=9.4, color=RED)

fig.text(0.5, 0.985, "两条链 = 同一个联合分布的两个条件切片(各自一个引力场)",
         ha="center", fontsize=14, color=DARK, fontweight="bold")
fig.text(0.5, 0.945, "$p(v_t,\\,a_t\\,|\\,h) \\;=\\; p(v_t\\,|\\,h) \\;\\cdot\\; p(a_t\\,|\\,h,\\,v_t)$",
         ha="center", fontsize=14.5, color=DARK)
fig.text(0.392, 0.902, "↑ 视频链(世界模型)", ha="center", fontsize=9.6, color=RED)
fig.text(0.612, 0.902, "↑ 动作链(策略 / 逆动力学)", ha="center", fontsize=9.6, color=BLUE)
fig.text(0.5, 0.866, "—— 这是一个「视频先行」的因果分解(谁先谁后就写在 frame_id 的 ±1 错位里)",
         ha="center", fontsize=9.8, color="#5d6d7e")
fig.text(0.5, 0.072,
         "① 每条链的引力场由它自己的条件分布决定:源 = 数据集里的视频 chunk / 动作 chunk;"
         "② 历史(以及 $\\hat v$)只是【换先验】= 给这个场加一块透镜(乘性、不造源、可把源压到 0);"
         "③ 每一步去噪 = 在那个 $\\sigma$ 特有的场里,取当前读数处的加权合成方向",
         ha="center", fontsize=9.8, color="#34495e")
fig.text(0.5, 0.022,
         "④ 代价:反向那条边不存在 → 模型答不了「如果我做这个动作,画面会变成什么」(what-if 反事实推演)",
         ha="center", fontsize=9.8, color=RED)
fig.savefig(RF + "story_two_fields.png", bbox_inches="tight")

# ================================================================ Fig 2
fig2, axes = plt.subplots(1, 2, figsize=(13.4, 4.9), dpi=150)
xs = np.linspace(-3, 3, 800)
PRIOR = 0.5 * np.exp(-((xs - 1.0) ** 2) / (2 * 0.45 ** 2)) + \
        0.5 * np.exp(-((xs + 1.0) ** 2) / (2 * 0.45 ** 2))


def posterior(center, width):
    p = PRIOR * np.exp(-((xs - center) ** 2) / (2 * width ** 2))
    return p / p.max()


for ax, mode in zip(axes, ["train", "infer"]):
    ax.fill_between(xs, PRIOR * 0.42, color="#bdc3c7", alpha=0.35)
    ax.plot(xs, PRIOR * 0.42, color="#7f8c8d", lw=1.5, ls="--")
    if mode == "train":
        post = posterior(0.95, 0.30)
        ax.fill_between(xs, post, color=BLUE, alpha=0.22)
        ax.plot(xs, post, color=BLUE, lw=2.4)
        pm = xs[np.argmax(post)]
        ax.plot(pm, 1.0, "o", ms=10, color=BLUE, mec="w", mew=1.3)
        ax.annotate("输出 $\\hat a$", xy=(pm, 1.0), xytext=(pm + 0.15, 1.18),
                    fontsize=9.2, color=BLUE, arrowprops=dict(arrowstyle="->", lw=1.1, color=BLUE))
        ax.text(-2.9, 1.30, "透镜 = 数据集里【那一个】真值 $v$\n(它和动作来自同一段真实轨迹 → 天然对齐)",
                fontsize=9.2, color=BLUE, va="top")
        ax.set_title("训练:透镜是确定的", fontsize=11.5)
    else:
        for center, width, col, ls, lab in [(0.55, 0.30, BLUE, "-", "第 1 次采样"),
                                            (1.35, 0.30, GREEN, "--", "第 2 次采样")]:
            post = posterior(center, width)
            ax.plot(xs, post, color=col, lw=2.2, ls=ls)
            ax.fill_between(xs, post, color=col, alpha=0.13)
            pm = xs[np.argmax(post)]
            ax.plot(pm, 1.0, "o", ms=9, color=col, mec="w", mew=1.2)
            ax.text(pm, 1.04, "①" if center < 1 else "②", fontsize=11, color=col, ha="center")
        p1, p2 = xs[np.argmax(posterior(0.55, 0.30))], xs[np.argmax(posterior(1.35, 0.30))]
        ax.annotate("", xy=(p2, 0.86), xytext=(p1, 0.86),
                    arrowprops=dict(arrowstyle="<|-|>", lw=1.8, color="#c0392b"))
        ax.text((p1 + p2) / 2, 0.78, "同一个历史,视频采样不同 → 动作输出不同",
                fontsize=9.0, color="#c0392b", ha="center")
        ax.text(-2.9, 1.30, "透镜 = 视频链【采样出来】的 $\\hat v$\n(它是随机变量 → 动作后验也是随机的)",
                fontsize=9.2, color=GREEN, va="top")
        ax.text(2.9, 0.62, "①② = 视频链两次独立采样", fontsize=8.8, color="#5d6d7e", ha="right")
        ax.set_title("推理:透镜是随机的", fontsize=11.5)
    ax.set_xlim(-3, 3); ax.set_ylim(0, 1.42)
    ax.set_xlabel("动作空间", fontsize=9.8); ax.set_yticks([])
    ax.tick_params(labelsize=8)
axes[0].set_ylabel("密度", fontsize=9.8)
fig2.suptitle("唯一的信息差,用引力场的语言说:动作场的透镜是一次采样\n"
              "灰色虚线 = 数据集里动作的先验 $p(a|h)$;彩色 = 乘上透镜(当前视频)之后的后验 $p(a|h,v)$",
              fontsize=11)
fig2.tight_layout(rect=[0, 0, 1, 0.845])
fig2.savefig(RF + "story_lens_random.png", bbox_inches="tight")
print("wrote", RF + "story_lens_random.png")
