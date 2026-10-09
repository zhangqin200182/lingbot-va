r"""RL 会不会"破坏 flow matching",让采样路径变弯?

纠正前提:flow matching 的 ODE 轨迹【本来就不直】
  - 直线是【插值】x_σ=(1−σ)x₀+σε(必须知道真实 (x₀,ε) 才能画)
  - ODE 轨迹是【条件平均场】的积分,多点数据下必然弯曲
  - 只有数据是单点时,ODE 轨迹才恰好是直线
(a) 单点数据 → 直;两点数据 → 弯(同样都是标准 flow matching)
(b) RL 前后:场变了 → 轨迹形状变了;但"从噪声 ODE 积到数据"的能力没变
"""
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
for _f in ["/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
           "/System/Library/Fonts/STHeiti Medium.ttc"]:
    try:
        font_manager.fontManager.addfont(_f)
        matplotlib.rcParams["font.family"] = matplotlib.rcParams["font.family"] = \
            font_manager.FontProperties(fname=_f).get_name()
        break
    except Exception:
        continue
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np

RED, BLUE, GREEN, DARK, GREY, ORANGE, PURPLE = ("#c0392b", "#2471a3", "#1e8449",
                                                "#2c3e50", "#95a5a6", "#b9770e", "#7d3c98")
RF = "/Users/kevin/code/lingbot-va/.claude/scratch/rf/"
SIGG = np.linspace(1.0, 0.0, 300)


def make_field(mu, w):
    """两/一原子数据的 v*(x,σ) = (x − E[x0|x])/σ"""
    mu = np.asarray(mu, float); w = np.asarray(w, float)

    def post(x, s):
        m = (1 - s) * mu
        lw = (np.log(w)[None, :] - 0.5 * np.log(2 * np.pi * s ** 2)
              - (x[:, None] - m[None, :]) ** 2 / (2 * s ** 2))
        lw -= lw.max(axis=1, keepdims=True)
        pw = np.exp(lw)
        return pw / pw.sum(axis=1, keepdims=True)

    def vf(x, s):
        return (x - post(x, s) @ mu) / max(s, 1e-6)

    return vf


def odo(vf, x0):
    x = np.array([x0], float); tr = [x[0]]
    for i in range(len(SIGG) - 1):
        s, sn = SIGG[i], SIGG[i + 1]
        x = x + vf(x, s) * (sn - s)
        tr.append(x[0])
    return np.array(tr)


fig = plt.figure(figsize=(15.2, 5.5), dpi=150)

# ---------------- (a) 单点 vs 两点
ax = fig.add_axes([0.05, 0.17, 0.29, 0.66])
vf1 = make_field([0.55], [1.0])
vf2 = make_field([1.0, -1.0], [0.5, 0.5])
rng = np.random.default_rng(3)
for k in range(3):
    e = rng.normal(scale=0.75)
    t = odo(vf1, e)
    ax.plot(SIGG, t, color=BLUE, lw=2.4, alpha=0.95,
            )
    ax.plot([1], [t[0]], "o", ms=7, color=BLUE); ax.plot([0], [t[-1]], "s", ms=7, color=BLUE)
for k in range(3):
    e = rng.normal()
    t = odo(vf2, e)
    ax.plot(SIGG, t, color=RED, lw=1.9, alpha=0.95,
            )
    ax.plot([1], [t[0]], "o", ms=7, color=RED); ax.plot([0], [t[-1]], "s", ms=7, color=RED)
ax.annotate("", xy=(0.06, -2.62), xytext=(0.94, -2.62),
            arrowprops=dict(arrowstyle="-|>", lw=2.2, color=DARK))
ax.text(0.5, -2.85, "生成方向:σ 从 1 递减到 0", fontsize=9.4, color=DARK, ha="center")
ax.text(0.99, 1.80, "● 起点(σ=1,纯噪声)", fontsize=9.2, color=DARK, ha="left")
ax.text(0.01, 1.80, "■ 终点(σ=0,数据)", fontsize=9.2, color=DARK, ha="right")
ax.text(0.55, 1.30, "单点数据 → 轨迹是【直线】", fontsize=9.6, color=BLUE, ha="center")
ax.text(0.42, -1.55, "两点数据 → 轨迹是【弯的】", fontsize=9.6, color=RED, ha="center")
ax.set_xlim(1.02, -0.02); ax.set_ylim(-2.95, 2.0)
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlabel("$\\sigma$", fontsize=10.5)
ax.set_title("(a) 纠正前提:标准 flow matching 的 ODE 轨迹\n【本来就不直】,只有单点数据才直", fontsize=11.3)

# ---------------- (b) RL 前后
ax2 = fig.add_axes([0.40, 0.17, 0.32, 0.66])
vf_before = make_field([1.0, -1.0], [0.5, 0.5])
vf_after = make_field([1.0, -1.0], [0.8, 0.2])
rng = np.random.default_rng(11)
for k in range(4):
    e = rng.normal()
    t = odo(vf_before, e)
    ax2.plot(SIGG, t, color=GREY, lw=1.8, alpha=0.95,
             )
    ax2.plot([1], [t[0]], "o", ms=6, color=GREY)
for k in range(4):
    e = rng.normal()
    t = odo(vf_after, e)
    ax2.plot(SIGG, t, color=GREEN, lw=2.1, alpha=0.95,
             )
    ax2.plot([1], [t[0]], "o", ms=6, color=GREEN)
ax2.axhline(1.0, color=GREEN, ls=":", lw=1.1); ax2.axhline(-1.0, color=GREY, ls=":", lw=1.1)
ax2.annotate("", xy=(0.06, -2.62), xytext=(0.94, -2.62),
             arrowprops=dict(arrowstyle="-|>", lw=2.2, color=DARK))
ax2.text(0.5, -2.85, "生成方向:σ 从 1 递减到 0", fontsize=9.4, color=DARK, ha="center")
ax2.set_xlim(1.02, -0.02); ax2.set_ylim(-2.95, 2.5)
ax2.set_xticks([]); ax2.set_yticks([])
ax2.set_xlabel("$\\sigma$", fontsize=10.5)
ax2.set_title("(b) RL 改的是【场】,所以轨迹形状变了\n但\"从噪声 ODE 积到数据\"这件事没变", fontsize=11.3)
ax2.text(0.99, 2.15, "灰 = 训练前($0.5/0.5$)", fontsize=9.4, color=GREY, ha="left")
ax2.text(0.5, -2.35, "绿 = RL 之后($0.8/0.2$)", fontsize=9.4, color=GREEN, ha="center")
ax2.text(0.5, 2.42, "轨迹依然弯(仍然是多点数据)", fontsize=9.2, color=RED, ha="center")

# ---------------- (c) 什么变、什么没变
ax3 = fig.add_axes([0.755, 0.17, 0.235, 0.66]); ax3.axis("off")
ax3.text(0.5, 0.99, "(c) 什么变了 / 什么没变", ha="center", va="top", fontsize=11.6, color=DARK)
rows = [["", "变?", "说明"],
        ["路径定义\n$x_\\sigma=(1-\\sigma)x_0+\\sigma\\epsilon$", "否", "在训练目标里\nRL 不碰"],
        ["网络输出\n速度 $v$", "否", "接口不变"],
        ["能否用 ODE\n从噪声积到数据", "否", "能力不变"],
        ["场 $v_\\theta$ 的形状", "是", "RL 就是干这个"],
        ["轨迹弯度 /\n少步采样质量", "可能", "推断\n论文未测"]]
tb = ax3.table(cellText=rows, cellLoc="center", bbox=[0.0, 0.30, 1.0, 0.62])
tb.auto_set_font_size(False); tb.set_fontsize(8.0)
for (r, c), cell in tb.get_celld().items():
    if r == 0:
        cell.set_facecolor("#eef2f7")
    if c == 1:
        t = cell.get_text().get_text()
        cell.set_facecolor({"否": "#eef7f1", "是": "#fdecea", "可能": "#fff4e5"}.get(t, "w"))
ax3.text(0.0, 0.24,
         "关键:轨迹从\"弯\"变\"更弯/不一样\"\n不是【破坏范式】,而是【目标变了】——\n"
         "RL 的极限效果 ≈ 把目标分布\n从 $p_{data}$ 倾斜成 reward-tilted 分布。",
         fontsize=8.8, color=DARK, va="top",
         bbox=dict(fc="#fffbe6", ec="#e0c060"))
ax3.text(0.0, 0.03,
         "KL 约束($\\beta$)把 $\\theta$ 拉在原模型附近,\n所以变形通常不会失控;\n"
         "真想要少步可采样,得额外约束或蒸馏。",
         fontsize=8.8, color=GREEN, va="bottom", bbox=dict(fc="#eef7f1", ec="#a9d5bb"))

fig.suptitle("RL 会不会破坏 flow matching?—— 不会破坏\"怎么用\",但会改变\"走出来的形状\"", fontsize=12.4, y=0.99)
fig.savefig(RF + "fg_rl_vs_flow.png", bbox_inches="tight")

# 数值核对:RL 前后的终点分布
rng = np.random.default_rng(7)
for lab, vf in [("训练前 0.5/0.5", vf_before), ("RL 后 0.8/0.2", vf_after)]:
    ends = np.array([odo(vf, e)[-1] for e in rng.normal(size=3000)])
    print(f"{lab}: P(+1)={float((ends>0).mean()):.3f}  (弧长/直线距离比 = 弯度指标,见下)")
    # 弯度:单条轨迹的弧长 / 起终点直线距离
    tr = odo(vf, rng.normal())
    arc = np.sum(np.hypot(np.diff(SIGG), np.diff(tr)))
    chord = abs(tr[-1] - tr[0])
    print(f"    一条轨迹:弧长 {arc:.3f},起终点直线距离 {chord:.3f},弯曲度 {arc/max(chord,1e-9):.2f}")
print("wrote", RF + "fg_rl_vs_flow.png")
