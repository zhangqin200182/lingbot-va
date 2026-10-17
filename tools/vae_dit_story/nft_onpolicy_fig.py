r"""NFT 就没有 on/off-policy 问题了吗?—— 不是"没有",而是"换了形态"。

  GRPO:off-policy 是【约束】
    数据必须基本来自当前策略,否则 ratio 失配 ⇒ 估计有偏/爆 ⇒ 必须 clip/KL 压回来
    ⇒ 它是"技术故障"层面的问题

  NFT:off-policy 是【旋钮】
    损失里没有 ρ,所以不会"技术上出错"
    但偏离仍让"优化的是旧策略给出的方向" ⇒ 【偏差】
    ⇒ 它是"统计偏差"层面的问题

  ★ 最硬的反证:η 的消融
      η=0(最 on-policy)  → 早期快,但【不稳】
      η=0.9(最 off-policy) → 【收敛慢】
      ⇒ 小→大调度最好
    如果 off-policy 无所谓,η 就不会有两端都变差的行为。
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

# ---------------- (a) 形态变化
ax = fig.add_axes([0.03, 0.10, 0.40, 0.78]); ax.axis("off")
ax.text(0.5, 1.00, "(a) 同一个问题,两种形态", ha="center", va="top", fontsize=12, color=DARK)
ax.add_patch(plt.Rectangle((0.02, 0.58), 0.96, 0.34, fc="#fdecea", ec=RED, lw=1.8))
ax.text(0.05, 0.875, "GRPO:off-policy 是【约束】", fontsize=10.4, color=RED, va="top")
ax.text(0.05, 0.795,
        "数据必须基本来自当前策略,\n否则 $\\rho=\\pi_{\\rm new}/\\pi_{\\rm old}$ 失配\n"
        "⇒ 估计有偏 / 数值爆炸\n⇒ 必须用 clip + KL 把它【压回来】\n\n"
        "性质:技术故障(会明确失效)",
        fontsize=8.8, color=DARK, va="top")
ax.add_patch(plt.Rectangle((0.02, 0.20), 0.96, 0.34, fc="#eef7f1", ec=GREEN, lw=1.8))
ax.text(0.05, 0.495, "NFT:off-policy 是【旋钮】", fontsize=10.4, color=GREEN, va="top")
ax.text(0.05, 0.415,
        "损失里没有 $\\rho$ ⇒ 不会\"技术上出错\"\n"
        "但偏离仍让\"优化的是旧策略给出的方向\"\n"
        "⇒ 【有偏】,而且没有 IS 修正\n\n"
        "性质:统计偏差(能用,但不等于真梯度)",
        fontsize=8.8, color=DARK, va="top")
ax.text(0.5, 0.09,
        "所以准确说法:\nNFT 把 off-policy 从【必须压制的约束】\n变成【可以主动调节的旋钮】",
        fontsize=9.4, color=DARK, ha="center", va="center",
        bbox=dict(fc="#fffbe6", ec="#e0c060"))
ax.set_xlim(0, 1); ax.set_ylim(0.03, 1.02)

# ---------------- (b) eta 消融:反证
ax2 = fig.add_axes([0.48, 0.17, 0.28, 0.66])
eta = np.linspace(0.001, 0.95, 400)
# 示意:两端都差(左:不稳;右:慢),中间最好 —— 据论文文字描述画的示意曲线
score = 1.0 - 0.55 * np.exp(-eta / 0.06) - 0.45 * (eta / 0.95) ** 2.2
ax2.plot(eta, score, color=DARK, lw=2.8)
ax2.plot([0.001], [score[0]], "o", ms=10, color=RED)
ax2.annotate("$\\eta=0$:早期快但【不稳】\n(论文:destabilizes)", xy=(0.02, score[0]),
             xytext=(0.16, 0.34), fontsize=8.4, color=RED,
             arrowprops=dict(arrowstyle="-|>", lw=1.2, color=RED))
i_best = int(np.argmax(score))
ax2.plot([eta[i_best]], [score[i_best]], "o", ms=10, color=GREEN)
ax2.annotate("小→大 调度:\n平衡点", xy=(eta[i_best], score[i_best]),
             xytext=(0.42, 0.99), fontsize=8.4, color=GREEN,
             arrowprops=dict(arrowstyle="-|>", lw=1.2, color=GREEN))
ax2.plot([0.9], [score[np.argmin(np.abs(eta - 0.9))]], "o", ms=10, color=ORANGE)
ax2.annotate("$\\eta=0.9$:太陈旧 ⇒ 【收敛慢】", xy=(0.9, score[np.argmin(np.abs(eta - 0.9))]),
             xytext=(0.30, 0.72), fontsize=8.4, color=ORANGE,
             arrowprops=dict(arrowstyle="-|>", lw=1.2, color=ORANGE))
ax2.set_xlabel("$\\eta$(软更新率 ⇒ off-policy 程度)", fontsize=10.2)
ax2.set_ylabel("训练效果(示意)", fontsize=10.0)
ax2.set_title("(b) 反证:如果 off-policy 无所谓,\n$\\eta$ 的消融就不会两端都差", fontsize=11.0)
ax2.set_yticks([]); ax2.tick_params(labelsize=9)
ax2.set_ylim(0.25, 1.12)
ax2.text(0.985, 0.03, "示意曲线(据论文文字描述)", fontsize=7.6, color=GREY,
         va="bottom", ha="right", transform=ax2.transAxes)

# ---------------- (c) 代价对照
ax3 = fig.add_axes([0.79, 0.10, 0.20, 0.78]); ax3.axis("off")
ax3.text(0.5, 1.00, "(c) 代价对照", ha="center", va="top", fontsize=11.6, color=DARK)
rows = [["", "代价", "怎么处理"],
        ["GRPO", "$\\rho$ 失配\n估计有偏/爆", "clip+KL\n压回 on-policy"],
        ["NFT", "方向过时\n更新有偏", "$\\eta/\\beta$\n主动调节\n并接受偏差"]]
tb = ax3.table(cellText=rows, cellLoc="center", bbox=[0.0, 0.52, 1.0, 0.40])
tb.auto_set_font_size(False); tb.set_fontsize(7.8)
for (r_, c_), cell in tb.get_celld().items():
    if r_ == 0:
        cell.set_facecolor("#eef2f7")
    if c_ == 1 and r_ == 1:
        cell.set_facecolor("#fdecea")
    if c_ == 1 and r_ == 2:
        cell.set_facecolor("#eef7f1")
ax3.text(0.0, 0.46,
         "两个都【有代价】,\n只是形式不同:\n"
         "一个\"必须压制\",\n一个\"接受偏差\"。\n\n"
         "而且 NFT 的偏差\n本身就是\n【off-policy 造成的】。",
         fontsize=8.4, color=DARK, va="top", bbox=dict(fc="#fffbe6", ec="#e0c060"))
ax3.text(0.0, 0.10,
         "概念澄清:\nNFT 里\"on-policy\"只\n能从【结构】说\n(采集策略=训练策略?),\n不是从【估计量】说\n(没有 $\\rho$ 这个量)。",
         fontsize=8.0, color=GREEN, va="bottom", bbox=dict(fc="#eef7f1", ec="#a9d5bb"))

fig.suptitle("NFT 就没有 on/off-policy 问题了吗?—— 不是\"没有\",而是从【约束】变成【旋钮】,代价从\"会失效\"变成\"有偏差\"",
             fontsize=11.2, y=0.99)
fig.savefig(RF + "fg_nft_onpolicy.png", bbox_inches="tight")
print("wrote", RF + "fg_nft_onpolicy.png")
