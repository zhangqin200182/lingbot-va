r"""η 怎么"控制"off-policy?没有重要性采样,是不是就拍一个值?

  两件事要拆开:
    ① η 控制 off-policy 的【程度】—— 机制问题,可以精确量化
         v_old ← η v_old + (1−η) v_θ
         ⇒ 采集策略(也是锚点)平均【落后】η/(1−η) 次迭代
         η=0.001 → 几乎不落后;0.5 → 1 次;0.9 → 9 次;0.999 → 999 次
    ② 没有重要性采样 ⇒ 决定【不修正】这个程度 —— 设计选择

  为什么可以不要 IS:
      IS 是为了让"策略梯度估计"【无偏】(把旧数据按 π_new/π_old 重新加权)
      而 NFT 的目标不是"估计策略梯度",是"【构造】一个改进方向"(Theorem 3.1)
      ⇒ 不追求无偏,就不需要 IS
  ⇒ 所以 η 确实是启发式超参(靠消融定界),
     but 它的作用是可解释、可量化的:控制"数据/锚点落后多少" ⇒ 保守程度
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

fig = plt.figure(figsize=(15.2, 5.9), dpi=150)

# ---------------- (a) 两个问题拆开
ax = fig.add_axes([0.03, 0.08, 0.32, 0.82]); ax.axis("off")
ax.text(0.5, 1.00, "(a) 两个问题,别混", ha="center", va="top", fontsize=12, color=DARK)
ax.add_patch(plt.Rectangle((0.02, 0.58), 0.96, 0.34, fc="#eef7f1", ec=GREEN, lw=1.8))
ax.text(0.05, 0.875, "① η 控制【程度】—— 机制问题", fontsize=10.2, color=GREEN, va="top")
ax.text(0.05, 0.795,
        "$v_{\\rm old}\\leftarrow\\eta v_{\\rm old}+(1-\\eta)v_\\theta$\n\n"
        "⇒ 采集策略(=锚点)平均【落后】训练策略\n"
        "　 $\\dfrac{\\eta}{1-\\eta}$ 次迭代(可精确量化)\n\n"
        "η=0.001 → 几乎不落后\n"
        "η=0.5 → 1 次;η=0.9 → 9 次\n"
        "η=0.999(OCR 用)→ 999 次",
        fontsize=8.8, color=DARK, va="top")
ax.add_patch(plt.Rectangle((0.02, 0.22), 0.96, 0.30, fc="#fdf3e7", ec=ORANGE, lw=1.8))
ax.text(0.05, 0.495, "② 没有 IS ⇒ 决定【不修正】—— 选择问题", fontsize=10.2, color=ORANGE, va="top")
ax.text(0.05, 0.415,
        "有 IS 的做法:把旧数据按 $\\pi_{\\rm new}/\\pi_{\\rm old}$\n"
        "重新加权,让更新在期望上等于 on-policy\n\n"
        "NFT:不修正 ⇒ 优化的是\"旧策略给出的方向\"\n"
        "　 ⇒ 有偏,但仍是下降方向(能推进)",
        fontsize=8.8, color=DARK, va="top")
ax.text(0.5, 0.10,
        "两个说法不冲突:\nη 决定\"有多 off-policy\";无 IS 决定\"不修正它\"",
        fontsize=9.2, color=DARK, ha="center", va="center",
        bbox=dict(fc="#fffbe6", ec="#e0c060"))
ax.set_xlim(0, 1); ax.set_ylim(0.03, 1.02)

# ---------------- (b) 滞后曲线
ax2 = fig.add_axes([0.40, 0.17, 0.28, 0.66])
eta = np.linspace(0, 0.995, 500)
lag = eta / (1 - eta)
ax2.plot(eta, lag, color=GREEN, lw=2.8)
for e, lab, c in [(0.001, "论文起始\n$\\eta_i=0.001i$", BLUE), (0.5, "论文上限 0.5", PURPLE),
                  (0.9, "消融:太慢", RED), (0.999, "OCR 用 0.999", ORANGE)]:
    ax2.plot([e], [e / (1 - e)], "o", ms=9, color=c)
    ax2.annotate(lab, xy=(e, e / (1 - e)), xytext=(e + 0.02, e / (1 - e) + 25),
                 fontsize=8.0, color=c, arrowprops=dict(arrowstyle="-|>", lw=1.0, color=c))
ax2.set_xlabel("$\\eta$(软更新率)", fontsize=10.4)
ax2.set_ylabel("平均滞后 $\\eta/(1-\\eta)$(次迭代)", fontsize=10.2)
ax2.set_title("(b) η 就是\"采集策略落后多少\"的旋钮\n(实测与公式完全一致)", fontsize=11.0)
ax2.tick_params(labelsize=9)
ax2.set_ylim(0, 260)
ax2.text(0.02, 0.75, "滞后小 ⇒ 数据/锚点新 ⇒ 激进\n滞后大 ⇒ 数据/锚点旧 ⇒ 保守",
         fontsize=8.6, color=DARK, va="top", bbox=dict(fc="#eef7f1", ec="#a9d5bb"),
         transform=ax2.transAxes)

# ---------------- (c) 为什么能不要 IS + 诚实结论
ax3 = fig.add_axes([0.72, 0.08, 0.27, 0.82]); ax3.axis("off")
ax3.text(0.5, 1.00, "(c) 为什么能不要 IS", ha="center", va="top", fontsize=11.6, color=DARK)
ax3.text(0.0, 0.92,
         "IS 是为谁服务的?\n"
         "　为了让【策略梯度的估计】无偏\n"
         "　(把旧数据按比值重加权)\n\n"
         "NFT 要的是什么?\n"
         "　不是\"无偏估计\",而是\"【构造】一个\n"
         "　 改进方向\"(Theorem 3.1)\n\n"
         "⇒ 不追求无偏 ⇒ 不需要 IS",
         fontsize=9.0, color=DARK, va="top")
ax3.text(0.0, 0.50,
         "诚实结论:\n"
         "· 是的,$\\eta$ 本质是【启发式超参】\n"
         "　 (靠消融定界:$\\eta=0$ 不稳、0.9 太慢)\n"
         "· 但它的作用【可解释、可量化】:\n"
         "　 控制数据/锚点落后多少\n"
         "· 与 GRPO 的 clip/KL 不同:\n"
         "　 后者是在【有保证的框架】里做受控近似;\n"
         "　 $\\eta$ 是在【没有无偏保证的框架】里\n"
         "　 调启发式强度",
         fontsize=8.6, color=DARK, va="top", bbox=dict(fc="#fffbe6", ec="#e0c060"))
ax3.text(0.0, 0.06,
         "\"拍一个值\"这句判断是对的,\n只是它不是随便拍:它等价于\n声明\"我愿意用多旧的数据\"。",
         fontsize=8.6, color=GREEN, va="bottom", bbox=dict(fc="#eef7f1", ec="#a9d5bb"))

fig.suptitle("η 怎么控制 off-policy:它调的是\"采集策略落后多少次迭代\"($\\eta/(1-\\eta)$);而\"没有 IS\"是【不修正】这个滞后",
             fontsize=11.2, y=0.99)
fig.savefig(RF + "fg_nft_eta_offpolicy.png", bbox_inches="tight")
print("wrote", RF + "fg_nft_eta_offpolicy.png")
