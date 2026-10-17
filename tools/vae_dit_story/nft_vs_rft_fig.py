r"""DiffusionNFT 更像 RFT 吗?—— 是近亲,而且论文自己就说 RFT 是它的"正样本特例"。

  RFT(Rejection FineTuning):采 K 个 → 打分 → 【只留好的】→ 对好的做 FM 回归,坏的丢掉
      L_RFT = Σ_{i: 好} ‖v_θ − v_i‖²

  NFT:不丢坏的,而是【正着用好的、反着用坏的】,并用锚点控制步长
      L = Σ_i [ r_i‖v⁺_θ − v_i‖² + (1−r_i)‖v⁻_θ − v_i‖² ]
            └≈RFT(拉向好样本)┘   └≈反RFT(推离坏样本)┘

  论文原话:
    "previous work ... performs diffusion training solely on D+, known as RFT.
     Despite the simplicity, RFT cannot effectively leverage negative data"
    "We posit that negative feedback is crucial to policy improvement."
  消融:"Without the negative policy loss ... rewards collapse almost instantly"
  并指出这与 LLM 相反:"in LLMs, RFT remains a strong baseline"
"""
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
for _f in ["/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
           "/System/Library/Fonts/STHeiti Medium.ttc"]:
    try:
        font_manager.fontManager.addfont(_f)
        matplotlib.rcParams["font.family"] = matplotlib.font_manager.FontProperties(fname=_f).get_name()
        break
    except Exception:
        continue
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

RED, BLUE, GREEN, DARK, GREY, ORANGE, PURPLE = ("#c0392b", "#2471a3", "#1e8449",
                                                "#2c3e50", "#95a5a6", "#b9770e", "#7d3c98")
RF = "/Users/kevin/code/lingbot-va/.claude/scratch/rf/"

fig = plt.figure(figsize=(15.2, 5.8), dpi=150)

# ---------------- (a) 损失对照
ax = fig.add_axes([0.03, 0.08, 0.32, 0.82]); ax.axis("off")
ax.text(0.5, 1.00, "(a) RFT 与 NFT:差在哪", ha="center", va="top", fontsize=12, color=DARK)
ax.add_patch(plt.Rectangle((0.02, 0.68), 0.96, 0.24, fc="#f7f9fb", ec=GREY, lw=1.6))
ax.text(0.05, 0.875, "RFT(只用正样本)", fontsize=10.2, color=GREY, va="top")
ax.text(0.05, 0.795, "$\\mathcal{L}=\\sum_{i\\,:\\,r_i>{\\rm thr}}\\|v_\\theta-v_i\\|^2$",
        fontsize=10.2, color=DARK, va="top")
ax.text(0.05, 0.705, "坏样本【丢弃】⇒ 对坏区域没有梯度\n硬 0/1 筛选;无锚点;通常不算 RL",
        fontsize=8.6, color=GREY, va="top")
ax.add_patch(plt.Rectangle((0.02, 0.30), 0.96, 0.32, fc="#eef7f1", ec=GREEN, lw=1.8))
ax.text(0.05, 0.595, "NFT(= RFT + 负分支 + 锚点 + 软权重)", fontsize=10.2, color=GREEN, va="top")
ax.text(0.05, 0.525,
        "$\\mathcal{L}=\\sum_i r_i\\|v^+_\\theta-v_i\\|^2+(1-r_i)\\|v^-_\\theta-v_i\\|^2$\n\n"
        "· 好样本:拉向它(≈ RFT)\n"
        "· 坏样本:绕锚点【反射】着推离它\n"
        "· $r_i\\in[0,1]$ 是【软】权重,不是硬筛选",
        fontsize=8.8, color=DARK, va="top")
ax.text(0.5, 0.18,
        "论文自己的定位:RFT 是 NFT 的\n【正样本特例】;NFT 的贡献就是\n\"别把负样本丢了\"",
        fontsize=9.2, color=RED, ha="center", va="center",
        bbox=dict(fc="#fdecea", ec="#e6a9a0"))
ax.set_xlim(0, 1); ax.set_ylim(0.05, 1.02)

# ---------------- (b) 系谱
ax2 = fig.add_axes([0.38, 0.08, 0.30, 0.82]); ax2.axis("off")
ax2.text(0.5, 1.00, "(b) 系谱:每步加了什么", ha="center", va="top", fontsize=11.8, color=DARK)
ys = [0.78, 0.54, 0.30]
items = [("SFT / 自训练", "对【所有】采样数据做 FM 回归\n(奖励完全不用)", GREY, "#f7f9fb"),
         ("RFT", "只用【好】样本;坏的丢掉\n(硬筛选)", BLUE, "#eaf2fa"),
         ("RWR", "用连续权重 $w=\\exp(R/\\beta)$\n但权重【全是正的】", PURPLE, "#f4ecf7"),
         ("DiffusionNFT", "软权重 + 【负分支】+ 锚点\n= 有方向,不只是有目标", GREEN, "#eef7f1")]
y = 0.86
for name, desc, col, bg in items:
    ax2.add_patch(plt.Rectangle((0.03, y - 0.15), 0.94, 0.15, fc=bg, ec=col, lw=1.7))
    ax2.text(0.07, y - 0.045, name, fontsize=10.0, color=col, va="center")
    ax2.text(0.07, y - 0.108, desc, fontsize=8.2, color=DARK, va="center")
    if y > 0.34:
        ax2.add_patch(FancyArrowPatch((0.5, y - 0.155), (0.5, y - 0.195),
                                      arrowstyle="-|>", mutation_scale=11, lw=1.6, color=DARK))
    y -= 0.20
ax2.text(0.5, 0.06,
         "所以更准的说法:\nNFT ≈ (在线)RWR + 负分支 + 锚点步长控制",
         fontsize=9.0, color=DARK, ha="center", va="center",
         bbox=dict(fc="#fffbe6", ec="#e0c060"))
ax2.set_xlim(0, 1); ax2.set_ylim(0.02, 1.02)

# ---------------- (c) 为什么 LLM 里 RFT 行、扩散里崩
ax3 = fig.add_axes([0.71, 0.08, 0.27, 0.82]); ax3.axis("off")
ax3.text(0.5, 1.00, "(c) 为什么 LLM 行、扩散崩", ha="center", va="top", fontsize=11.4, color=DARK)
ax3.text(0.0, 0.92,
         "论文实证(消融):\n"
         "去掉负分支 ⇒ 奖励【几乎立刻崩溃】\n"
         "并指出这与 LLM 相反(RFT 在 LLM\n"
         "仍是强 baseline)",
         fontsize=8.8, color=RED, va="top", bbox=dict(fc="#fdecea", ec="#e6a9a0"))
ax3.text(0.0, 0.68,
         "我的机制解释(推断):\n\n"
         "LLM:SFT 在好回答上,softmax\n"
         "的归一化让\"提高好 token\"\n"
         "【同时精确压低其他 token】\n"
         "⇒ 天然带负反馈",
         fontsize=8.6, color=DARK, va="top")
ax3.text(0.0, 0.40,
         "扩散:FM 损失是【逐点局部回归】\n"
         "$\\|v_\\theta(x_t,t)-v\\|^2$ 在好样本附近的点上\n"
         "降误差,对别处的坏区域\n"
         "【没有任何直接梯度】\n"
         "⇒ 坏模式可以一直留着",
         fontsize=8.6, color=GREEN, va="top", bbox=dict(fc="#eef7f1", ec=GREEN))
ax3.text(0.0, 0.10,
         "⇒ 所以\"丢掉坏的\"在扩散里\n不等于\"压低了坏的\"。",
         fontsize=8.8, color=DARK, va="top", bbox=dict(fc="#fffbe6", ec="#e0c060"))

fig.suptitle("DiffusionNFT 更像 RFT 吗?—— 是它的推广:RFT 只学好的(丢掉坏的),NFT 把坏的当【反向目标】,并用锚点控制步长",
             fontsize=11.2, y=0.99)
fig.savefig(RF + "fg_nft_vs_rft.png", bbox_inches="tight")
print("wrote", RF + "fg_nft_vs_rft.png")
