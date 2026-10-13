r"""DiffusionNFT 如何支持 off-policy?

  在 NFT 里 "off-policy" 具体指:
    数据由【采集策略 v_old】产生,而训练在更新【v_θ】
    一旦 v_θ 偏离 v_old,后续梯度步用的就是"旧策略采的数据"
    ⇒ 偏离程度 = off-policy 程度,由软更新率 η 控制

  为什么它【天然】支持(不像策略梯度那样怕陈旧):
    ① 损失里【没有 ratio】(不需要 π_new/π_old)
       L = r‖v⁺_θ−v‖² + (1−r)‖v⁻_θ−v‖²  只用 v_θ、v_old、v、r
    ② 目标【有界】:T± = v_old ± (v−v_old)/β,陈旧只让 v_old 落后,目标仍是有限值
       (而重要性权重 ρ 在陈旧时会指数级跑偏 ⇒ 必须靠 clip/KL 兜)
    ③ 只需存 (x₀, r) ⇒ 可放进 buffer 反复回归(天然支持 replay)
    ④ η 就是 off-policy 程度的旋钮:v_old ← η v_old + (1−η)v_θ

  代价:没有重要性修正 ⇒ 方向会"过时"(收敛慢,η=0.9 的观察)
  副产品:锚点 v_old 陈旧 ⇒ 更新被锚在旧策略附近 = 【隐式 KL/信赖域】
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

# ---------------- (a) off-policy 指什么
ax = fig.add_axes([0.03, 0.08, 0.33, 0.82]); ax.axis("off")
ax.text(0.5, 1.00, "(a) NFT 里的 off-policy 指什么", ha="center", va="top", fontsize=11.8, color=DARK)
ax.text(0.0, 0.92,
         "两个角色分开:\n\n"
         "　采集策略(产生数据):$v_{\\rm old}$\n"
         "　训练策略(被更新):$v_\\theta$\n\n"
         "θ 每走一步,$v_\\theta$ 就离 $v_{\\rm old}$ 更远;\n"
         "而数据仍是 $v_{\\rm old}$ 采的 ⇒ 【off-policy】\n\n"
         "偏离程度由软更新控制:\n"
         "　$v_{\\rm old}\\leftarrow\\eta v_{\\rm old}+(1-\\eta)v_\\theta$",
         fontsize=9.2, color=DARK, va="top")
ax.text(0.0, 0.42,
         "$\\eta=0$:每步把锚点重置成当前策略\n"
         "　　　⇒ 数据最新、锚点最新 ⇒ 最 on-policy\n"
         "$\\eta\\to1$:锚点几乎不动\n"
         "　　　⇒ 数据旧、锚点旧 ⇒ 最 off-policy",
         fontsize=9.0, color=GREEN, va="top", bbox=dict(fc="#eef7f1", ec=GREEN))
ax.text(0.0, 0.14,
         "论文默认:$\\eta_i=\\min(0.001i,\\,0.5)$\n(由小到大:先激进,后保守)",
         fontsize=9.0, color=DARK, va="top", bbox=dict(fc="#fffbe6", ec="#e0c060"))
ax.set_xlim(0, 1); ax.set_ylim(0.03, 1.02)

# ---------------- (b) 为什么回归天然容忍陈旧
ax2 = fig.add_axes([0.39, 0.08, 0.31, 0.82]); ax2.axis("off")
ax2.text(0.5, 1.00, "(b) 为什么监督式回归不怕陈旧", ha="center", va="top", fontsize=11.6, color=DARK)
rows = [["", "策略梯度(GRPO)", "NFT"],
        ["用 ratio 吗", "用 $\\rho=\\pi_{\\rm new}/\\pi_{\\rm old}$", "不用"],
        ["陈旧时", "$\\rho$ 指数级跑偏\n⇒ 必须 clip/KL 兜", "只是目标参照\n旧,仍是有限值"],
        ["失效形式", "比值爆掉/估计有偏", "方向\"过时\"\n(不会爆)"],
        ["复用数据", "受 inner_epochs\n限制", "buffer 反复\n回归都行"],
        ["陈旧程度", "误差来源\n(要压制)", "可调超参\n($\\eta$)"]]
tb = ax2.table(cellText=rows, cellLoc="center", bbox=[0.0, 0.42, 1.0, 0.50])
tb.auto_set_font_size(False); tb.set_fontsize(7.8)
for (r_, c_), cell in tb.get_celld().items():
    if r_ == 0:
        cell.set_facecolor("#eef2f7")
    if c_ == 1 and r_ > 0:
        cell.set_facecolor("#fdecea")
    if c_ == 2 and r_ > 0:
        cell.set_facecolor("#eef7f1")
ax2.text(0.0, 0.36,
         "一句话:ratio 是\"必须新鲜的量\",\n"
         "回归目标是\"不新鲜也仍然有定义的量\"。\n\n"
         "所以 NFT 的 off-policy 不是\"勉强支持\",\n"
         "而是它的监督本质【自带】的能力。",
         fontsize=8.8, color=DARK, va="top", bbox=dict(fc="#fffbe6", ec="#e0c060"))
ax2.text(0.0, 0.10,
         "代价:没有重要性修正 ⇒ 严格说\n优化的不是\"当前策略的期望回报\"。",
         fontsize=8.8, color=RED, va="top", bbox=dict(fc="#fdecea", ec="#e6a9a0"))

# ---------------- (c) v_old 兼作隐式 KL
ax3 = fig.add_axes([0.72, 0.08, 0.27, 0.82]); ax3.axis("off")
ax3.text(0.5, 1.00, "(c) 副产品:$v_{\\rm old}$ 兼作\"隐式 KL\"", ha="center", va="top",
         fontsize=11.2, color=DARK)
ax3.text(0.0, 0.92,
         "锚点陈旧 ⇒ 所有目标都写成\n\"相对旧策略该往哪改\"\n\n"
         "$T=v_{\\rm old}+(2r-1)\\dfrac{v-v_{\\rm old}}{\\beta}$\n\n"
         "⇒ 相当于在损失里放了一个\n"
         "　 \"把 θ 拉在旧策略附近\"的项\n"
         "　 (类似 KL 正则的作用)",
         fontsize=9.0, color=GREEN, va="top")
ax3.text(0.0, 0.50,
         "所以两个旋钮一起控制保守程度:\n\n"
         "　$\\eta$ 大:锚点旧 ⇒ 隐式 KL 强 ⇒ 稳\n"
         "　$\\beta$ 大:目标离锚点近 ⇒ 保守\n\n"
         "反之:η 小 / β 小 ⇒ 激进,但可能不稳",
         fontsize=9.0, color=DARK, va="top", bbox=dict(fc="#fffbe6", ec="#e0c060"))
ax3.text(0.0, 0.14,
         "对上消融:\n"
         "· $\\eta=0$ 完全 on-policy ⇒ 不稳\n"
         "· $\\eta=0.9$ 太陈旧 ⇒ 收敛慢\n"
         "· 小→大调度最好 ⇒ 先激进后稳",
         fontsize=8.6, color=DARK, va="top", bbox=dict(fc="#eef7f1", ec="#a9d5bb"))

fig.suptitle("NFT 如何支持 off-policy:它的损失里【没有 ratio】—— 陈旧只让目标\"过时\",不会让估计爆掉;而 η 就是这个程度的旋钮",
             fontsize=11.4, y=0.99)
fig.savefig(RF + "fg_nft_offpolicy.png", bbox_inches="tight")
print("wrote", RF + "fg_nft_offpolicy.png")
