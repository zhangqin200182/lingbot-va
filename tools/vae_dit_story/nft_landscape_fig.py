r"""GRPO 啰嗦(要似然)、NFT 有偏(无 IS)—— 有没有更好的算法?

  根本原因:你手里只有两种信息
    (a) "这条轨迹在当前策略下的概率"      → 能算【无偏】策略梯度(要似然 ⇒ 啰嗦)
    (b) "这个样本好不好"(标量奖励)        → 只能构造【有偏】代理目标(便宜)
  想在 (b) 上拿无偏梯度,只剩"扰动参数看奖励"(零阶/ES)
    ⇒ 但它的样本复杂度随参数维度爆炸 ⇒ 只能在小维度(LoRA 子空间)用

  五条路线:
    ① 策略梯度 GRPO/DDPO        无偏  要似然   方差大
    ② 监督式重加权 RWR/NFT       有偏  不要似然  便宜、稳、快
    ③ 可微奖励反传 DRaFT/ReFL    无偏  不要似然  但要奖励可微 + 显存贵
    ④ 零阶 / ES(参数扰动)       无偏  不要似然  但方差随维度爆炸
    ⑤ 控制/最优传输 SOC/adjoint  回归式构造 ⇒ 又回到 ② 的偏差问题
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

# ---------------- (a) 信息层面的两难
ax = fig.add_axes([0.03, 0.08, 0.32, 0.82]); ax.axis("off")
ax.text(0.5, 1.00, "(a) 两难来自信息,不是算法", ha="center", va="top", fontsize=11.8, color=DARK)
ax.text(0.0, 0.92,
         "你手里只有两种信息:\n\n"
         "(a) 这条轨迹在当前策略下的【概率】\n"
         "　 ⇒ 能算【无偏】策略梯度\n"
         "　 ⇒ 但必须算似然 ⇒ \"啰嗦\"\n\n"
         "(b) 这个样本好不好(标量奖励)\n"
         "　 ⇒ 只能构造【有偏】代理目标\n"
         "　 ⇒ 但便宜、稳、快",
         fontsize=9.2, color=DARK, va="top")
ax.text(0.0, 0.46,
         "想在 (b) 上拿到无偏梯度?\n"
         "只剩\"扰动参数、看奖励变化\"(零阶/ES)\n"
         "⇒ 样本复杂度随参数维度爆炸\n"
         "⇒ 只能在小维度(LoRA 子空间)用",
         fontsize=9.0, color=ORANGE, va="top", bbox=dict(fc="#fdf3e7", ec=ORANGE))
ax.text(0.0, 0.18,
         "⇒ 所以\"不要似然\"与\"无偏\"\n在只给标量奖励时,基本不可兼得。",
         fontsize=9.2, color=RED, va="top", bbox=dict(fc="#fdecea", ec="#e6a9a0"))
ax.set_xlim(0, 1); ax.set_ylim(0.03, 1.02)

# ---------------- (b) 五条路线取舍表
ax2 = fig.add_axes([0.38, 0.08, 0.34, 0.82]); ax2.axis("off")
ax2.text(0.5, 1.00, "(b) 五条路线的取舍", ha="center", va="top", fontsize=11.8, color=DARK)
rows = [["路线", "无偏?", "要似然?", "其他代价"],
        ["① 策略梯度\nGRPO / DDPO", "✓", "✓", "方差大\n绑 solver + CFG"],
        ["② 监督重加权\nRWR / NFT", "✗", "✗", "温度/锚点\n靠启发式"],
        ["③ 可微奖励反传\nDRaFT / ReFL", "✓", "✗", "奖励必须可微\n显存贵(整条链)"],
        ["④ 零阶 / ES\n(参数扰动)", "✓", "✗", "方差随维度爆炸\n只适合小维度"],
        ["⑤ 控制 / 最优传输\nSOC / adjoint", "近似", "✗", "回归式构造\n回到②的偏差"]]
tb = ax2.table(cellText=rows, cellLoc="center", bbox=[0.0, 0.40, 1.0, 0.52])
tb.auto_set_font_size(False); tb.set_fontsize(7.8)
for (r_, c_), cell in tb.get_celld().items():
    if r_ == 0:
        cell.set_facecolor("#eef2f7")
    if c_ == 1 and r_ > 0:
        t = cell.get_text().get_text()
        cell.set_facecolor("#eef7f1" if t.startswith("✓") else "#fdecea")
    if c_ == 2 and r_ > 0:
        cell.set_facecolor("#fdecea" if cell.get_text().get_text() == "✓" else "#eef7f1")
ax2.text(0.0, 0.34,
         "★ 关键:③ 是唯一【同时】避开两个问题的路线 ——\n"
         "　 无偏、不要似然;代价换成\"奖励必须可微 + 显存\"。\n\n"
         "　 所以如果你的奖励可微(CLIP、可微渲染、物理),\n"
         "　 就已经有\"更好的算法\"了。",
         fontsize=8.8, color=GREEN, va="top", bbox=dict(fc="#eef7f1", ec=GREEN))
ax2.text(0.0, 0.10,
         "④ 是理论上的另一个出口,但工程上只在小维度可行。",
         fontsize=8.4, color=DARK, va="top", bbox=dict(fc="#fffbe6", ec="#e0c060"))

# ---------------- (c) 实践建议
ax3 = fig.add_axes([0.735, 0.08, 0.25, 0.82]); ax3.axis("off")
ax3.text(0.5, 1.00, "(c) 实践上怎么选", ha="center", va="top", fontsize=11.6, color=DARK)
ax3.text(0.0, 0.92,
         "按【奖励的可微性】分岔:\n\n"
         "奖励可微 ⇒ 走 ③\n"
         "　 无偏、无似然、无方差\n"
         "　 代价:显存 + 采样链要可微\n\n"
         "奖励黑盒 + 要理论保证 ⇒ ①\n"
         "　 接受啰嗦与方差\n\n"
         "奖励黑盒 + 要简单快稳 ⇒ ②\n"
         "　 接受偏差(但实测很能打)",
         fontsize=8.8, color=DARK, va="top")
ax3.text(0.0, 0.36,
         "三条\"合成\"路线(已有实践):\n"
         "· 可微奖励当辅助项\n"
         "　$\\mathcal{L}=\\mathcal{L}_{\\rm RL}+\\lambda(-R_{\\rm diff})$\n"
         "· 两阶段:先②对齐,再①精修\n"
         "· ② 之后蒸馏到少步\n"
         "　 (回归式目标天然适合蒸馏)",
         fontsize=8.4, color=DARK, va="top", bbox=dict(fc="#fffbe6", ec="#e0c060"))
ax3.text(0.0, 0.06,
         "想给②补 IS?\n就又要似然 ⇒ 回到①",
         fontsize=8.6, color=RED, va="bottom", bbox=dict(fc="#fdecea", ec="#e6a9a0"))

fig.suptitle("有没有更好的算法?—— 没有\"既要又要\"的通用解(信息层面的取舍);但换掉一个约束(奖励可微 / 参数维度小)就有更好的解",
             fontsize=11.2, y=0.99)
fig.savefig(RF + "fg_algorithm_landscape.png", bbox_inches="tight")
print("wrote", RF + "fg_algorithm_landscape.png")
