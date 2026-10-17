r"""SFT 没有 on/off-policy 的问题,为什么 NFT 有?—— 因为 on/off-policy 是 RL 独有的概念。

  SFT: L(θ) = E_{x ~ D}[ ℓ(f_θ(x), y) ]        D 是【给定的数据集】
       ⇒ 没有"策略"、没有"谁产生的数据"这个问题
       ⇒ on/off-policy 【无从定义】

  RL : J(θ) = E_{x ~ π_θ}[ R(x) ]              期望在【θ 自己的分布】下取
       ⇒ 若数据来自 π_old ≠ π_θ,sample average 估的【不是 J(θ)】
       ⇒ 需要重要性权重 π_θ/π_old ⇒ 这才有 on/off-policy

  NFT: L = Σ_i [ r_i‖v⁺−v_i‖² + (1−r_i)‖v⁻−v_i‖² ]
       形式上 = SFT(拟合一个奖励加权的数据集)⇒ 按此读法【没有】on/off-policy 问题
       意图上 = RL(想提升 E_{π_θ}[R])        ⇒ 按此读法【有偏】

  ⇒ 同一个"陈旧":
       SFT 的尺子:目标漂移(我拟合的分布过时了)
       RL  的尺子:off-policy 偏差(数据分布 ≠ 当前策略)
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

RED, BLUE, GREEN, DARK, GREY, ORANGE, PURPLE = ("#c0392b", "#2471a3", "#1e8449",
                                                "#2c3e50", "#95a5a6", "#b9770e", "#7d3c98")
RF = "/Users/kevin/code/lingbot-va/.claude/scratch/rf/"

fig = plt.figure(figsize=(15.2, 5.4), dpi=150)

# ---------------- (a) 三者的目标式
ax = fig.add_axes([0.03, 0.08, 0.44, 0.80]); ax.axis("off")
ax.text(0.5, 1.00, "(a) 关键差别:期望在【谁的分布】下取", ha="center", va="top",
        fontsize=12, color=DARK)
items = [
    ("SFT(监督微调)", "L(θ) = E_{x ~ D}[ ℓ(f_θ(x), y) ]",
     "D 是【给定的数据集】⇒ 没有\"策略\"\n⇒ on/off-policy 【无从定义】", GREY, "#f7f9fb"),
    ("RL(策略梯度)", "J(θ) = E_{x ~ π_θ}[ R(x) ]",
     "期望在【θ 自己的分布】下取\n⇒ 数据来自 π_old ≠ π_θ 时\n   sample average 估的不是 J(θ)\n⇒ 需要重要性权重 ⇒ 才有 on/off-policy",
     RED, "#fdecea"),
    ("DiffusionNFT", "L = Σ_i [ r_i‖v⁺−v_i‖² + (1−r_i)‖v⁻−v_i‖² ]",
     "形式上 = SFT(拟合奖励加权的数据集)\n意图上 = RL(想提升 E_{π_θ}[R])\n⇒ 两种读法,两种结论",
     GREEN, "#eef7f1"),
]
y = 0.90
for name, formula, note, col, bg in items:
    ax.add_patch(plt.Rectangle((0.02, y - 0.27), 0.96, 0.27, fc=bg, ec=col, lw=1.8))
    ax.text(0.05, y - 0.035, name, fontsize=10.4, color=col, va="top")
    ax.text(0.05, y - 0.085, "$" + formula + "$", fontsize=9.4, color=DARK, va="top")
    ax.text(0.05, y - 0.135, note, fontsize=8.4, color=DARK, va="top")
    y -= 0.305
ax.set_xlim(0, 1); ax.set_ylim(0.02, 1.02)

# ---------------- (b) 两种尺子
ax2 = fig.add_axes([0.51, 0.08, 0.23, 0.80]); ax2.axis("off")
ax2.text(0.5, 1.00, "(b) 同一个\"陈旧\",两种叫法", ha="center", va="top", fontsize=11.4, color=DARK)
rows = [["用哪把尺子", "叫什么", "后果"],
        ["SFT 的尺子\n(我在拟合数据)", "目标漂移", "不会失效\n只是拟合的\n目标过时"],
        ["RL 的尺子\n(我要提升 E[R])", "off-policy\n偏差", "不是 J(θ)\n的梯度"]]
tb = ax2.table(cellText=rows, cellLoc="center", bbox=[0.0, 0.46, 1.0, 0.46])
tb.auto_set_font_size(False); tb.set_fontsize(8.0)
for (r_, c_), cell in tb.get_celld().items():
    if r_ == 0:
        cell.set_facecolor("#eef2f7")
    if c_ == 1 and r_ == 1:
        cell.set_facecolor("#f7f9fb")
    if c_ == 1 and r_ == 2:
        cell.set_facecolor("#fdecea")
ax2.text(0.0, 0.40,
         "术语本身就是线索:\n"
         "· RL 说 on/off-policy\n"
         "· SFT 说 distribution shift /\n"
         "　 mode collapse\n\n"
         "NFT 的两侧风险:\n"
         "· SFT 侧:分布漂移 / 模式坍塌\n"
         "· RL  侧:off-policy 偏差",
         fontsize=8.4, color=DARK, va="top", bbox=dict(fc="#fffbe6", ec="#e0c060"))

# ---------------- (c) 为什么我还是标"有偏"
ax3 = fig.add_axes([0.77, 0.08, 0.21, 0.80]); ax3.axis("off")
ax3.text(0.5, 1.00, "(c) 那为什么还说它有偏?", ha="center", va="top", fontsize=11.2, color=DARK)
ax3.text(0.0, 0.92,
         "因为【论文自己用了 RL 的架子】:\n\n"
         "· 标题:Online Diffusion Reinforcement\n"
         "· 有 old policy 的 EMA($v_{\\rm old}$)\n"
         "· 有组内优势($2r-1$)\n"
         "· 有\"policy improvement\"的说法\n\n"
         "⇒ 用了 RL 的脚手架,\n"
         "　 就该接受 RL 的告诫:\n"
         "　 没有 IS ⇒ 不是无偏的策略梯度",
         fontsize=8.6, color=DARK, va="top")
ax3.text(0.0, 0.30,
         "反过来说:\n"
         "如果它老老实实叫\n"
         "\"奖励加权的自我训练\",\n"
         "就没人会提 off-policy。",
         fontsize=8.6, color=GREEN, va="top", bbox=dict(fc="#eef7f1", ec="#a9d5bb"))
ax3.text(0.0, 0.08,
         "所以\"有偏\"不是凭空加的,\n而是【借用 RL 语言】的代价。",
         fontsize=8.4, color=RED, va="bottom", bbox=dict(fc="#fdecea", ec="#e6a9a0"))

fig.suptitle("SFT 为什么没有 on/off-policy 问题?因为 SFT 里没有\"策略\"—— 期望是在【给定数据集】上取的,不是在【自己的分布】上取的",
             fontsize=11.0, y=0.99)
fig.savefig(RF + "fg_why_sft_no_offpolicy.png", bbox_inches="tight")
print("wrote", RF + "fg_why_sft_no_offpolicy.png")
