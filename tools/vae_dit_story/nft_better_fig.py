r"""NFT 是不是"不会把随机的地方训坏、不会让步数变长"?

  对的地方:训练与【采集器】解耦
    · NFT 不需要似然 ⇒ 采集时可以用【你部署的那个 solver/步数】(甚至确定性 ODE)
    · 目标函数不依赖 solver 参数(GRPO 的权重 m/n 依赖步长与噪声调度)
    ⇒ 优化目标与部署设置【天然对齐】;采集数据也不会被 SDE 噪声污染
       (论文消融:ODE 采集优于 SDE,尤其 PickScore 这类对噪声敏感的奖励)

  需要纠正的地方:路径形状照样会变
    · 两者都只改 θ;θ 变了 ⇒ 场变了 ⇒ 路径形状【当然】会变
    · NFT 里没有任何机制"冻结路径几何"
    · 所以"分布被推得更偏 ⇒ 场更难积分 ⇒ 需要更多步"这个风险【两者共有】
    · 而且【两篇论文都没测"质量 vs 步数"曲线】
      (NFT 的对比固定在 10 步 ⇒ 至少"10 步下 RL 后仍好"有实证;4 步/2 步仍是空白)
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
from matplotlib.patches import FancyBboxPatch

RED, BLUE, GREEN, DARK, GREY, ORANGE, PURPLE = ("#c0392b", "#2471a3", "#1e8449",
                                                "#2c3e50", "#95a5a6", "#b9770e", "#7d3c98")
RF = "/Users/kevin/code/lingbot-va/.claude/scratch/rf/"

fig = plt.figure(figsize=(15.2, 5.9), dpi=150)

# ---------------- (a) 对的那一半 / 要纠正的那一半
ax = fig.add_axes([0.03, 0.08, 0.33, 0.82]); ax.axis("off")
ax.text(0.5, 1.00, "(a) 你的直觉:一半对", ha="center", va="top", fontsize=12, color=DARK)
ax.add_patch(FancyBboxPatch((0.02, 0.62), 0.96, 0.30, boxstyle="round,pad=0.02",
                            fc="#eef7f1", ec=GREEN, lw=1.8))
ax.text(0.05, 0.875, "✓ 对的那一半:训练与【采集器】解耦", fontsize=10.0, color=GREEN, va="top")
ax.text(0.05, 0.795,
        "· 不需要似然 ⇒ 采集时可以用【你要部署的\n"
        "　 那个 solver 和步数】(甚至确定性 ODE)\n"
        "· 目标不含 solver 参数\n"
        "　 (GRPO 的权重 $m/n$ 依赖步长与噪声调度)\n"
        "⇒ 优化目标与部署设置【天然对齐】;\n"
        "　 采集数据也不会被 SDE 噪声污染",
        fontsize=8.6, color=DARK, va="top")
ax.add_patch(FancyBboxPatch((0.02, 0.22), 0.96, 0.34, boxstyle="round,pad=0.02",
                            fc="#fdecea", ec=RED, lw=1.8))
ax.text(0.05, 0.525, "✗ 要纠正的一半:\"路径形状不变\"不成立", fontsize=10.0, color=RED, va="top")
ax.text(0.05, 0.445,
        "· 两者都【只改 θ】;θ 变了 ⇒ 场变了\n"
        "　 ⇒ 路径形状【当然】会变 —— NFT 里没有\n"
        "　 任何机制冻结路径几何\n"
        "· 所以\"分布被推得更偏 ⇒ 场更难积分 ⇒\n"
        "　 需要更多步\"这个风险【两者共有】\n"
        "· 而且【两篇论文都没测】\"质量 vs 步数\"曲线",
        fontsize=8.6, color=DARK, va="top")
ax.text(0.5, 0.10,
        "正确说法:NFT 的优势是【与采样器解耦】,\n不是\"路径不会变\"",
        fontsize=9.4, color=DARK, ha="center", va="center",
        bbox=dict(fc="#fffbe6", ec="#e0c060"))
ax.set_xlim(0, 1); ax.set_ylim(0.03, 1.02)

# ---------------- (b) "步数变长"的风险来源
ax2 = fig.add_axes([0.39, 0.08, 0.30, 0.82]); ax2.axis("off")
ax2.text(0.5, 1.00, "(b) \"步数变长\"的风险从哪来", ha="center", va="top", fontsize=11.6, color=DARK)
rows = [["风险来源", "GRPO", "NFT"],
        ["改 θ ⇒ 场/路径变", "有", "有"],
        ["目标随 solver 变\n(权重 $m/n$)", "有", "无"],
        ["采集器被迫用 SDE\n(SDE 采集质量更差)", "有", "无"],
        ["推理还得叠加 CFG\n(CFG 让场更弯)", "有", "无"]]
tb = ax2.table(cellText=rows, cellLoc="center", bbox=[0.0, 0.50, 1.0, 0.40])
tb.auto_set_font_size(False); tb.set_fontsize(8.4)
for (r_, c_), cell in tb.get_celld().items():
    if r_ == 0:
        cell.set_facecolor("#eef2f7")
    if c_ == 1 and r_ > 0:
        cell.set_facecolor("#fdecea")
    if c_ == 2 and r_ > 0:
        cell.set_facecolor("#eef7f1" if r_ > 1 else "#fdecea")
ax2.text(0.0, 0.44,
         "读法:\n"
         "· 第一行【两者共有】—— 这是\"步数可能变长\"\n"
         "　 的根本机制,谁都躲不掉\n"
         "· 后三行 NFT 确实避免了 —— 但它们影响的是\n"
         "　 \"采集质量\"与\"目标是否稳定\",\n"
         "　 不等于\"路径不会变难积分\"",
         fontsize=8.6, color=DARK, va="top", bbox=dict(fc="#fffbe6", ec="#e0c060"))
ax2.text(0.0, 0.14,
         "实证边界:\n"
         "NFT 的对比固定在 10 步 ⇒ \"10 步下 RL 后仍然好\"\n"
         "有实证;但 4 步 / 2 步仍是【空白】。",
         fontsize=8.6, color=GREEN, va="top", bbox=dict(fc="#eef7f1", ec="#a9d5bb"))

# ---------------- (c) 记分卡
ax3 = fig.add_axes([0.72, 0.08, 0.27, 0.82]); ax3.axis("off")
ax3.text(0.5, 1.00, "(c) 记分卡", ha="center", va="top", fontsize=11.8, color=DARK)
ax3.text(0.0, 0.92, "NFT 更好:", fontsize=10.2, color=GREEN, va="top")
ax3.text(0.0, 0.86,
         "· 不要似然 ⇒ 显存低、只存干净图\n"
         "· solver 任意(可用 ODE)\n"
         "· 不需要 CFG\n"
         "· 效率:1k 步 vs 5k+ 步\n"
         "· 训练更稳(方差小)",
         fontsize=8.6, color=DARK, va="top")
ax3.text(0.0, 0.62, "GRPO 更好 / NFT 更弱:", fontsize=10.2, color=RED, va="top")
ax3.text(0.0, 0.56,
         "· 理论:策略梯度定理(无偏)\n"
         "　 NFT 是构造式\"改进方向\",无无偏保证\n"
         "· NFT 是监督+RL 的混合体,不是标准 RL\n"
         "· NFT 超参更敏感(β、η、时间权重;\n"
         "　 消融:去掉负分支→奖励崩溃,\n"
         "　 时间权重反了→也崩)",
         fontsize=8.6, color=DARK, va="top")
ax3.text(0.0, 0.24,
         "两者都【没有】保证的一项:\n"
         "RL 之后少步采样质量不变。",
         fontsize=9.0, color=ORANGE, va="top", bbox=dict(fc="#fdf3e7", ec=ORANGE))
ax3.text(0.0, 0.10,
         "但 NFT 可以【直接按部署设置训练】\n(因为不依赖似然)—— 这才是它的真优势。",
         fontsize=8.8, color=GREEN, va="top", bbox=dict(fc="#eef7f1", ec="#a9d5bb"))

fig.suptitle("NFT 是不是更好?—— 在\"与采样器解耦\"上确实更好;但\"步数不会变长\"不是它的保证(两者共有此风险,且都未测)",
             fontsize=11.4, y=0.99)
fig.savefig(RF + "fg_nft_better.png", bbox_inches="tight")
print("wrote", RF + "fg_nft_better.png")
