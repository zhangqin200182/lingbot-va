r"""奖励"路径"还是奖励"通向目的的速度"?

论文自己的分类(Alignment for T2I,五条主线):
  (1) direct fine-tuning with scalar rewards [30,31,32,33]   ← 你说的"直接奖励速度"属于这里
  (2) Reward Weighted Regression (RWR)
  (3) DPO 及其变体
  (4) PPO-style policy gradients                            ← Flow-GRPO 在这里
  (5) training-free alignment

路线 A(可微奖励反传,DRaFT/ReFL/AlignProp):不做路径抽样,把终点奖励直接对场求导
路线 B(GRPO):终点打分 → 优势 → 每步似然比 → 更新(需要 ODE→SDE 才有似然)

(a) 两条路线的结构对比
(b) 取舍表
(c) 三个折中方案
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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

RED, BLUE, GREEN, DARK, GREY, ORANGE, PURPLE = ("#c0392b", "#2471a3", "#1e8449",
                                                "#2c3e50", "#95a5a6", "#b9770e", "#7d3c98")
RF = "/Users/kevin/code/lingbot-va/.claude/scratch/rf/"


def box(ax, x, y, w, h, text, fc, ec, fs=8.8, tc=DARK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc)


fig = plt.figure(figsize=(15.2, 5.8), dpi=150)

# ---------------- (a) 两条路线
ax = fig.add_axes([0.03, 0.06, 0.44, 0.84]); ax.axis("off")
ax.text(0.5, 1.00, "(a) 两条路线的结构", ha="center", va="top", fontsize=12, color=DARK)

# 路线 A
ax.text(0.0, 0.90, "路线 A:可微奖励反传(DRaFT / ReFL / AlignProp)", fontsize=9.8, color=BLUE)
box(ax, 0.00, 0.72, 0.17, 0.11, "初始噪声 $\\epsilon$", "#eef2f7", "#c8d4e0")
box(ax, 0.21, 0.72, 0.19, 0.11, "采样链 $F_\\theta$\n(确定性 ODE)", "#eaf2fa", BLUE)
box(ax, 0.44, 0.72, 0.16, 0.11, "输出 $x(T)$", "#eef2f7", "#c8d4e0")
box(ax, 0.64, 0.72, 0.16, 0.11, "奖励 $R$", "#eaf2fa", BLUE)
box(ax, 0.84, 0.72, 0.16, 0.11, "$\\partial R/\\partial\\theta$", "#eef7f1", GREEN)
for x0 in [0.17, 0.40, 0.60, 0.80]:
    ax.add_patch(FancyArrowPatch((x0, 0.775), (x0 + 0.035, 0.775), arrowstyle="-|>",
                                 mutation_scale=11, lw=1.8, color=DARK))
ax.add_patch(FancyArrowPatch((0.92, 0.71), (0.30, 0.71), arrowstyle="-|>", mutation_scale=13,
                             lw=2.2, color=GREEN, ls="--"))
ax.text(0.60, 0.665, "整条链留在计算图里,误差直接反传", fontsize=8.8, color=GREEN, ha="center")
ax.text(0.02, 0.60, "要点:① 奖励必须【可微】 ② 不需要似然(不必 ODE→SDE) ③ 梯度无噪声 ④ 显存贵",
        fontsize=8.8, color=BLUE)

# 路线 B
ax.text(0.0, 0.50, "路线 B:GRPO(本方法)", fontsize=9.8, color=RED)
box(ax, 0.00, 0.32, 0.15, 0.11, "SDE 采 $G$ 个\n(记每步 $\\log\\pi$)", "#fdecea", RED)
box(ax, 0.19, 0.32, 0.15, 0.11, "打分 $R_i$", "#fdecea", RED)
box(ax, 0.38, 0.32, 0.19, 0.11, "组内标准化\n$A_i$", "#fdecea", RED)
box(ax, 0.61, 0.32, 0.18, 0.11, "每步 $\\rho_j=\\pi_\\theta/\\pi_{\\rm old}$", "#fdecea", RED)
box(ax, 0.83, 0.32, 0.17, 0.11, "更新 $\\theta$", "#fdecea", RED)
for x0 in [0.15, 0.34, 0.57, 0.79]:
    ax.add_patch(FancyArrowPatch((x0, 0.375), (x0 + 0.035, 0.375), arrowstyle="-|>",
                                 mutation_scale=11, lw=1.8, color=DARK))
ax.text(0.02, 0.20, "要点:① 奖励只要【标量】(黑盒打分器都行) ② 需要每步似然 ③ 梯度有噪声(无偏) ④ 便宜",
        fontsize=8.8, color=RED)
ax.text(0.02, 0.09, "为什么不可微时只能走 B:一个标量\"好/坏\"【不提供方向】——\n"
                    "只能靠\"这次随机方向 $\\xi$ 是否与好结果相关\"统计地估出方向:"
                    "$\\mathbb{E}[A\\xi]=s\\,\\partial\\mathbb{E}[R]/\\partial m$",
        fontsize=8.8, color=DARK, va="top", bbox=dict(fc="#fffbe6", ec="#e0c060"))

# ---------------- (b) 取舍表
ax2 = fig.add_axes([0.49, 0.13, 0.29, 0.77]); ax2.axis("off")
ax2.text(0.5, 1.00, "(b) 取舍", ha="center", va="top", fontsize=11.8, color=DARK)
rows = [["", "A:反传奖励", "B:GRPO"],
        ["奖励要求", "必须可微", "只要标量"],
        ["需要似然", "不需要", "必须要"],
        ["显存", "整条链\n(贵)", "只看 log-prob\n(便宜)"],
        ["梯度", "无噪声\n(精确)", "有噪声\n(无偏)"],
        ["主要风险", "奖励被钻空子\n(对抗性)", "方差大\n无逐步信用"],
        ["典型奖励", "CLIP 分数\n检测框回归", "OCR / 规则\n偏好模型"]]
tb = ax2.table(cellText=rows, cellLoc="center", bbox=[0.0, 0.30, 1.0, 0.62])
tb.auto_set_font_size(False); tb.set_fontsize(8.0)
for (r, c), cell in tb.get_celld().items():
    if r == 0:
        cell.set_facecolor("#eef2f7")
    if c == 1 and r > 0:
        cell.set_facecolor("#eaf2fa")
    if c == 2 and r > 0:
        cell.set_facecolor("#fdecea")
ax2.text(0.0, 0.24,
         "论文自己的分类(T2I 对齐五条主线):\n"
         "(1) 直接用标量奖励微调 $\\leftarrow$ 【你的想法在这】\n"
         "(2) RWR  (3) DPO 及变体\n"
         "(4) PPO 式策略梯度 $\\leftarrow$ 【Flow-GRPO 在这】\n"
         "(5) 免训练对齐",
         fontsize=8.6, color=DARK, va="top",
         bbox=dict(fc="#f7f9fb", ec="#c8d4e0"))

# ---------------- (c) 折中方案
ax3 = fig.add_axes([0.79, 0.13, 0.20, 0.77]); ax3.axis("off")
ax3.text(0.5, 1.00, "(c) 三个折中", ha="center", va="top", fontsize=11.6, color=DARK)
ax3.text(0.0, 0.90,
         "① 可微奖励当【辅助项】\n"
         "$\\mathcal{L}=\\mathcal{L}_{\\rm GRPO}+\\lambda(-R_{\\rm diff})$\n"
         "　 只对最后几步保留计算图\n"
         "　 ⇒ 省显存,又吃到精确梯度\n\n"
         "② 先 reflow/蒸馏【修直路径】\n"
         "　 步数变少 ⇒ 反传代价低,\n"
         "　 路径本身也更接近直线\n\n"
         "③ 引入 【per-step 优势】\n"
         "　 (代码里广播就是为它留的口子)\n"
         "　 ⇒ 真正做逐步信用分配",
         fontsize=8.6, color=DARK, va="top")
ax3.text(0.0, 0.03,
         "真正会咬人的两个地方:\n"
         "· 少步采样质量(论文未测)\n"
         "· reward hacking(KL 压制但没根除)\n"
         "—— 都不是\"学了歪路径\"。",
         fontsize=8.4, color=GREEN, va="bottom", bbox=dict(fc="#eef7f1", ec="#a9d5bb"))

fig.suptitle("\"直接奖励通向目的的速度\"= 可微奖励反传(论文分类的方向 1);GRPO 走路径似然,是因为不可微标量奖励不提供方向",
             fontsize=11.8, y=0.99)
fig.savefig(RF + "fg_reward_path_vs_field.png", bbox_inches="tight")
print("wrote", RF + "fg_reward_path_vs_field.png")
