r"""修正措辞:baseline 的本质是【平移不变性】,不是"消除了某个共同的随机源"。

  A_i ∝ R_i − R̄        ⇒ 给整组加任何常数 c,A_i 完全不变
  "对这一批所有样本都一样的量"包括:
    (a) prompt 难度                  —— 真常数
    (b) 奖励模型对该 prompt 的系统偏差 —— 真常数
    (c) G 个样本共享的条件/设置       —— 确定量
    (d) 这批 G 个独立运气的【平均】 mean(η) —— 随机量,但对这一批是同一个数

(a) 平移不变性:R 整体加 10,优势一模一样
(b) mean(η) 是随机量:重抽 1000 次,std = σ/√G,期望 0
(c) 清单 + 留一均值为什么更严格
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

D = np.array([1.0, -1.0, 0.5, -0.5])
ETA = np.array([0.2, 0.3, -0.1, 0.1])
R = 5.0 + D + ETA
A = (R - R.mean()) / (R.std() + 1e-4)
Rc = R + 10.0                       # 整组加常数
Ac = (Rc - Rc.mean()) / (Rc.std() + 1e-4)

fig = plt.figure(figsize=(15.2, 5.5), dpi=150)

# ---------------- (a) 平移不变性
ax = fig.add_axes([0.055, 0.17, 0.30, 0.66])
x = np.arange(4)
ax.bar(x - 0.19, A, width=0.36, color=BLUE, label="原始奖励")
ax.bar(x + 0.19, Ac, width=0.36, color=ORANGE, label="整组奖励 $+10$")
ax.axhline(0, color=GREY, lw=0.9)
for i in x:
    ax.text(i, A[i] + (0.06 if A[i] > 0 else -0.20), f"{A[i]:+.2f}", ha="center", fontsize=8.8, color=BLUE)
ax.set_xticks(x); ax.set_xticklabels([f"样本{i+1}" for i in x], fontsize=9.4)
ax.set_ylim(-2.1, 2.2)
ax.set_ylabel("优势 $A_i$", fontsize=10.5)
ax.set_title("(a) 基线的本质是【平移不变性】\n整组加任何常数 $c$,优势完全不变", fontsize=11.4)
ax.legend(fontsize=8.8, frameon=False, loc="upper left")
ax.tick_params(labelsize=9)
ax.text(1.5, -1.95, "$A_i\\propto R_i-\\bar R$:减去一个\"对本批所有样本都一样的数\"",
        fontsize=8.8, color=GREEN, ha="center", bbox=dict(fc="#eef7f1", ec="#a9d5bb"))

# ---------------- (b) mean(eta) 是随机量
ax2 = fig.add_axes([0.425, 0.17, 0.30, 0.66])
rng = np.random.default_rng(0)
sig, G = 0.5, 4
means = rng.normal(0, sig / np.sqrt(G), size=20000)
ax2.hist(means, bins=80, density=True, color=PURPLE, alpha=0.55)
xs = np.linspace(-1, 1, 400)
ax2.plot(xs, np.exp(-xs ** 2 / (2 * (sig / np.sqrt(G)) ** 2)) / (sig / np.sqrt(G) * np.sqrt(2 * np.pi)),
         color=DARK, lw=2.2)
ax2.axvline(0, color=GREY, ls="--", lw=1.4)
ax2.axvline(ETA.mean(), color=RED, lw=2.4)
ax2.text(ETA.mean() + 0.03, 1.45, f"这一次抽到的\n$\\overline{{\\eta}}={ETA.mean():+.3f}$", fontsize=9.4, color=RED)
ax2.text(-0.95, 1.35, "重抽 20000 次的分布:\n期望 $=0$,标准差 $=\\sigma/\\sqrt{G}$\n"
                      f"($\\sigma={sig},G={G}$ → {sig/np.sqrt(G):.3f})",
         fontsize=9.0, color=DARK, va="top")
ax2.set_xlim(-1.1, 1.1); ax2.set_ylim(0, 2.1)
ax2.set_xlabel("$\\overline{\\eta}=$ 这一批运气的平均", fontsize=10.5)
ax2.set_title("(b) 但它本身是【随机量】:\n不是某个共同的随机源,而是独立运气的样本均值", fontsize=11.4)
ax2.tick_params(labelsize=9)

# ---------------- (c) 清单
ax3 = fig.add_axes([0.755, 0.17, 0.24, 0.66]); ax3.axis("off")
ax3.text(0.5, 0.99, "(c) \"对本批都一样\"的量", ha="center", va="top", fontsize=11.4, color=DARK)
rows = [["什么", "性质"],
        ["prompt 难度", "真常数"],
        ["奖励模型的\n系统性偏差", "真常数"],
        ["共享的设置\n(条件/σ网格/θ)", "确定量"],
        ["$\\overline{\\eta}$ 这批运气平均", "随机量\n(期望 0)"]]
tb = ax3.table(cellText=rows, cellLoc="center", bbox=[0.0, 0.56, 1.0, 0.40])
tb.auto_set_font_size(False); tb.set_fontsize(8.2)
for (r, c), cell in tb.get_celld().items():
    if r == 0:
        cell.set_facecolor("#eef2f7")
    if r == 4:
        cell.set_facecolor("#f4ecf7")
ax3.text(0.0, 0.50,
         "减法把【前三类真常数】干净去掉;\n"
         "对 $\\overline{\\eta}$ 是\"顺手\"去掉,\n"
         "但代价是 $O(1/G)$ 的小偏差 ——\n"
         "因为 $\\overline{\\eta}$ 里含样本 $i$ 自己的运气。\n\n"
         "严格无偏版:留一均值 $\\bar R_{-i}$\n"
         "(用其他 $G-1$ 个算基线)。",
         fontsize=8.6, color=DARK, va="top",
         bbox=dict(fc="#fffbe6", ec="#e0c060"))
ax3.text(0.0, 0.03,
         "一句话:基线去掉的是\n【与本批样本无关的平移】,\n不是某个随机源。",
         fontsize=8.8, color=GREEN, va="bottom", bbox=dict(fc="#eef7f1", ec="#a9d5bb"))

fig.suptitle("纠正措辞:组内基线 = 平移不变性;\"共同运气\"其实是【这批独立运气的平均值】(随机量,但本批共享)", fontsize=12.2, y=0.99)
fig.savefig(RF + "fg_shift_invariance.png", bbox_inches="tight")
print(f"原始 R = {np.round(R,3).tolist()}, A = {np.round(A,3).tolist()}")
print(f"整组 +10 后 A = {np.round(Ac,3).tolist()}  → 完全相同: {np.allclose(A, Ac)}")
print(f"这一次的 mean(η) = {ETA.mean():+.4f};重抽 20000 次的标准差 = {means.std():.4f} (理论 σ/√G = {sig/np.sqrt(G):.4f})")
print("wrote", RF + "fg_shift_invariance.png")
