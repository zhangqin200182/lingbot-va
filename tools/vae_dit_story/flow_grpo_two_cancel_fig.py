r"""两个完全不同的"抵消",别再混:
  层次 A(采样器/密度): 噪声的扩散  ⇄  漂移的补偿   → 【精确抵消】⇒ 分布不变
  层次 B(奖励/优势):   运气 η_i                      → 【不抵消】;只有"减组内平均"这一步
                                                    把 η 拆成 批平均(去掉)+ 个体偏离(留下)

(a) 两层对照
(b) 噪声的三个去处(采样 / 梯度 / 奖励)与各自的处理
(c) 数值:个体运气不被消掉(σ²(1−1/G) 几乎就是 σ²);但估计方差靠平均按 1/G 下降
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

# ---------------- (a) 两层对照
ax = fig.add_axes([0.04, 0.10, 0.40, 0.78]); ax.axis("off")
ax.text(0.5, 1.00, "(a) 两个\"抵消\"完全不同", ha="center", va="top", fontsize=12, color=DARK)
ax.add_patch(plt.Rectangle((0.0, 0.60), 1.0, 0.30, fc="#eef7f1", ec=GREEN, lw=1.6))
ax.text(0.02, 0.875, "层次 A:采样器 / 密度", fontsize=10.6, color=GREEN, va="top", fontweight="bold")
ax.text(0.04, 0.815,
        "噪声的扩散项  $+(g^2/2)\\,p''$\n"
        "漂移的补偿项  $-(g^2/2)\\,p''$\n"
        "⇒ 【精确抵消】⇒ 分布不变(这是前面推的那件事)",
        fontsize=9.4, color=DARK, va="top")
ax.text(0.04, 0.645, "≠", fontsize=20, color=RED, va="center")
ax.add_patch(plt.Rectangle((0.0, 0.16), 1.0, 0.36, fc="#fdecea", ec=RED, lw=1.6))
ax.text(0.02, 0.50, "层次 B:奖励 / 优势", fontsize=10.6, color=RED, va="top", fontweight="bold")
ax.text(0.04, 0.44,
        "$R_i=\\dots+\\eta_i$  ← 运气(噪声造成的奖励波动)\n"
        "$A_i\\propto R_i-\\bar R=\\underbrace{(\\Delta_i-\\overline{\\Delta})}_{\\rm 保留}+"
        "\\underbrace{(\\eta_i-\\overline{\\eta})}_{\\rm 保留(这就是方差)}$\n"
        "⇒ 【不抵消】!只有 $\\overline{\\eta}$ 这个【批平均】被减掉,\n"
        "$\\eta_i-\\overline{\\eta}$ 原样留在优势里",
        fontsize=9.2, color=DARK, va="top")
ax.text(0.04, 0.185, "⇒ 运气不会被\"抵消\",只会被\"分解\":批平均去掉,个体偏离留下。",
        fontsize=9.4, color=RED, va="center")
# 手工去掉不支持的 underbrace
for t in ax.texts:
    if "underbrace" in t.get_text():
        t.set_text("$A_i\\propto R_i-\\bar R=(\\Delta_i-\\overline{\\Delta})+(\\eta_i-\\overline{\\eta})$\n"
                   "　　　　　　　保留(信号)　　　保留(方差)\n"
                   "⇒ 【不抵消】!只有 $\\overline{\\eta}$ 这个【批平均】被减掉,\n"
                   "$\\eta_i-\\overline{\\eta}$ 原样留在优势里")

# ---------------- (b) 噪声的三个去处
ax2 = fig.add_axes([0.46, 0.10, 0.31, 0.78]); ax2.axis("off")
ax2.text(0.5, 1.00, "(b) 噪声的三个去处", ha="center", va="top", fontsize=11.8, color=DARK)
rows = [["噪声出现在", "作用", "怎么处理"],
        ["① 采样器里\n$mean+s\\xi$", "让输出随机\n(探索)", "不动它\n这是引擎"],
        ["② 梯度里\n$(\\xi/s)\\partial m/\\partial\\theta$", "决定这次\n往哪个方向推", "直接用\n$\\mathbb{E}[\\xi]=0$"],
        ["③ 奖励里\n(运气 $\\eta$)", "让优势估计\n带噪声", "减组内平均\n(只去共同平移)"]]
tb = ax2.table(cellText=rows, cellLoc="center", bbox=[0.0, 0.50, 1.0, 0.44])
tb.auto_set_font_size(False); tb.set_fontsize(8.4)
for (r, c), cell in tb.get_celld().items():
    if r == 0:
        cell.set_facecolor("#eef2f7")
    if r == 3:
        cell.set_facecolor("#fdecea")
    if r == 2:
        cell.set_facecolor("#eef7f1")
ax2.text(0.0, 0.44,
         "要点:\n"
         "· \"抵消\"只发生在【层次 A 的密度】上\n"
         "· 层次 B 里没有任何东西去抵消噪声;\n"
         "　我们只是(a)把批平均减掉,(b)接受剩余方差\n"
         "· 剩余方差靠【大量迭代平均】变小,不是被消掉",
         fontsize=9.0, color=DARK, va="top",
         bbox=dict(fc="#fffbe6", ec="#e0c060"))

# ---------------- (c) 数值
ax3 = fig.add_axes([0.79, 0.17, 0.20, 0.66])
Gs = np.array([2, 4, 8, 16, 32, 64, 128])
resid = np.sqrt(1 - 1 / Gs)          # 个体运气残留(相对 σ)
est = 1 / np.sqrt(Gs)                # 估计方差(相对)
ax3.plot(Gs, resid, "o-", color=RED, lw=2.0, label="个体运气残留 $\\sqrt{1-1/G}$")
ax3.plot(Gs, est, "s-", color=BLUE, lw=2.0, label="估计标准差 $1/\\sqrt{G}$")
ax3.axhline(1.0, color=GREY, ls="--", lw=1.2)
ax3.set_xscale("log", base=2)
ax3.set_xlabel("组大小 $G$", fontsize=10.5)
ax3.set_ylabel("相对 $\\sigma$", fontsize=10.5)
ax3.set_ylim(0, 1.15)
ax3.set_title("(c) $G$ 变大:个体运气几乎不变,\n但平均后估计误差按 $1/\\sqrt{G}$ 下降", fontsize=10.8)
ax3.legend(fontsize=8.2, frameon=False, loc="center right")
ax3.tick_params(labelsize=8.6)

fig.suptitle("\"运气\"没有被抵消:采样器里那对(扩散⇄补偿)才是精确抵消;奖励层的运气只是被【分解】和【平均】",
             fontsize=12.0, y=0.99)
fig.savefig(RF + "fg_two_cancellations.png", bbox_inches="tight")
print("层次A: +(g²/2)p'' 与 −(g²/2)p'' 精确相消 → 分布不变(已在前面的图验证)")
print("层次B: η 被拆成 mean(η)(去掉) 与 η_i−mean(η)(留下)")
for G in [4, 16, 64]:
    print(f"  G={G:<3}: 个体运气残留 √(1−1/G) = {np.sqrt(1-1/G):.3f}σ; 估计标准差 1/√G = {1/np.sqrt(G):.3f}")
print("wrote", RF + "fg_two_cancellations.png")
