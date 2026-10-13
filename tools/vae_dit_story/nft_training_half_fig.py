r"""DiffusionNFT 的"然后呢":奖励 → 权重 → 前向回归 → 软更新参照。

完整流水线(Algorithm 1 + Appendix C):
  ① Rollout     用【标准 solver】(2 阶 ODE,10 步)采 K=24 张;每个样本不同 ε;打分
  ② 优势→权重   r_norm = r_raw − mean(r_raw);a = clip{r_norm/Z_c, ±1};r = 0.5+0.5a
  ③ 存缓冲      D ← {c, x₀, r}        ← 【只要干净样本】,不要轨迹/不要 log-prob
  ④ 前向加噪    x_t = α_t x₀ + σ_t ε ;  v = α̇_t x₀ + σ̇_t ε   (用采样时的 t 网格)
  ⑤ 隐式双分支  v⁺_θ = (1−β)v_old + βv_θ ;  v⁻_θ = (1+β)v_old − βv_θ
  ⑥ 加权回归    L = r‖v⁺_θ − v‖² + (1−r)‖v⁻_θ − v‖²  (时间权重 w(t),大 t 更高)
  ⑦ 软更新      v_old ← η v_old + (1−η) v_θ ,  η_i = min(0.001i, 0.5)

关键:最小化 ⑥ 得到的目标
      v_θ 的目标 = v_old + (2r−1)·(v − v_old)/β
                    └ 锚点 ┘   └ 符号 ┘   └ 幅度 ┘
  · 符号(模仿/反学)由【奖励】决定
  · 幅度由"样本离锚点多远"与 1/β 决定
  · 锚点由【软更新 η】决定:新鲜(η小)= 激进;陈旧(η大)= 保守
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
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

RED, BLUE, GREEN, DARK, GREY, ORANGE, PURPLE = ("#c0392b", "#2471a3", "#1e8449",
                                                "#2c3e50", "#95a5a6", "#b9770e", "#7d3c98")
RF = "/Users/kevin/code/lingbot-va/.claude/scratch/rf/"


def box(ax, x, y, w, h, text, fc="w", ec="#c8d4e0", fs=8.4, tc=DARK, lw=1.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.010", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc)


fig = plt.figure(figsize=(15.2, 6.0), dpi=150)

# ---------------- (a) 完整流水线
ax = fig.add_axes([0.02, 0.05, 0.50, 0.85]); ax.axis("off")
ax.text(0.5, 1.00, "(a) \"然后呢\":训练那一半(7 步)", ha="center", va="top", fontsize=12, color=DARK)
items = [
    ("① Rollout:标准 solver 采 $K=24$\n(2 阶 ODE,10 步;各样本不同 $\\epsilon$)", "#eef2f7", GREY),
    ("② 打分 → 组内减均值 → 截断 $[\\pm1]$ → $r=0.5+0.5a\\in[0,1]$", "#eaf2fa", BLUE),
    ("③ 存 $\\{c,\\,x_0,\\,r\\}$  ← 【只要干净样本】\n(不要轨迹,不要 $\\log\\pi$)", "#eaf2fa", BLUE),
    ("④ 前向加噪:$x_t=\\alpha_t x_0+\\sigma_t\\epsilon$,$v=\\dot\\alpha_t x_0+\\dot\\sigma_t\\epsilon$", "#eef7f1", GREEN),
    ("⑤ 隐式双分支:$v^+_\\theta=(1-\\beta)v_{\\rm old}+\\beta v_\\theta$\n"
     "$v^-_\\theta=(1+\\beta)v_{\\rm old}-\\beta v_\\theta$", "#eef7f1", GREEN),
    ("⑥ 加权回归:$\\mathcal{L}=r\\|v^+_\\theta-v\\|^2+(1-r)\\|v^-_\\theta-v\\|^2$\n(时间权重 $w(t)$:大 $t$ 更高)", "#eef7f1", GREEN),
    ("⑦ 软更新参照:$v_{\\rm old}\\leftarrow\\eta v_{\\rm old}+(1-\\eta)v_\\theta$,$\\eta_i=\\min(0.001i,0.5)$",
     "#fdf3e7", ORANGE),
]
y = 0.930
for txt, fc, ec in items:
    box(ax, 0.02, y - 0.098, 0.96, 0.098, txt, fc, ec, fs=8.5)
    if y > 0.21:
        ax.add_patch(FancyArrowPatch((0.5, y - 0.102), (0.5, y - 0.126), arrowstyle="-|>",
                                     mutation_scale=11, lw=1.5, color=DARK))
    y -= 0.126
ax.text(0.5, 0.035, "没有的东西:log-prob、轨迹、ratio、clip、KL、CFG、value 网络",
        fontsize=9.0, color=RED, ha="center", bbox=dict(fc="#fdecea", ec="#e6a9a0"))
ax.set_xlim(0, 1); ax.set_ylim(0.02, 1.02)

# ---------------- (b) 目标公式的三个因子
ax2 = fig.add_axes([0.55, 0.10, 0.22, 0.78]); ax2.axis("off")
ax2.text(0.5, 1.00, "(b) 目标公式的三个因子", ha="center", va="top", fontsize=11.4, color=DARK)
ax2.text(0.0, 0.955,
         "最小化 ⑥ 后:\n"
         "$v_\\theta^{\\rm target}=v_{\\rm old}+(2r-1)\\dfrac{v-v_{\\rm old}}{\\beta}$",
         fontsize=9.4, color=DARK, va="top")
rows = [["因子", "由谁决定", "作用"],
        ["$v_{\\rm old}$", "软更新 $\\eta$", "锚点\n(新鲜=激进\n陈旧=保守)"],
        ["$2r-1$", "奖励", "符号\n(模仿 / 反学)"],
        ["$(v-v_{\\rm old})$", "样本与锚点\n的距离", "幅度来源"],
        ["$1/\\beta$", "超参", "力度\n(β=1 稳,\n0.1 更猛)"]]
tb = ax2.table(cellText=rows, cellLoc="center", bbox=[0.0, 0.34, 1.0, 0.48])
tb.auto_set_font_size(False); tb.set_fontsize(7.6)
for (r, c), cell in tb.get_celld().items():
    if r == 0:
        cell.set_facecolor("#eef2f7")
    if c == 1 and r > 0:
        cell.set_facecolor("#fdf3e7")
ax2.text(0.0, 0.28,
         "消融(论文):\n"
         "· 完全 on-policy($\\eta=0$)不稳\n"
         "· $\\eta=0.9$ 太陈旧,收敛慢\n"
         "· 小→大的调度最好\n"
         "· 去掉负分支 → 奖励崩溃",
         fontsize=8.2, color=DARK, va="top",
         bbox=dict(fc="#fffbe6", ec="#e0c060"))

# ---------------- (c) 几何
ax3 = fig.add_axes([0.80, 0.17, 0.185, 0.66])
vold_, vs = 0.0, 1.0
for a, c, lab in [(1.0, BLUE, "$r=1$:模仿"), (0.5, GREEN, "$r=0.75$"), (0.0, DARK, "$r=0.5$:不动"),
                  (-1.0, ORANGE, "$r=0$:反学")]:
    tgt = vold_ + a * (vs - vold_) / 1.0
    ax3.plot([tgt], [a], "o", ms=11, color=c)
    ax3.annotate("", xy=(tgt, a), xytext=(vold_, a),
                 arrowprops=dict(arrowstyle="-|>", lw=2.0, color=c))
    ax3.text(tgt + 0.06, a, lab, fontsize=8.2, color=c, va="center")
ax3.axvline(vold_, color=ORANGE, ls="--", lw=1.8)
ax3.text(vold_, 1.28, "锚点 $v_{\\rm old}$", fontsize=8.6, color=ORANGE, ha="center")
ax3.axvline(vs, color=BLUE, ls=":", lw=1.4)
ax3.text(vs, 1.28, "样本 $v$", fontsize=8.6, color=BLUE, ha="center")
ax3.set_xlim(-1.4, 1.9); ax3.set_ylim(-1.5, 1.5)
ax3.set_yticks([-1, 0, 0.5, 1]); ax3.set_yticklabels(["$r=0$", "$0.5$", "$0.75$", "$r=1$"], fontsize=8.4)
ax3.set_xlabel("$v_\\theta$ 的目标", fontsize=9.4)
ax3.set_title("(c) 目标随奖励\n在锚点两侧移动", fontsize=10.6)
ax3.tick_params(labelsize=8.2)

fig.suptitle("DiffusionNFT 的训练:奖励只当\"权重+方向\",参照策略当\"锚点\" —— 前向回归,不碰采样过程",
             fontsize=11.8, y=0.99)
fig.savefig(RF + "fg_nft_training_half.png", bbox_inches="tight")
print("wrote", RF + "fg_nft_training_half.png")
print()
print("目标公式核对(β=1, v_old=0, v=1):")
for r in [1.0, 0.75, 0.5, 0.25, 0.0]:
    print(f"  r={r:4.2f} → 目标 = {0 + (2*r-1)*(1-0)/1.0:+.2f}")
