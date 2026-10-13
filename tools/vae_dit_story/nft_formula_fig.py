r"""把 NFT 的损失公式逐符号拆开。

符号表:
  v_θ    正在训练的网络(要更新的那个),输出"速度"
  v_old  参照速度(锚点):上一轮 v_θ 的软平均,【不参与求导】
  v      这张图的"正确答案"(考题):v = ε − x₀(x₀ 采到的,ε 新抽的)
  r      这张图有多好:∈[0,1],0.5 = 组内平均,1 = 组内最好,0 = 组内最差
  β      引导强度(超参):越大越保守,越小越激进

两个隐式策略(同一个网络的"分身",以 v_old 为支点):
  好策略 v⁺_θ = v_old + β(v_θ − v_old)
  坏策略 v⁻_θ = v_old − β(v_θ − v_old)
  (展开就是 (1−β)v_old + βv_θ 与 (1+β)v_old − βv_θ)

损失:
  L = r‖v⁺_θ − v‖² + (1−r)‖v⁻_θ − v‖²

  · r 大 → 第一项权重大 → 要求【好策略】去匹配 v
  · r 小 → 第二项权重大 → 要求【坏策略】去匹配 v
  (注意:两项都在要求匹配同一个 v,但匹配的是不同的函数!)

解 dL/dv_θ = 0:
  v_θ 的目标 = v_old + (2r−1)·(v − v_old)/β
                └ 支点 ┘   └ 优势 a ┘  └ 方向与幅度 ┘
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
from matplotlib.patches import FancyArrowPatch

RED, BLUE, GREEN, DARK, GREY, ORANGE, PURPLE = ("#c0392b", "#2471a3", "#1e8449",
                                                "#2c3e50", "#95a5a6", "#b9770e", "#7d3c98")
RF = "/Users/kevin/code/lingbot-va/.claude/scratch/rf/"

fig = plt.figure(figsize=(15.2, 5.9), dpi=150)

# ---------------- (a) 符号表
ax = fig.add_axes([0.03, 0.08, 0.31, 0.82]); ax.axis("off")
ax.text(0.5, 1.00, "(a) 每个符号是什么", ha="center", va="top", fontsize=11.8, color=DARK)
rows = [["符号", "是什么", "范围/来源"],
        ["$v_\\theta$", "正在训练的网络\n(要更新的)", "参数 $\\theta$"],
        ["$v_{\\rm old}$", "参照速度【锚点】\n不参与求导", "上轮 $v_\\theta$ 的\n软平均(EMA)"],
        ["$v$", "这张图的【正确答案】\n(考题答案)", "$v=\\epsilon-x_0$\n$x_0$ 采到,$\\epsilon$ 新抽"],
        ["$r$", "这张图【有多好】", "$\\in[0,1]$\n$0.5$=组内平均"],
        ["$\\beta$", "引导强度\n(超参)", "$1$(单奖励)\n$0.1$(多奖励)"]]
tb = ax.table(cellText=rows, cellLoc="center", bbox=[0.0, 0.30, 1.0, 0.64])
tb.auto_set_font_size(False); tb.set_fontsize(8.2)
for (r_, c_), cell in tb.get_celld().items():
    if r_ == 0:
        cell.set_facecolor("#eef2f7")
    if c_ == 0 and r_ > 0:
        cell.set_facecolor("#eaf2fa")
ax.text(0.0, 0.24,
         "$r$ 怎么来(组内):\n"
         "$r_{\\rm raw}\\to r_{\\rm raw}-\\mathrm{mean}\\to \\mathrm{clip}[\\pm1]\\to a$\n"
         "$\\to r=0.5+0.5a$\n\n"
         "所以 $a=2r-1$ 就是【组内相对优势】\n"
         "$r>0.5$ 拉近;$r=0.5$ 不动;$r<0.5$ 推远",
         fontsize=8.6, color=DARK, va="top", bbox=dict(fc="#fffbe6", ec="#e0c060"))

# ---------------- (b) 两个隐式策略的几何
ax2 = fig.add_axes([0.37, 0.15, 0.30, 0.70])
vold_, vth = 0.0, 0.6
for beta, dy, col in [(1.0, 0.0, GREEN), (0.5, -0.55, PURPLE)]:
    vp = vold_ + beta * (vth - vold_)
    vm = vold_ - beta * (vth - vold_)
    ax2.plot([vm, vp], [dy, dy], color=col, lw=2.0, alpha=0.4)
    ax2.plot([vp], [dy], "o", ms=11, color=col)
    ax2.plot([vm], [dy], "s", ms=11, color=col)
    ax2.text(vp + 0.05, dy, f"$v^+_\\theta$($\\beta={beta}$)", fontsize=8.8, color=col, va="center")
    ax2.text(vm - 0.05, dy, f"$v^-_\\theta$", fontsize=8.8, color=col, va="center", ha="right")
    ax2.annotate("", xy=(vold_, dy - 0.18), xytext=(vp, dy - 0.18),
                 arrowprops=dict(arrowstyle="<|-|>", lw=1.4, color=col))
    ax2.text((vold_ + vp) / 2, dy - 0.30, f"$\\beta|v_\\theta-v_{{\\rm old}}|$", fontsize=8.0,
             color=col, ha="center")
ax2.axvline(vold_, color=ORANGE, ls="--", lw=2.2)
ax2.text(vold_, 0.62, "支点 $v_{\\rm old}$", fontsize=9.6, color=ORANGE, ha="center")
ax2.plot([vth], [0.30], "o", ms=12, color=BLUE)
ax2.text(vth, 0.38, "$v_\\theta$(当前网络)", fontsize=9.4, color=BLUE, ha="center")
ax2.add_patch(FancyArrowPatch((vold_, 0.30), (vth, 0.30), arrowstyle="-|>", mutation_scale=12,
                              lw=2.2, color=BLUE))
ax2.text(0.5 * (vold_ + vth), 0.24, "$v_\\theta-v_{\\rm old}$(偏离)", fontsize=8.6, color=BLUE,
         ha="center", va="top")
ax2.set_xlim(-0.95, 1.15); ax2.set_ylim(-0.85, 0.80)
ax2.set_yticks([])
ax2.set_xlabel("速度空间", fontsize=10.2)
ax2.set_title("(b) 同一个网络的两个\"分身\"\n以 $v_{\\rm old}$ 为支点,左右各 $\\beta$ 倍", fontsize=11.0)
ax2.tick_params(labelsize=8.8)

# ---------------- (c) 损失的组合与解
ax3 = fig.add_axes([0.70, 0.08, 0.29, 0.82]); ax3.axis("off")
ax3.text(0.5, 1.00, "(c) 两项怎么合成一个目标", ha="center", va="top", fontsize=11.6, color=DARK)
ax3.text(0.0, 0.92,
         "$\\mathcal{L}=r\\|v^+_\\theta-v\\|^2+(1-r)\\|v^-_\\theta-v\\|^2$\n\n"
         "两项【都在匹配同一个 $v$】,\n"
         "但各自匹配的是不同的函数:\n"
         "　$r$ 大 ⇒ 逼着【好策略 $v^+$】去等于 $v$\n"
         "　$r$ 小 ⇒ 逼着【坏策略 $v^-$】去等于 $v$",
         fontsize=9.0, color=DARK, va="top")
ax3.text(0.0, 0.60,
         "分开看($\\beta=1$):\n"
         "　只留第一项:$v^+_\\theta=v$ ⇒ $v_\\theta=v$ 　(模仿)\n"
         "　只留第二项:$v^-_\\theta=v$ ⇒ $v_\\theta=2v_{\\rm old}-v$ (反射)",
         fontsize=8.8, color=GREEN, va="top", bbox=dict(fc="#eef7f1", ec=GREEN))
ax3.text(0.0, 0.40,
         "两项一起($d\\mathcal{L}/dv_\\theta=0$):\n"
         "$v_\\theta^{\\rm target}=v_{\\rm old}+(2r-1)\\dfrac{v-v_{\\rm old}}{\\beta}$",
         fontsize=9.0, color=DARK, va="top")
ax3.text(0.0, 0.22,
         "读法:从支点 $v_{\\rm old}$ 出发,\n"
         "朝样本 $v$ 的方向走 $\\dfrac{2r-1}{\\beta}$ 倍\n\n"
         "$\\beta$ 的作用(同一个 $r=1$):\n"
         "　$\\beta=1$ 　目标 $=v$　　　 (到位)\n"
         "$\\beta=0.5$ 目标 $=2v-v_{\\rm old}$ (更远)\n"
         "　$\\beta=2$ 　目标 $=v_{\\rm old}+\\frac{v-v_{\\rm old}}{2}$(半路)",
         fontsize=8.6, color=DARK, va="top", bbox=dict(fc="#fdf3e7", ec=ORANGE))

fig.suptitle("公式逐符号拆解:$v_\\theta$ 是要训的网络,$v_{\\rm old}$ 是锚点,$v$ 是考题答案,$r$ 是这张图多好,$\\beta$ 是力度",
             fontsize=11.4, y=0.99)
fig.savefig(RF + "fg_nft_formula_explained.png", bbox_inches="tight")

# 数值核对
print("解公式核对(β=1, v_old=0, v=1):")
for rr in [1.0, 0.5, 0.0]:
    print(f"  r={rr:.1f} → 目标 = {0 + (2*rr-1)*(1-0)/1.0:+.2f}")
print("β 的作用(同一个 r=1, v_old=0, v=1):")
for bb in [2.0, 1.0, 0.5]:
    print(f"  β={bb:.1f} → 目标 = {0 + 1*(1-0)/bb:+.2f}")
print("wrote", RF + "fg_nft_formula_explained.png")
