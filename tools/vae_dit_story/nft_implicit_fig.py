r""""两个分身"到底在算什么?—— 改的是损失,不是网络的预测,也没有新网络。

  部署/采样时用的:只有 v_θ(它就是模型)
  损失里出现的:  v⁺_θ = v_old + β(v_θ − v_old)     ← 只是 v_θ 与常数 v_old 的仿射组合
                v⁻_θ = v_old − β(v_θ − v_old)

  把 L 改写(完全等价,已数值验证):
      L = β²·[ r‖v_θ − T⁺‖² + (1−r)‖v_θ − T⁻‖² ]
      T⁺ = v_old + (v−v_old)/β ,  T⁻ = v_old − (v−v_old)/β     ← 关于锚点【对称】
  ⇒ 两项不过是"对 v_θ 的两个对称软目标"
  ⇒ 梯度 = 朝加权目标走:∇_θL = 2β²(v_θ − T)·∂v_θ/∂θ
      T = rT⁺ + (1−r)T⁻ = v_old + (2r−1)(v−v_old)/β
  ⇒ 没有任何额外机制
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

# ---------------- (a) 三件事分清
ax = fig.add_axes([0.03, 0.08, 0.31, 0.82]); ax.axis("off")
ax.text(0.5, 1.00, "(a) 分清三件事", ha="center", va="top", fontsize=11.8, color=DARK)
ax.add_patch(plt.Rectangle((0.02, 0.72), 0.96, 0.20, fc="#eaf2fa", ec=BLUE, lw=1.8))
ax.text(0.05, 0.885, "① 模型本身 = $v_\\theta$", fontsize=10.0, color=BLUE, va="top")
ax.text(0.05, 0.825, "部署、采样时用的【只有它】\n$v^+_\\theta,v^-_\\theta$ 从来不参与生成",
        fontsize=8.8, color=DARK, va="top")
ax.add_patch(plt.Rectangle((0.02, 0.40), 0.96, 0.26, fc="#eef7f1", ec=GREEN, lw=1.8))
ax.text(0.05, 0.625, "② 损失里的两个\"分身\"", fontsize=10.0, color=GREEN, va="top")
ax.text(0.05, 0.565,
        "$v^\\pm_\\theta=v_{\\rm old}\\pm\\beta(v_\\theta-v_{\\rm old})$\n"
        "= $v_\\theta$ 与【常数】$v_{\\rm old}$ 的仿射组合\n"
        "⇒ 不是新网络,不是新参数,不增加前向",
        fontsize=8.8, color=DARK, va="top")
ax.add_patch(plt.Rectangle((0.02, 0.14), 0.96, 0.20, fc="#fdf3e7", ec=ORANGE, lw=1.8))
ax.text(0.05, 0.305, "③ 所以改的是【损失】", fontsize=10.0, color=ORANGE, va="top")
ax.text(0.05, 0.245, "不是\"改网络的预测\":网络的输出永远是 $v_\\theta$\n改的是\"要求它去拟合谁\"",
        fontsize=8.8, color=DARK, va="top")
ax.text(0.5, 0.07, "名字:implicit parameterization\n(一套参数,两个策略)", fontsize=9.2,
        color=GREEN, ha="center", va="center", bbox=dict(fc="#eef7f1", ec="#a9d5bb"))
ax.set_xlim(0, 1); ax.set_ylim(0.02, 1.02)

# ---------------- (b) 等价变换
ax2 = fig.add_axes([0.37, 0.08, 0.31, 0.82]); ax2.axis("off")
ax2.text(0.5, 1.00, "(b) 等价变换:两个对称目标", ha="center", va="top", fontsize=11.6, color=DARK)
ax2.text(0.0, 0.92,
         "原始:$\\mathcal{L}=r\\|v^+_\\theta-v\\|^2+(1-r)\\|v^-_\\theta-v\\|^2$\n\n"
         "注意 $v^+_\\theta-v=\\beta\\,(v_\\theta-T^+)$,其中\n"
         "$T^+=v_{\\rm old}+\\dfrac{v-v_{\\rm old}}{\\beta}$," 
         "$\\ T^-=v_{\\rm old}-\\dfrac{v-v_{\\rm old}}{\\beta}$\n\n"
         "⇒ $\\mathcal{L}=\\beta^2\\,[\\,r\\|v_\\theta-T^+\\|^2+(1-r)\\|v_\\theta-T^-\\|^2\\,]$\n\n"
         "⇒ 两项只是\"对 $v_\\theta$ 的两个【对称】软目标\"",
         fontsize=9.0, color=DARK, va="top")
ax2.text(0.0, 0.44,
         "梯度:$\\nabla_\\theta\\mathcal{L}=2\\beta^2(v_\\theta-T)\\cdot\\dfrac{\\partial v_\\theta}{\\partial\\theta}$\n"
         "$T=rT^++(1-r)T^-=v_{\\rm old}+(2r-1)\\dfrac{v-v_{\\rm old}}{\\beta}$\n\n"
         "⇒ 就是\"朝加权目标 $T$ 走\"(加权平方误差的一般事实:\n"
         "　 最优解 = 各目标的加权平均)",
         fontsize=9.0, color=GREEN, va="top", bbox=dict(fc="#eef7f1", ec=GREEN))
ax2.text(0.0, 0.12,
         "数值核对($\\beta=0.7,v_{\\rm old}=0,v=1,r=0.8$):\n"
         "$v_\\theta=0$:原损失 1.000000 = 改写后 1.000000\n"
         "　　　　　直接求导 $-0.84000$ = $2\\beta^2(v_\\theta-T)$ $-0.84000$ ✓",
         fontsize=8.6, color=DARK, va="top", bbox=dict(fc="#fffbe6", ec="#e0c060"))

# ---------------- (c) 几何 + β 的两个作用
ax3 = fig.add_axes([0.71, 0.13, 0.27, 0.74])
vold_, v_ = 0.0, 1.0
for beta, dy, col in [(1.0, 0.0, GREEN), (0.5, -0.35, PURPLE), (2.0, 0.35, BLUE)]:
    Tp = vold_ + (v_ - vold_) / beta
    Tm = vold_ - (v_ - vold_) / beta
    ax3.plot([Tm, Tp], [dy, dy], color=col, lw=1.6, alpha=0.5)
    ax3.plot([Tp], [dy], "o", ms=9, color=col)
    ax3.plot([Tm], [dy], "s", ms=9, color=col)
    ax3.text(Tp + 0.06, dy, f"$\\beta={beta}$", fontsize=8.6, color=col, va="center")
ax3.axvline(vold_, color=ORANGE, ls="--", lw=2.0)
ax3.text(vold_, 0.62, "锚点 $v_{\\rm old}$", fontsize=9.0, color=ORANGE, ha="center")
ax3.axvline(v_, color=GREY, ls=":", lw=1.4)
ax3.text(v_, 0.62, "样本 $v$", fontsize=9.0, color=GREY, ha="center")
ax3.set_xlim(-2.6, 2.6); ax3.set_ylim(-0.6, 0.72)
ax3.set_yticks([])
ax3.set_xlabel("$T^\\pm$ 的位置", fontsize=9.8)
ax3.set_title("(c) $\\beta$ 的双重作用", fontsize=10.8)
ax3.text(-2.5, -0.52,
         "① 目标位置 $T^\\pm=v_{\\rm old}\\pm\\frac{v-v_{\\rm old}}{\\beta}$\n"
         "　 $\\beta$ 大 ⇒ 目标【近】⇒ 保守\n"
         "② 梯度量级 $\\propto\\beta^2$\n"
         "　 $\\beta$ 大 ⇒ 梯度【大】⇒ 步子快\n"
         "(起步时 $|\\nabla|\\propto\\beta\\,a\\,|v-v_{\\rm old}|$)",
         fontsize=8.0, color=DARK, va="bottom")
ax3.tick_params(labelsize=8.6)

fig.suptitle("\"两个分身\"在算什么:它们只活在【损失】里,等价于\"让 $v_\\theta$ 去追两个关于锚点对称的目标\"",
             fontsize=11.8, y=0.99)
fig.savefig(RF + "fg_implicit_branches.png", bbox_inches="tight")
print("wrote", RF + "fg_implicit_branches.png")
