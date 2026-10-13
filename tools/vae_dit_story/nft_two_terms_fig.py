r"""两项形式明明一样,凭什么一正一反?

  表面:两项都是 ‖◇ − v‖²,都是"让那个东西等于 v"
  但那个东西与 v_θ 的连接符号【相反】:
      v⁺_θ = v_old + β(v_θ − v_old)     ← v_θ 的系数 +β
      v⁻_θ = v_old − β(v_θ − v_old)     ← v_θ 的系数 −β
  ⇒ 链式法则后梯度【反号】:
      ∂/∂v_θ ‖v⁺_θ−v‖² = 2(v⁺_θ−v)(+β)
      ∂/∂v_θ ‖v⁻_θ−v‖² = 2(v⁻_θ−v)(−β)
  ⇒ "让 v⁺ 等于 v" 翻译成对 v_θ 的要求就是 T⁺;"让 v⁻ 等于 v"就是 T⁻
      T⁺ = v_old + (v−v_old)/β   (在样本那一侧)
      T⁻ = v_old − (v−v_old)/β   (在对面)
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

fig = plt.figure(figsize=(15.2, 5.7), dpi=150)

# ---------------- (a) 表面矛盾
ax = fig.add_axes([0.03, 0.10, 0.30, 0.80]); ax.axis("off")
ax.text(0.5, 1.00, "(a) 表面上的矛盾", ha="center", va="top", fontsize=12, color=DARK)
ax.add_patch(plt.Rectangle((0.02, 0.60), 0.96, 0.32, fc="#f7f9fb", ec=GREY, lw=1.6))
ax.text(0.06, 0.87, "两项形式完全一样:", fontsize=10.0, color=DARK, va="top")
ax.text(0.06, 0.775, "$\\|v^+_\\theta-v\\|^2$     和     $\\|v^-_\\theta-v\\|^2$",
        fontsize=11.0, color=DARK, va="top")
ax.text(0.06, 0.655, "都是\"让那个东西【等于 $v$】\"\n⇒ 看起来两项都在做同一件事",
        fontsize=9.2, color=GREY, va="top")
ax.text(0.5, 0.50, "那凭什么一个模仿、一个反学?", fontsize=10.6, color=RED, ha="center",
        va="center", bbox=dict(fc="#fdecea", ec="#e6a9a0"))
ax.text(0.0, 0.36,
         "钥匙:那个\"东西\"和 $v_\\theta$ 的连接\n符号是【相反】的:\n\n"
         "　$v^+_\\theta=v_{\\rm old}+\\beta(v_\\theta-v_{\\rm old})$  系数 $+\\beta$\n"
         "　$v^-_\\theta=v_{\\rm old}-\\beta(v_\\theta-v_{\\rm old})$  系数 $-\\beta$",
         fontsize=9.4, color=GREEN, va="top", bbox=dict(fc="#eef7f1", ec=GREEN))
ax.text(0.0, 0.06,
         "形式相同 ⇒ 要求相同;\n连接符号相反 ⇒ 对 $v_\\theta$ 的要求相反。",
         fontsize=9.0, color=DARK, va="top", bbox=dict(fc="#fffbe6", ec="#e0c060"))
ax.set_xlim(0, 1); ax.set_ylim(0.03, 1.02)

# ---------------- (b) 梯度反号(数值)
ax2 = fig.add_axes([0.36, 0.10, 0.31, 0.80]); ax2.axis("off")
ax2.text(0.5, 1.00, "(b) 链式法则后梯度反号", ha="center", va="top", fontsize=11.8, color=DARK)
ax2.text(0.0, 0.92,
         "$\\dfrac{\\partial}{\\partial v_\\theta}\\|v^+_\\theta-v\\|^2=2(v^+_\\theta-v)\\cdot(+\\beta)$\n"
         "$\\dfrac{\\partial}{\\partial v_\\theta}\\|v^-_\\theta-v\\|^2=2(v^-_\\theta-v)\\cdot(-\\beta)$",
         fontsize=9.6, color=DARK, va="top", bbox=dict(fc="#f7f9fb", ec="#c8d4e0"))
ax2.text(0.0, 0.66,
         "起点 $v_\\theta=v_{\\rm old}=0$,$\\ \\beta=1$,$\\ v_{\\rm old}=0$,$\\ v=1$:\n\n"
         "　第一项:值 $=1.000$,梯度 $=-2.000$\n"
         "　　　　⇒ 梯度下降把 $v_\\theta$ 往上推(朝 $v$)\n\n"
         "　第二项:值 $=1.000$,梯度 $=+2.000$\n"
         "　　　　⇒ 梯度下降把 $v_\\theta$ 往下推(背离 $v$)",
         fontsize=9.2, color=DARK, va="top")
ax2.text(0.0, 0.34,
         "★ 两项的【值】一样(都是 1.000),\n"
         "　 但【梯度】一模一样吗?不 —— 符号相反。\n\n"
         "所以\"形式相同\"完全不妨碍\n\"一个拉近、一个推远\"。",
         fontsize=9.2, color=RED, va="top", bbox=dict(fc="#fdecea", ec="#e6a9a0"))
ax2.text(0.0, 0.08,
         "一句话:两项在\"要求分身等于 $v$\"这件事上相同;\n"
         "在\"这对 $v_\\theta$ 意味着什么\"这件事上相反。",
         fontsize=9.0, color=GREEN, va="top", bbox=dict(fc="#eef7f1", ec="#a9d5bb"))

# ---------------- (c) 翻译 + 两种写法
ax3 = fig.add_axes([0.70, 0.10, 0.29, 0.80]); ax3.axis("off")
ax3.text(0.5, 1.00, "(c) 翻译成\"对 $v_\\theta$ 的要求\"", ha="center", va="top",
         fontsize=11.4, color=DARK)
ax3.text(0.0, 0.92,
         "第一项要求 $v^+_\\theta=v$  即  要求 $v_\\theta=T^+$\n"
         "第二项要求 $v^-_\\theta=v$  即  要求 $v_\\theta=T^-$\n\n"
         "$T^+=v_{\\rm old}+\\dfrac{v-v_{\\rm old}}{\\beta}$(样本那侧)\n"
         "$T^-=v_{\\rm old}-\\dfrac{v-v_{\\rm old}}{\\beta}$(对面)\n\n"
         "⇒ 两个要求关于锚点【对称】",
         fontsize=9.2, color=DARK, va="top")
ax3.text(0.0, 0.56,
         "谁主导?由权重决定:\n"
         "　$r\\approx1$ ⇒ 第一项重 ⇒ 推向 $T^+$(模仿)\n"
         "　$r\\approx0$ ⇒ 第二项重 ⇒ 推向 $T^-$(反学)\n"
         "　$r=0.5$ ⇒ 等权 ⇒ 目标 $=v_{\\rm old}$(不动)",
         fontsize=9.0, color=GREEN, va="top", bbox=dict(fc="#eef7f1", ec=GREEN))
ax3.text(0.0, 0.30,
         "两种写法完全等价,各有用处:\n"
         "· 分身写法 $v^\\pm_\\theta$ → 便于讲\"两个策略\"\n"
         "　 (论文定理的出发点)\n"
         "· 目标写法 $T^\\pm$ → 便于看梯度\n"
         "　 $\\nabla_\\theta\\mathcal{L}=2\\beta^2(v_\\theta-T)\\partial v_\\theta/\\partial\\theta$",
         fontsize=8.8, color=DARK, va="top", bbox=dict(fc="#fffbe6", ec="#e0c060"))

fig.suptitle("两项形式一样,凭什么一正一反?—— 因为分身与 $v_\\theta$ 的连接符号相反(+β / −β),链式法则后梯度反号",
             fontsize=11.6, y=0.99)
fig.savefig(RF + "fg_two_terms_sign.png", bbox_inches="tight")
print("wrote", RF + "fg_two_terms_sign.png")
