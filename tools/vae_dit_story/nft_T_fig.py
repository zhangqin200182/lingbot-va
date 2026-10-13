r"""T 是什么?—— "如果只剩这一项,网络输出 v_θ 应该变成的目标值"。

  第一项 ‖v⁺_θ − v‖²,  v⁺_θ = (1−β)v_old + βv_θ
      令括号为 0:(1−β)v_old + βv_θ = v
      ⇒ βv_θ = v − (1−β)v_old
      ⇒ v_θ = v_old + (v − v_old)/β  =: T⁺        ← "模仿目标"

  第二项 ‖v⁻_θ − v‖²,  v⁻_θ = (1+β)v_old − βv_θ
      令括号为 0:(1+β)v_old − βv_θ = v
      ⇒ v_θ = v_old − (v − v_old)/β  =: T⁻        ← "反学目标"

⇒ T⁺ / T⁻ 关于锚点 v_old 【对称】,各在距离 |v−v_old|/β 处
   T⁺ 在样本那一侧;T⁻ 在对面
⇒ 网络真正要追的目标 = 按奖励在两者之间滑动:T = rT⁺ + (1−r)T⁻
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

fig = plt.figure(figsize=(15.2, 5.7), dpi=150)

# ---------------- (a) 推导
ax = fig.add_axes([0.03, 0.10, 0.31, 0.80]); ax.axis("off")
ax.text(0.5, 1.00, "(a) $T^\\pm$ 是怎么解出来的", ha="center", va="top", fontsize=11.8, color=DARK)
ax.text(0.0, 0.92,
         "第一项:$\\|v^+_\\theta-v\\|^2$,$\\ v^+_\\theta=(1-\\beta)v_{\\rm old}+\\beta v_\\theta$\n"
         "　令括号 $=0$:$(1-\\beta)v_{\\rm old}+\\beta v_\\theta=v$\n"
         "　$\\Rightarrow \\beta v_\\theta=v-(1-\\beta)v_{\\rm old}$\n"
         "　$\\Rightarrow v_\\theta=v_{\\rm old}+\\dfrac{v-v_{\\rm old}}{\\beta}=:T^+$",
         fontsize=9.2, color=BLUE, va="top")
ax.text(0.0, 0.60,
         "第二项:$\\|v^-_\\theta-v\\|^2$,$\\ v^-_\\theta=(1+\\beta)v_{\\rm old}-\\beta v_\\theta$\n"
         "　令括号 $=0$:$\\Rightarrow v_\\theta=v_{\\rm old}-\\dfrac{v-v_{\\rm old}}{\\beta}=:T^-$",
         fontsize=9.2, color=ORANGE, va="top")
ax.text(0.0, 0.38,
         "所以 $T^\\pm$ 的含义非常直白:\n"
         "\"若这一项【独占】损失,网络输出该变成的值\"\n\n"
         "网络真正要追的目标($r$ 加权平均):\n"
         "$T=rT^++(1-r)T^-$",
         fontsize=9.2, color=GREEN, va="top", bbox=dict(fc="#eef7f1", ec=GREEN))
ax.text(0.0, 0.10,
         "为什么引入 $T$:把\"两个奇怪的分身\"翻译成\n最熟悉的语言 —— 【加权平方误差回归】。",
         fontsize=9.0, color=DARK, va="top", bbox=dict(fc="#fffbe6", ec="#e0c060"))

# ---------------- (b) 几何:对称 + 滑动
ax2 = fig.add_axes([0.37, 0.16, 0.36, 0.68])
vold_, v_, beta = 0.0, 1.0, 0.75
d = v_ - vold_
Tp, Tm = vold_ + d / beta, vold_ - d / beta
ax2.axhline(0, color=GREY, lw=1.2)
ax2.axvline(vold_, color=ORANGE, ls="--", lw=2.2)
ax2.text(vold_, 0.42, "锚点 $v_{\\rm old}$", fontsize=9.2, color=ORANGE, ha="center")
ax2.plot([v_], [0], "^", ms=13, color=BLUE)
ax2.text(v_, -0.30, "样本 $v$", fontsize=9.4, color=BLUE, ha="center", va="top")
ax2.plot([Tp], [0], "o", ms=12, color=GREEN)
ax2.text(Tp, 0.12, "$T^+$\n(模仿目标)", fontsize=9.0, color=GREEN, ha="center")
ax2.plot([Tm], [0], "s", ms=12, color=ORANGE)
ax2.text(Tm, 0.12, "$T^-$\n(反学目标)", fontsize=9.0, color=ORANGE, ha="center")
for x0, x1, col in [(vold_, Tp, GREEN), (Tm, vold_, ORANGE)]:
    ax2.add_patch(FancyArrowPatch((x0, -0.16), (x1, -0.16), arrowstyle="<|-|>",
                                  mutation_scale=10, lw=1.6, color=col))
ax2.text((vold_ + Tp) / 2, -0.24, "$|v-v_{\\rm old}|/\\beta$", fontsize=8.4, color=GREEN, ha="center")
ax2.text((Tm + vold_) / 2, -0.24, "$|v-v_{\\rm old}|/\\beta$", fontsize=8.4, color=ORANGE, ha="center")
# 滑动:T 随 r
for r_, c_ in [(1.0, BLUE), (0.75, PURPLE), (0.5, DARK), (0.0, ORANGE)]:
    T = r_ * Tp + (1 - r_) * Tm
    ax2.plot([T], [0.62], "o", ms=9, color=c_)
    ax2.text(T, 0.70, f"$r={r_}$", fontsize=8.4, color=c_, ha="center")
ax2.add_patch(FancyArrowPatch((Tm, 0.62), (Tp, 0.62), arrowstyle="-|>", mutation_scale=11,
                              lw=1.4, color=GREY, ls=":"))
ax2.text(vold_, 0.92, "$T$ 随奖励在这两个对称目标之间滑动", fontsize=9.4, color=DARK, ha="center")
ax2.set_xlim(-2.4, 2.4); ax2.set_ylim(-0.55, 1.05)
ax2.set_yticks([])
ax2.set_xlabel("$v_\\theta$ 的目标值(一维投影;此图取 $\\beta=0.75$)", fontsize=10.0)
ax2.set_title("(b) $T^+$ 与 $T^-$ 关于锚点对称;$T$ 在两者之间滑动", fontsize=11.2)
ax2.tick_params(labelsize=8.8)

# ---------------- (c) 数值表
ax3 = fig.add_axes([0.76, 0.10, 0.23, 0.80]); ax3.axis("off")
ax3.text(0.5, 1.00, "(c) 数值($v_{\\rm old}=0,\\ v=1$)", ha="center", va="top",
         fontsize=11.2, color=DARK)
rows = [["$\\beta$", "$T^+$", "$T^-$"],
        ["1.0", "$+1.0$", "$-1.0$"],
        ["0.5", "$+2.0$", "$-2.0$"],
        ["2.0", "$+0.5$", "$-0.5$"]]
tb = ax3.table(cellText=rows, cellLoc="center", bbox=[0.0, 0.62, 1.0, 0.26])
tb.auto_set_font_size(False); tb.set_fontsize(9.0)
for (r_, c_), cell in tb.get_celld().items():
    if r_ == 0:
        cell.set_facecolor("#eef2f7")
    if c_ == 1 and r_ > 0:
        cell.set_facecolor("#eef7f1")
    if c_ == 2 and r_ > 0:
        cell.set_facecolor("#fdf3e7")
rows2 = [["$r$", "$T$", "含义"],
         ["1.0", "$+1.0$", "模仿"],
         ["0.75", "$+0.5$", "偏模仿"],
         ["0.5", "$0.0$", "不动"],
         ["0.0", "$-1.0$", "反学"]]
tb2 = ax3.table(cellText=rows2, cellLoc="center", bbox=[0.0, 0.18, 1.0, 0.36])
tb2.auto_set_font_size(False); tb2.set_fontsize(8.6)
for (r_, c_), cell in tb2.get_celld().items():
    if r_ == 0:
        cell.set_facecolor("#eef2f7")
    if r_ == 3:
        cell.set_facecolor("#fffbe6")
ax3.text(0.0, 0.13,
         "$\\beta=1$ 时 $T^+=v$:模仿目标就是样本本身\n"
         "$\\beta$ 越小 ⇒ 目标越远(越激进)",
         fontsize=8.4, color=DARK, va="top",
         bbox=dict(fc="#fffbe6", ec="#e0c060"))

fig.suptitle("$T$ 是什么:它是\"若这一项独占损失,网络输出该变成的值\" —— 两项给出两个关于锚点对称的目标",
             fontsize=11.8, y=0.99)
fig.savefig(RF + "fg_T_meaning.png", bbox_inches="tight")

vold_, v_, beta = 0.0, 1.0, 0.75
print(f"β={beta}, v_old={vold_}, v={v_}:  T+={vold_+(v_-vold_)/beta:+.3f}, T-={vold_-(v_-vold_)/beta:+.3f}")
for r_ in [1.0, 0.75, 0.5, 0.0]:
    T = r_ * (vold_ + (v_ - vold_) / beta) + (1 - r_) * (vold_ - (v_ - vold_) / beta)
    print(f"  r={r_:4.2f} → T = {T:+.4f}")
print("wrote", RF + "fg_T_meaning.png")
