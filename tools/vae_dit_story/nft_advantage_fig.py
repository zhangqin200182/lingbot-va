r"""DiffusionNFT 怎么得到组间优势?

① 候选多样性从哪来:每个样本用【不同的初始噪声 ε】(确定性 ODE 也可)⇒ K 张不同
② 优势怎么算(Algorithm 1 第 4–5 行):
      r_norm = r_raw − mean({r_raw})          ← 组内减均值(和 GRPO 第一步一样)
      a      = clip{r_norm / Z_c, −1, +1}     ← 归一化 + 截断到 [−1,1]
      r      = 0.5 + 0.5·a ∈ [0,1]            ← 映射成"最优概率"
   ⇒ 2r − 1 = a,就是【组内相对优势】
③ 优势怎么用:不再乘在 log π 上,而是作为"朝样本走多远"的系数
      最小化 L(r) = r‖v⁺_θ − v‖² + (1−r)‖v⁻_θ − v‖²  得(见数值验证):
      v_θ 的目标 = v_old + (2r−1)·(v − v_old)/β
    ⇒ a=+1:目标 = v(模仿) ;a=0:目标 = v_old(不更新!) ;a=−1:目标 = 反射点(反学)
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

# ---------- 数值验证:最小化 L(r) 得到的目标
def opt_target(r, beta, v_old=-0.3, v=1.0, lo=-3, hi=3):
    g = np.linspace(lo, hi, 400001)
    vp = (1 - beta) * v_old + beta * g
    vm = (1 + beta) * v_old - beta * g
    L = r * (vp - v) ** 2 + (1 - r) * (vm - v) ** 2
    return g[int(np.argmin(L))]

beta = 1.0
print(f"数值验证(β={beta}, v_old=-0.3, v=+1.0):")
print("   a=2r-1 |  数值最优 v_θ  |  解析 v_old+a(v-v_old)/β")
for r in [1.0, 0.75, 0.5, 0.25, 0.0]:
    a = 2 * r - 1
    num = opt_target(r, beta)
    ana = -0.3 + a * (1.0 - (-0.3)) / beta
    print(f"   {a:+.1f}  |   {num:+.4f}      |   {ana:+.4f}")

fig = plt.figure(figsize=(15.2, 5.6), dpi=150)

# ---------------- (a) 多样性来自初始噪声
ax = fig.add_axes([0.05, 0.17, 0.29, 0.66])
SIG = np.linspace(1.0, 0.0, 200)
gen = np.random.default_rng(3)
for k in range(6):
    e = gen.normal()
    x = e; tr = [x]
    for i in range(199):
        s, sn = SIG[i], SIG[i + 1]
        x = x + (x / max(s, 1e-6)) * (sn - s) if False else x + (1.0 * x / max(s, 1e-6)) * 0
    # 用单原子场(x_σ=e·σ+... )简化:线性插值场 v=(x-c)/σ 的解是直线
    tr = np.array([e * s + 0.55 * (1 - s) for s in SIG])
    ax.plot(SIG, tr, lw=2.0, color=(BLUE if e > 0 else RED), alpha=0.9)
ax.plot([1], [0.55], "o", ms=0)
ax.axhline(0.55, color=GREY, ls=":", lw=1.2)
ax.text(0.03, 0.62, "数据 $c=0.55$", fontsize=9.2, color=GREY)
ax.set_xlim(1.02, -0.02); ax.set_ylim(-2.2, 2.2)
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlabel("$\\sigma$(左=1 纯噪声 → 右=0 数据)", fontsize=10.5)
ax.set_title("(a) 多样性来自【初始噪声 ε】\n$K$ 个样本各抽一个,确定性 ODE 也够", fontsize=11.3)
ax.text(0.05, 1.75, "$K$ 条轨迹 = $K$ 个不同的 $\\epsilon$\n(不需要在采样中注入噪声)",
        fontsize=9.2, color=DARK, bbox=dict(fc="#fffbe6", ec="#e0c060"))

# ---------------- (b) 优势怎么算、怎么用
ax2 = fig.add_axes([0.40, 0.17, 0.35, 0.66]); ax2.axis("off")
ax2.text(0.5, 1.00, "(b) 优势:算出来 → 当\"走多远\"的系数", ha="center", va="top",
         fontsize=11.6, color=DARK)
ax2.text(0.0, 0.91,
         "$r_{\\rm norm}=r_{\\rm raw}-\\mathrm{mean}(\\{r_{\\rm raw}\\})$\n"
         "$a=\\mathrm{clip}\\{r_{\\rm norm}/Z_c,\\,-1,\\,+1\\}$\n"
         "$r=0.5+0.5a\\ \\in[0,1]$\n\n"
         "⇒ $2r-1=a$ 就是【组内相对优势】",
         fontsize=9.6, color=BLUE, va="top", bbox=dict(fc="#eaf2fa", ec=BLUE))
ax2.text(0.0, 0.60,
         "最小化 $\\mathcal{L}=r\\|v_\\theta^+-v\\|^2+(1-r)\\|v_\\theta^--v\\|^2$:\n"
         "$v_\\theta$ 的目标 $=v_{\\rm old}+(2r-1)\\cdot\\dfrac{v-v_{\\rm old}}{\\beta}$\n\n"
         "$a=+1$:目标 $=v$ → 模仿\n"
         "$a=0$:目标 $=v_{\\rm old}$ → 【不更新】✓\n"
         "$a=-1$:目标 $=2v_{\\rm old}-v$ → 反着学",
         fontsize=9.4, color=GREEN, va="top", bbox=dict(fc="#eef7f1", ec=GREEN))
ax2.text(0.0, 0.10,
         "对比 GRPO:$A_i$ 是乘在 $\\log\\pi$ 上的(策略梯度),\n还有 ratio/clip/KL 兜着;\n"
         "NFT:$a$ 只当【回归目标的插值系数】——\n目标本身有界,所以不需要 ratio/clip。",
         fontsize=8.8, color=DARK, va="top", bbox=dict(fc="#fffbe6", ec="#e0c060"))

# ---------------- (c) 几何 + 对照
ax3 = fig.add_axes([0.775, 0.17, 0.215, 0.66])
for a, c, lab in [(1.0, BLUE, "$a=+1$\n模仿"), (0.0, DARK, "$a=0$\n不更新"), (-1.0, ORANGE, "$a=-1$\n反学")]:
    tgt = -0.3 + a * (1.0 - (-0.3)) / 1.0
    ax3.plot([tgt], [a], "o", ms=12, color=c)
    ax3.annotate("", xy=(tgt, a), xytext=(-0.3, a),
                 arrowprops=dict(arrowstyle="-|>", lw=2.2, color=c))
    ax3.text(tgt + (0.08 if a >= 0 else -0.08), a, lab, fontsize=9.0, color=c,
             ha="left" if a >= 0 else "right", va="center")
ax3.axvline(-0.3, color=GREY, ls="--", lw=1.6)
ax3.text(-0.3, 1.35, "$v_{\\rm old}$", fontsize=10, color=GREY, ha="center")
ax3.axvline(1.0, color=BLUE, ls=":", lw=1.4)
ax3.text(1.0, 1.35, "样本 $v$", fontsize=10, color=BLUE, ha="center")
ax3.set_xlim(-1.8, 1.8); ax3.set_ylim(-1.6, 1.6)
ax3.set_yticks([-1, 0, 1]); ax3.set_yticklabels(["$a=-1$", "$0$", "$a=+1$"], fontsize=9)
ax3.set_xlabel("$v_\\theta$ 的目标(在 $v_{\\rm old}$ 与 $v$ 的连线上)", fontsize=9.4)
ax3.set_title("(c) 优势只决定\n\"朝样本走多远\"", fontsize=11.0)
ax3.tick_params(labelsize=8.6)

fig.suptitle("DiffusionNFT 的组间优势:$K$ 个不同初始噪声 → 打分 → 组内减均值 → 优势变成\"朝样本走多远\"的系数",
             fontsize=11.8, y=0.99)
fig.savefig(RF + "fg_nft_advantage.png", bbox_inches="tight")
print("\n⇒ 优势 a=2r−1 只当作【回归目标的插值系数】:a=0 时目标=v_old,恰好不更新")
print("wrote", RF + "fg_nft_advantage.png")
