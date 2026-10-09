r"""增加"歪歪扭扭的 step"的概率,会让以后所有轨迹都变歪吗?—— 不会。

三条理由:
  ① 抖动幅度 s = std_dev_t·√|dσ| 只依赖 σ 与 noise_level,【不含 θ】
     ⇒ RL 改不了抖动幅度,只能挪"中心(场)"
  ② 噪声推挤的【期望】是奖励的梯度,不是随机抖动:
     E[A·ξ] = s·∂E[R]/∂m        (Stein 恒等式 / score function)
     ⇒ 与奖励无关的方向:期望 0,跨迭代互相抵消
  ③ 只有"系统性有用"的方向才会累积 —— 那是学习,不是变歪

(a) 单次实现(抖)vs 场(平滑):RL 改的是后者
(b) Stein 恒等式数值验证
(c) 结论:什么变、什么不变
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
N = 300
SIG = np.linspace(1.0, 0.0, N + 1)
SMAX = SIG[1]; NL = 0.5
V = lambda s: (1 - s) ** 2 + s ** 2
vf = lambda x, s: (x - (1 - s) * x / V(s)) / max(s, 1e-6)
sc = lambda x, s: -x / V(s)
g_of = lambda s: np.sqrt(min(s, SMAX) / (1 - min(s, SMAX))) * NL


def run(mode, x0, gen, nl=NL):
    x = x0; tr = [x]
    for i in range(N):
        s, sn = SIG[i], SIG[i + 1]
        d = sn - s
        if mode == "ode":
            x = x + vf(x, s) * d
        else:
            g = np.sqrt(min(s, SMAX) / (1 - min(s, SMAX))) * nl
            x = x + (vf(x, s) - 0.5 * g ** 2 * sc(x, s)) * d + g * np.sqrt(-d) * gen.normal()
        tr.append(x)
    return np.array(tr)


fig = plt.figure(figsize=(15.2, 5.5), dpi=150)

# ---------------- (a) 单次实现 vs 场
ax = fig.add_axes([0.05, 0.17, 0.30, 0.66])
rng = np.random.default_rng(5)
for k in range(6):
    t = run("sde", -1.0, np.random.default_rng(100 + k))
    ax.plot(SIG, t, color=ORANGE, lw=1.2, alpha=0.8, label=("单次实现($g dW$ 的抖动)" if k == 0 else None))
t_ode = run("ode", -1.0, rng)
ax.plot(SIG, t_ode, color=BLUE, lw=3.0, label="场/均值路径(ODE,平滑,确定性)")
ax.set_xlim(1.02, -0.02); ax.set_ylim(-2.4, 2.0)
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlabel("$\\sigma$(左=1 纯噪声 → 右=0 数据)", fontsize=10.5)
ax.set_title("(a) 抖的是【单次实现】,场一直是平滑的\nRL 改的是场,不是抖动幅度", fontsize=11.3)
ax.legend(fontsize=8.6, frameon=False, loc="lower left")
ax.text(0.55, 1.28, "抖动幅度 $s=\\mathrm{std}_t\\sqrt{|d\\sigma|}$\n只依赖 $\\sigma$ 与 noise_level\n【不含 $\\theta$】",
        fontsize=9.2, color=DARK, ha="center", bbox=dict(fc="#fffbe6", ec="#e0c060"))

# ---------------- (b) Stein 恒等式
ax2 = fig.add_axes([0.42, 0.17, 0.31, 0.66])
rng = np.random.default_rng(0); M = 400_000; s = 1.0
ms = np.array([-0.5, 0.0, 0.5, 1.0, 1.5])
mc, th = [], []
for m in ms:
    xi = rng.normal(size=M)
    R = -((m + s * xi) - 1.0) ** 2
    mc.append(np.mean(R * xi))
    th.append(s * (-2 * (m - 1)))
x = np.arange(len(ms))
ax2.bar(x - 0.19, mc, width=0.36, color=BLUE, label="蒙特卡洛 $\\mathbb{E}[A\\xi]$(200 万样本)")
ax2.bar(x + 0.19, th, width=0.36, color=GREEN, label="解析 $s\\,\\partial\\mathbb{E}[R]/\\partial m$")
ax2.axhline(0, color=GREY, lw=0.9)
ax2.set_xticks(x); ax2.set_xticklabels([f"$m={m:+.1f}$" for m in ms], fontsize=9.2)
ax2.set_xlabel("均值 $m$", fontsize=10.5); ax2.set_ylabel("$\\mathbb{E}[A\\xi]$", fontsize=10.5)
ax2.set_title("(b) 关键恒等式:噪声推挤的【期望】\n= 奖励对均值的梯度(不是随机抖动)", fontsize=11.3)
ax2.legend(fontsize=8.4, frameon=False, loc="upper right")
ax2.tick_params(labelsize=9)

# ---------------- (c) 结论
ax3 = fig.add_axes([0.755, 0.17, 0.24, 0.66]); ax3.axis("off")
ax3.text(0.5, 0.99, "(c) 什么变、什么不变", ha="center", va="top", fontsize=11.6, color=DARK)
rows = [["", "会变?", "为什么"],
        ["抖动幅度 $s$", "否", "只依赖 $\\sigma$、noise_level\n不含 $\\theta$"],
        ["场 $v_\\theta$/均值路径", "是", "RL 就是改它"],
        ["单次实现的形状", "会", "因为场变了\n(但抖动幅度不变)"],
        ["与奖励无关的推挤", "不会累积", "$\\mathbb{E}[A\\xi]=0$\n跨迭代抵消"]]
tb = ax3.table(cellText=rows, cellLoc="center", bbox=[0.0, 0.49, 1.0, 0.43])
tb.auto_set_font_size(False); tb.set_fontsize(7.8)
for (r, c), cell in tb.get_celld().items():
    if r == 0:
        cell.set_facecolor("#eef2f7")
    if c == 1:
        t = cell.get_text().get_text()
        cell.set_facecolor({"否": "#eef7f1", "是": "#fdecea", "会": "#fff4e5",
                            "不会累积": "#eef7f1"}.get(t, "w"))
ax3.text(0.0, 0.46,
         "二分法:\n"
         "$\\mathbb{E}[\\Delta m]\\propto s\\,\\partial\\mathbb{E}[R]/\\partial m$  ← 学习方向\n"
         "$\\mathrm{Var}[\\Delta m]\\propto\\mathrm{Var}(A\\xi)$           ← 运气($\\propto1/G$)\n\n"
         "所以:噪声负责【探路】,\n奖励负责【筛选】,\n平均负责【去随机】。",
         fontsize=8.8, color=DARK, va="top",
         bbox=dict(fc="#fffbe6", ec="#e0c060"))
ax3.text(0.0, 0.03,
         "歪歪扭扭属于【单次实现】;\n场学到的是【期望上更好的方向】。",
         fontsize=8.8, color=GREEN, va="bottom", bbox=dict(fc="#eef7f1", ec="#a9d5bb"))

fig.suptitle("增加\"歪扭 step\"的概率,会让以后轨迹都变歪吗?—— 不会:抖动幅度与 $\\theta$ 无关,而推挤的期望是奖励梯度",
             fontsize=11.8, y=0.99)
fig.savefig(RF + "fg_no_wiggle.png", bbox_inches="tight")
print("Stein 恒等式核对(单位 $s=1$):")
for m, a, b in zip(ms, mc, th):
    print(f"  m={m:+.1f}: 蒙特卡洛 {a:+.5f}  vs  解析 {b:+.5f}   误差 {abs(a-b):.5f}")
print("wrote", RF + "fg_no_wiggle.png")
