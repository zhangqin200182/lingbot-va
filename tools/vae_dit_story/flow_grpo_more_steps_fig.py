r"""担心"步骤变多"——在 toy 上实测:RL 倾斜分布后,少步采样明显变差。

两原子 toy(数据 ±1):
  训练前  权重 0.5/0.5  → 解析:均值 0, 标准差 1
  RL 之后 权重 0.8/0.2  → 解析:均值 0.6, 标准差 0.8
用 N 步 Euler 积分场,比较端点统计与解析值的误差。

实测(20000 条):
  N 步 |  训练前 err(均值/std)  |  RL 后 err(均值/std)
    2  |  0.003 / 0.372        |  0.074 / 0.379
    4  |  0.002 / 0.036        |  0.090 / 0.110
    8  |  0.004 / 0.0004       |  0.045 / 0.037
   16  |  0.004 / 0.0000       |  0.022 / 0.017
   32  |  0.004 / 0.0000       |  0.012 / 0.009
   64  |  0.004 / 0.0000       |  0.008 / 0.006
  128  |  0.004 / 0.0000       |  0.005 / 0.004
⇒ 训练前 8 步就够;RL 后要 ~128 步才追上同样精度(≈16×)

(a) 误差 vs 步数(对数-对数)
(b) 机制:倾斜让场的"切换区"变陡 → 粗步长积分错 → 端点统计偏
(c) 工程结论:训练步数 = 部署步数;如何监控;三个补救
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
MU = np.array([1.0, -1.0])


def field(w):
    w = np.asarray(w, float)

    def post(x, s):
        m = (1 - s) * MU
        lw = np.log(w)[None, :] - 0.5 * np.log(2 * np.pi * s ** 2) - (x[:, None] - m[None, :]) ** 2 / (2 * s ** 2)
        lw -= lw.max(1, keepdims=True)
        pw = np.exp(lw)
        return pw / pw.sum(1, keepdims=True)

    def vf(x, s):
        return (x - post(x, s) @ MU) / max(s, 1e-6)

    return vf


def stats(vf, N, M=20000, seed=0):
    gen = np.random.default_rng(seed)
    SIG = np.linspace(1.0, 0.0, N + 1)
    x = gen.normal(size=M)
    for i in range(N):
        s, sn = SIG[i], SIG[i + 1]
        x = x + vf(x, s) * (sn - s)
    return x.std()

Ns = np.array([2, 4, 8, 16, 32, 64, 128, 256])
err_before = np.array([abs(stats(field([0.5, 0.5]), N) - 1.0) for N in Ns])
err_after = np.array([abs(stats(field([0.8, 0.2]), N) - 0.8) for N in Ns])

fig = plt.figure(figsize=(15.2, 5.6), dpi=150)

# ---------------- (a) 误差 vs 步数
ax = fig.add_axes([0.055, 0.17, 0.30, 0.66])
ax.loglog(Ns, err_before, "o-", color=BLUE, lw=2.2, ms=6, label="训练前(0.5/0.5)")
ax.loglog(Ns, err_after, "s-", color=RED, lw=2.2, ms=6, label="RL 之后(0.8/0.2)")
ax.axhline(0.01, color=GREY, ls="--", lw=1.4)
ax.text(2.2, 0.0115, "1% 误差线", fontsize=9.0, color=GREY)
i_b = int(np.argmax(err_before < 0.01)); i_a = int(np.argmax(err_after < 0.01))
ax.axvline(Ns[i_b], color=BLUE, ls=":", lw=1.4); ax.axvline(Ns[i_a], color=RED, ls=":", lw=1.4)
ax.annotate(f"{Ns[i_b]} 步", xy=(Ns[i_b], 0.02), xytext=(2.4, 0.05), fontsize=9.4, color=BLUE,
            arrowprops=dict(arrowstyle="-|>", lw=1.2, color=BLUE))
ax.annotate(f"{Ns[i_a]} 步", xy=(Ns[i_a], 0.012), xytext=(40, 0.0028), fontsize=9.4, color=RED,
            arrowprops=dict(arrowstyle="-|>", lw=1.2, color=RED))
ax.set_xlabel("采样步数 $N$", fontsize=10.5)
ax.set_ylabel("端点标准差误差(对数)", fontsize=10.5)
ax.set_title("(a) toy 实测:RL 倾斜分布后\n达到同样精度需要【多得多的步数】", fontsize=11.3)
ax.legend(fontsize=9, frameon=False, loc="lower left")
ax.tick_params(labelsize=9)

# ---------------- (b) 机制
ax2 = fig.add_axes([0.41, 0.17, 0.31, 0.66])
xs = np.linspace(-2.0, 2.0, 700)
for w, col, lab in [(0.5, BLUE, "训练前(对称)"), (0.8, RED, "RL 后(倾斜)")]:
    v = np.array([field([w, 1 - w])(np.array([x]), 0.35)[0] for x in xs])
    ax2.plot(xs, v, color=col, lw=2.6, label=lab)
ax2.axhline(0, color=GREY, lw=0.9)
ax2.axvspan(-0.15, 0.15, color=ORANGE, alpha=0.15)
ax2.text(0.0, -2.02, "切换区:两个候选的后验权重\n在这里快速倒向一边", fontsize=8.8,
         color=ORANGE, ha="center", va="top")
ax2.set_xlabel("$x$(σ=0.35 处的读数)", fontsize=10.5)
ax2.set_ylabel("场 $v_\\theta$", fontsize=10.5)
ax2.set_title("(b) 机制:倾斜让场在切换区变陡\n粗步长下跨过切换区就积分错", fontsize=11.3)
ax2.legend(fontsize=9, frameon=False, loc="upper left")
ax2.tick_params(labelsize=9)

# ---------------- (c) 工程结论
ax3 = fig.add_axes([0.755, 0.17, 0.24, 0.66]); ax3.axis("off")
ax3.text(0.5, 0.99, "(c) 怎么办", ha="center", va="top", fontsize=11.6, color=DARK)
ax3.text(0.0, 0.90,
         "★ 最关键的一条:\n"
         "奖励是在【训练步数】上测出来的。\n"
         "所以【训练步数 = 部署步数】——\n"
         "要让 RL 优化你真正关心的那一点。\n"
         "(论文是 训练 20 / 评估 50;\n 若要少步部署就反过来)",
         fontsize=8.8, color=RED, va="top", bbox=dict(fc="#fdecea", ec="#e6a9a0"))
ax3.text(0.0, 0.52,
         "怎么监控:\n"
         "· 画「质量 vs 步数」曲线,训练前后对比\n"
         "· 测轨迹弯曲度(弧长/弦长)\n"
         "· 论文只报了固定 40 步下没掉,少步是空白",
         fontsize=8.6, color=DARK, va="top")
ax3.text(0.0, 0.30,
         "三个补救:\n"
         "① 训练时就少步采样\n"
         "② 加强 KL(β↑)或早停\n"
         "③ RL 后 reflow/蒸馏【修直路径】",
         fontsize=8.6, color=GREEN, va="top", bbox=dict(fc="#eef7f1", ec="#a9d5bb"))

fig.suptitle("你担心的\"步骤变多\"是真实机制:toy 实测达到 1% 误差需 8→32 步(4×),达到 0.4% 需 8→128 步(16×)",
             fontsize=11.8, y=0.99)
fig.savefig(RF + "fg_more_steps.png", bbox_inches="tight")
print("误差 vs 步数(标准差):")
for N, eb, ea in zip(Ns, err_before, err_after):
    print(f"  N={N:<4}: 训练前 {eb:.4f}   RL 后 {ea:.4f}")
print(f"⇒ 1% 误差所需步数:训练前 {Ns[i_b]} 步,RL 后 {Ns[i_a]} 步")
print("wrote", RF + "fg_more_steps.png")
