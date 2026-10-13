r"""这跟 GRPO 有什么关系?—— "只是得到新训练数据"这个担心对了一半。

  如果损失只有一个分支:      L = ‖v_θ − v‖²
      ⇒ 与奖励无关,只是"复制自己采集到的数据"(RFT/自训练)⇒ 确实不是 RL

  NFT 的损失有两个分支:      L = r‖v⁺_θ − v‖² + (1−r)‖v⁻_θ − v‖²
      v⁺_θ = (1−β)v_old + βv_θ        ← 高奖励时"这个函数"要被拉向样本
      v⁻_θ = (1+β)v_old − βv_θ        ← 低奖励时"这个函数"要被拉向样本
  ⇒ 解出来:好样本把 v_θ 拉向 v;坏样本把 v_θ 推向 2v_old − v
  ⇒ 【同一份答案 v,按奖励决定正用/反用】—— 这才是奖励信号

  判别标准:奖励在组内【没有差异】时(a=0)目标 = v_old ⇒ 【完全不动】✓
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

# 一组的数值例
raw = np.array([1.0, 0.0, 1.0, 1.0])
Z = raw.std()
rn = raw - raw.mean()
a = np.clip(rn / Z, -1, 1)
r = 0.5 + 0.5 * a
print("组内 4 个样本:")
for i in range(4):
    print(f"  样本{i+1}: r_raw={raw[i]:.2f} → r_norm={rn[i]:+.3f} → a={a[i]:+.3f} → r={r[i]:.3f}")

fig = plt.figure(figsize=(15.2, 5.8), dpi=150)

# ---------------- (a) 判别
ax = fig.add_axes([0.03, 0.10, 0.32, 0.78]); ax.axis("off")
ax.text(0.5, 1.00, "(a) 一个分支 vs 两个分支", ha="center", va="top", fontsize=12, color=DARK)
ax.add_patch(plt.Rectangle((0.02, 0.66), 0.96, 0.24, fc="#f7f9fb", ec=GREY, lw=1.6))
ax.text(0.05, 0.855, "普通 FM / RFT(一个分支)", fontsize=10.0, color=GREY, va="top")
ax.text(0.05, 0.795, "$\\mathcal{L}=\\|v_\\theta-v\\|^2$", fontsize=10.4, color=DARK, va="top")
ax.text(0.05, 0.715, "⇒ 与奖励【无关】:只是把采集到的数据\n　 复述一遍 —— 确实不是 RL", fontsize=9.0,
        color=GREY, va="top")
ax.add_patch(plt.Rectangle((0.02, 0.22), 0.96, 0.40, fc="#eef7f1", ec=GREEN, lw=1.8))
ax.text(0.05, 0.585, "DiffusionNFT(两个分支 + 奖励加权)", fontsize=10.0, color=GREEN, va="top")
ax.text(0.05, 0.525,
        "$\\mathcal{L}=r\\|v^+_\\theta-v\\|^2+(1-r)\\|v^-_\\theta-v\\|^2$\n"
        "$v^+_\\theta=(1-\\beta)v_{\\rm old}+\\beta v_\\theta$\n"
        "$v^-_\\theta=(1+\\beta)v_{\\rm old}-\\beta v_\\theta$",
        fontsize=9.2, color=DARK, va="top")
ax.text(0.05, 0.34,
        "⇒ 解出来:\n"
        "　 $r$ 大 → $v_\\theta$ 的目标 $=v$(拉向样本)\n"
        "　 $r$ 小 → $v_\\theta$ 的目标 $=2v_{\\rm old}-v$(推远)",
        fontsize=9.2, color=GREEN, va="top")
ax.text(0.5, 0.13,
        "同一份\"答案\" $v$,按奖励决定\n正着用还是反着用 —— 这才是奖励信号",
        fontsize=9.4, color=RED, ha="center", va="center",
        bbox=dict(fc="#fdecea", ec="#e6a9a0"))
ax.set_xlim(0, 1); ax.set_ylim(0.05, 1.02)

# ---------------- (b) 组内数值例 + 无差异就不动
ax2 = fig.add_axes([0.38, 0.10, 0.30, 0.78]); ax2.axis("off")
ax2.text(0.5, 1.00, "(b) 一组的数值例", ha="center", va="top", fontsize=11.6, color=DARK)
rows = [["样本", "$r_{\\rm raw}$", "$a=2r-1$", "$v_\\theta$ 的目标"]]
for i in range(4):
    tgt = "$v$" if a[i] > 0 else ("$v_{\\rm old}$" if a[i] == 0 else "$2v_{\\rm old}-v$")
    rows.append([f"{i+1}", f"{raw[i]:.1f}", f"{a[i]:+.2f}", tgt])
rows.append(["平均", "0.75", "0", "$v_{\\rm old}$(不动)"])
tb = ax2.table(cellText=rows, cellLoc="center", bbox=[0.0, 0.48, 1.0, 0.46])
tb.auto_set_font_size(False); tb.set_fontsize(8.6)
for (r_, c_), cell in tb.get_celld().items():
    if r_ == 0:
        cell.set_facecolor("#eef2f7")
    if r_ == 5:
        cell.set_facecolor("#fffbe6")
    if c_ == 3 and 0 < r_ < 5:
        cell.set_facecolor("#eef7f1" if "2v" not in cell.get_text().get_text() else "#fdecea")
ax2.text(0.0, 0.42,
         "★ 判别的关键:\n"
         "若这一组奖励【全一样】⇒ $a\\equiv0$\n"
         "⇒ 每个样本的目标都是 $v_{\\rm old}$\n"
         "⇒ 损失最小点就是\"什么都不改\" ✓\n\n"
         "所以 NFT 也【必须】有组内相对比较\n"
         "(绝对奖励同样没用)—— 这一点和 GRPO 一样",
         fontsize=8.8, color=DARK, va="top", bbox=dict(fc="#eef7f1", ec=GREEN))
ax2.text(0.0, 0.06,
         "局部读法:某个 $x_t$ 附近的【有效优势】\n= 该处附近样本优势的加权平均",
         fontsize=8.6, color=DARK, va="bottom", bbox=dict(fc="#fffbe6", ec="#e0c060"))

# ---------------- (c) 与 GRPO 的关系
ax3 = fig.add_axes([0.705, 0.10, 0.28, 0.78]); ax3.axis("off")
ax3.text(0.5, 1.00, "(c) 和 GRPO 的关系", ha="center", va="top", fontsize=11.8, color=DARK)
rows2 = [["", "GRPO", "NFT"],
         ["奖励变成信号", "优势 $A$\n乘在 $\\log\\pi$ 上", "$r$ 决定\n拉近/推远"],
         ["载体", "似然", "回归目标"],
         ["组内相对化", "$(R-\\bar R)/{\\rm std}$", "$R-\\bar R$ →\n压到 $[0,1]$"],
         ["需要参照策略", "old policy\n的 $\\log\\pi$", "锚点\n$v_{\\rm old}$"],
         ["方向从哪来", "噪声 $\\xi$\n(随机探针)", "样本偏移\n$v-v_{\\rm old}$"]]
tb2 = ax3.table(cellText=rows2, cellLoc="center", bbox=[0.0, 0.46, 1.0, 0.48])
tb2.auto_set_font_size(False); tb2.set_fontsize(7.8)
for (r_, c_), cell in tb2.get_celld().items():
    if r_ == 0:
        cell.set_facecolor("#eef2f7")
    if c_ == 1 and r_ > 0:
        cell.set_facecolor("#fdecea")
    if c_ == 2 and r_ > 0:
        cell.set_facecolor("#eef7f1")
ax3.text(0.0, 0.40,
         "深层共同点:\n"
         "两者都在估计\"奖励的梯度方向\"\n\n"
         "· GRPO:用【噪声】当探针,\n"
         "　$\\mathbb{E}[A\\xi]=s\\,\\partial\\mathbb{E}[R]/\\partial m$\n"
         "　(统计估计,无偏但有方差)\n\n"
         "· NFT:用【正负样本对比】直接构造\n"
         "　$v_{\\rm old}+(2r-1)(v-v_{\\rm old})/\\beta$\n"
         "　(确定构造,无方差但无无偏保证)",
         fontsize=8.4, color=DARK, va="top", bbox=dict(fc="#fffbe6", ec="#e0c060"))
ax3.text(0.0, 0.03,
         "所以 NFT 是\"用监督学习实现策略改进\"的混合体,\n不是标准策略梯度算法。",
         fontsize=8.6, color=GREEN, va="bottom", bbox=dict(fc="#eef7f1", ec="#a9d5bb"))

fig.suptitle("重新加噪本身只是\"出题\";让它是 RL 的,是【奖励决定答案正用还是反用】—— 没有奖励差异,模型一步都不动",
             fontsize=11.6, y=0.99)
fig.savefig(RF + "fg_nft_is_rl.png", bbox_inches="tight")
print("wrote", RF + "fg_nft_is_rl.png")
