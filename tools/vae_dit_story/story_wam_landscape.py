r"""第五幕的两张图。

Fig 1  六个工作放在「耦合方向 × 想象与执行的关系」二维谱系上
Fig 2  支撑五条判据的关键实证数字(全部来自论文原文)
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
from matplotlib.patches import Rectangle

RED, BLUE, GREEN, DARK, GREY, ORANGE = "#c0392b", "#2471a3", "#1e8449", "#2c3e50", "#95a5a6", "#b9770e"
RF = "/Users/kevin/code/lingbot-va/.claude/scratch/rf/"

# ============================================================ Fig 1
fig = plt.figure(figsize=(15.4, 9.0), dpi=150)
ax = fig.add_axes([0.185, 0.205, 0.795, 0.655])
X = ["双向(联合注意力)", "视频先行(单向 Z→A)", "动作先行(单向 A→Z)", "无耦合(条件独立)"]
Y = ["不生成\n(想象只在训练期)", "联合生成\n(一条去噪, 视频+动作同时出)",
     "串行生成\n(视频链跑完再跑动作链)", "异步生成\n(后台慢钟, 脱离热路径)"]
for i in range(4):
    for j in range(4):
        ax.add_patch(Rectangle((i - 0.5, j - 0.5), 1, 1, fc="#f7f9fb" if (i + j) % 2 else "#ffffff",
                               ec="#e3e8ee", lw=1.0, zorder=0))
ax.set_xlim(-0.5, 3.5); ax.set_ylim(-0.5, 3.5)
ax.set_xticks(range(4)); ax.set_xticklabels(X, fontsize=10.6)
ax.set_yticks(range(4)); ax.set_yticklabels(Y, fontsize=10.2)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)

# (x, y, 名字, 颜色, 一行说明, 水平微调)
WORKS = [
    (3, 0, "FastWAM", GREEN, "视频共训, 推理不生成 · 190 ms", 0.0),
    (2, 0, "Motus2", BLUE, "默认只出动作; 规划/MBRL 时才跑 A→Z→U", 0.0),
    (0, 1, "DreamZero", RED, "单栈 · block 内双向 · 锁步 σ", -0.21),
    (0, 1, "Cosmos 3", RED, "2 塔 MoT · 生成塔全注意力 · 可问 what-if", 0.21),
    (1, 2, "lingbot-va", ORANGE, "动作看【预测视频】 · 20+50 步两段 · 本项目", 0.0),
    (1, 3, "GlanceWAM", GREEN, "单帧 3 s 远视 · 后台 1 步 · 48 ms", 0.0),
]
for x, y, name, col, note, dx in WORKS:
    small = abs(dx) > 0
    ax.plot([x + dx], [y + 0.11], "o", ms=14 if small else 17, color=col,
            mec="w", mew=1.8, zorder=4)
    ax.text(x + dx, y - 0.03, name, ha="center", va="top", fontsize=11.4 if small else 12,
            color=col, fontweight="bold", zorder=5)
    ax.text(x + dx, y + 0.25, note, ha="center", va="bottom", fontsize=8.0 if small else 8.5,
            color="#5d6d7e", zorder=5, bbox=dict(fc="w", ec="none", alpha=0.75, pad=1.2))
ax.set_ylabel("想象与执行的关系", fontsize=11.5, color=DARK, labelpad=10)
ax.text(1.5, -0.78, "视频 ↔ 动作 的耦合方向", fontsize=11.5, color=DARK, ha="center", va="top")
ax.set_title("六个世界-动作模型:耦合方向 × 想象与执行的关系", fontsize=15, pad=12)

fig.text(0.5, 0.098,
         "能问 what-if $p(v|a)$ 的只有【双向】与【动作先行】两列(DreamZero / Cosmos 3 / Motus2);"
         "带 value 与评估闭环的只有 Motus2",
         ha="center", fontsize=10.4, color=DARK)
fig.text(0.5, 0.040,
         "FastWAM 与 Motus2 的默认控制路径都不生成视频;DreamZero / Cosmos 3 是联合同步;"
         "lingbot-va 是串行同步(两段);GlanceWAM 是异步后台",
         ha="center", fontsize=10.4, color=DARK)
fig.savefig(RF + "story_wam_map.png", bbox_inches="tight")

# ============================================================ Fig 2
fig2, axes = plt.subplots(1, 4, figsize=(16.0, 4.3), dpi=150)

ax = axes[0]
names = ["纯共训\n(无前瞻)", "Cosmos Policy\n(同步想象)", "GlanceWAM\n(单层)", "GlanceWAM\n(多层)"]
vals = [64.4, 67.1, 71.5, 72.2]
b = ax.bar(range(4), vals, color=[GREY, ORANGE, GREEN, GREEN], width=0.66)
ax.set_ylim(55, 78); ax.set_xticks(range(4)); ax.set_xticklabels(names, fontsize=8.6)
for r, v in zip(b, vals):
    ax.text(r.get_x() + r.get_width() / 2, v + 0.5, f"{v}%", ha="center", fontsize=9.6, color=DARK)
ax.set_ylabel("RoboCasa 24 任务成功率", fontsize=9.5)
ax.set_title("① 想象有用,但要选对形式", fontsize=11)
ax.tick_params(labelsize=8)

ax = axes[1]
vals = [71.5, 47.0]
b = ax.bar([0, 1], vals, color=[GREEN, RED], width=0.58)
ax.set_ylim(0, 85); ax.set_xticks([0, 1])
ax.set_xticklabels(["prefix-LM mask\n(隔离)", "改成 bidirectional\n(不隔离)"], fontsize=8.8)
for r, v in zip(b, vals):
    ax.text(r.get_x() + r.get_width() / 2, v + 1.5, f"{v}%", ha="center", fontsize=10.5, color=DARK)
ax.set_ylabel("RoboCasa 成功率", fontsize=9.5)
ax.set_title("② 想象必须隔离(最硬的证据)", fontsize=11)
ax.annotate("", xy=(1, 47), xytext=(0, 71.5), arrowprops=dict(arrowstyle="-|>", lw=2, color=RED))
ax.text(0.5, 62, "−24.5 点", ha="center", fontsize=10, color=RED, fontweight="bold")
ax.tick_params(labelsize=8)

ax = axes[2]
ax.plot([0.8], [63.1], "o", color=GREY, ms=7)
ax.plot([1.6, 2.4], [65.7, 65.7], "o-", color=GREY, lw=2.2, ms=7, label="纯共训(表征)")
ax.plot([1.4, 3.0, 3.8], [66.3, 71.6, 71.1], "s-", color=GREEN, lw=2.4, ms=7, label="视觉前瞻(GlanceWAM)")
ax.axhline(64.4, color=RED, ls=":", lw=1.4)
ax.text(0.85, 64.6, "无想象基线 64.4%", fontsize=8.2, color=RED)
ax.set_xlabel("预测时间跨度 H(s)", fontsize=9.5)
ax.set_ylabel("RoboCasa 成功率", fontsize=9.5)
ax.set_title("③ 太近没用,3 s 达峰值", fontsize=11)
ax.legend(fontsize=8.2, frameon=False, loc="lower right")
ax.set_ylim(60, 74); ax.tick_params(labelsize=8)

ax = axes[3]
names = ["同步 WAM\n(Cosmos Policy 等)", "FastWAM\n(6B, 编译)", "GlanceWAM\n(1.6B)"]
vals = [1133, 91.5, 48.0]
b = ax.barh(range(3), vals, color=[RED, ORANGE, GREEN], height=0.55)
ax.set_xscale("log"); ax.set_xlim(20, 6000)
ax.set_yticks(range(3)); ax.set_yticklabels(names, fontsize=8.8)
for r, v in zip(b, vals):
    ax.text(v * 1.15, r.get_y() + r.get_height() / 2, f"{v:g} ms", va="center", fontsize=9.6, color=DARK)
ax.set_xlabel("每 chunk 控制延迟(ms, 对数轴)", fontsize=9.5)
ax.set_title("④ 想象别挂热路径 → 24–80× 差", fontsize=11)
ax.tick_params(labelsize=8)

fig2.suptitle("五条判据背后的实证数字(全部取自论文原文)", fontsize=13)
fig2.tight_layout(rect=[0, 0, 1, 0.90])
fig2.savefig(RF + "story_wam_evidence.png", bbox_inches="tight")
print("wrote", RF + "story_wam_map.png")
print("wrote", RF + "story_wam_evidence.png")
