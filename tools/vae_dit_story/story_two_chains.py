"""Timeline of the two diffusion chains: video loop then action loop,
with the KV-cache write / read moments and the visibility rules.

Facts taken from the code:
  * two independent schedulers, two independent randn draws
  * video loop runs FIRST, action loop runs SECOND
  * update_cache=0 -> write the current tokens temporarily, attend, then remove
    update_cache=1 -> write and KEEP  (marks is_pred=True: it is a prediction)
    update_cache=2 -> write and KEEP  (is_pred=False: it is a ground-truth observation)
  * the last step of each loop is the padded t=0 step -> it only writes the cache
  * the action loop therefore reads a cache that already contains THIS chunk's video
"""
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
# macOS ships a CJK-capable face; register it so the Chinese labels render
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
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

OUT = "/Users/kevin/code/lingbot-va/.claude/scratch/rf/story_two_chains.png"
RED, BLUE, GREEN, GREY, DARK = "#c0392b", "#2471a3", "#1e8449", "#b9c2cb", "#2c3e50"

fig, ax = plt.subplots(figsize=(13.6, 8.0), dpi=150)
ax.set_xlim(0, 88); ax.set_ylim(0, 62); ax.axis("off")

# ---------------------------------------------------------------- geometry
BW, BH, Y_VID, Y_ACT = 7.4, 4.4, 48.0, 35.5
Y_CACHE = 21.0
vid = [("1.00", 0), ("0.72", 0), ("0.38", 0), ("0.00", 1)]
act = [("1.00", 0), ("0.80", 0), ("0.55", 0), ("0.25", 0), ("0.00", 1)]
X0 = 17.0
GAP = 1.4


def row(items, y, color, title, sub):
    xs = []
    x = X0
    for val, wr in items:
        ax.add_patch(FancyBboxPatch((x, y), BW, BH, boxstyle="round,pad=0.35,rounding_size=0.8",
                                    fc=color, ec="w", lw=1.2, alpha=0.92))
        ax.text(x + BW / 2, y + BH - 1.35, val, ha="center", va="center", color="w", fontsize=9.2,
                fontweight="bold" if wr else "normal")
        ax.text(x + BW / 2, y + 1.35, "$t$", ha="center", va="center", color="w", fontsize=7.5, alpha=0.85)
        if wr:
            ax.text(x + BW / 2, y + BH + 1.9, "write → cache", ha="center", fontsize=8.2, color=color,
                    fontweight="bold")
            ax.add_patch(FancyArrowPatch((x + BW / 2, y - 0.4), (x + BW / 2, Y_CACHE + 4.0),
                                         arrowstyle="-|>", mutation_scale=13, lw=1.6, color=color,
                                         zorder=0.5))
        xs.append(x)
        x += BW + GAP
    ax.text(X0 - 1.2, y + BH / 2, title, ha="right", va="center", fontsize=11, color=DARK, fontweight="bold")
    ax.text(X0 - 1.2, y + BH / 2 - 2.3, sub, ha="right", va="center", fontsize=8, color="#7f8c8d")
    return xs, x


xv, x_end_v = row(vid, Y_VID, RED, "视频链", "先跑,独立 noise / $\\sigma$ 网格 / CFG")
xa, x_end_a = row(act, Y_ACT, BLUE, "动作链", "后跑,另一份 noise / $\\sigma$ 网格 / CFG")

# read-brackets: what each loop reads from the cache
def bracket(x1, x2, y, color, label):
    ax.add_patch(FancyArrowPatch((x1, y + 1.5), (x2, y + 1.5), arrowstyle="-", lw=1.5, color=color))
    ax.add_patch(FancyArrowPatch((x1, y + 1.5), (x1, y), arrowstyle="-", lw=1.5, color=color))
    ax.add_patch(FancyArrowPatch((x2, y + 1.5), (x2, y), arrowstyle="-", lw=1.5, color=color))
    ax.text((x1 + x2) / 2, y - 0.5, label, ha="center", va="top", fontsize=8.6, color=color)


bracket(X0, x_end_v - GAP - 0.5, Y_VID - 3.2, RED,
        "读 cache:观测 K/V + 更早 chunk 的 视频/动作 K/V     (看不到本 chunk 的动作 —— 那时还不存在)")
bracket(X0, x_end_a - GAP - 0.5, Y_ACT - 3.2, BLUE,
        "读 cache:观测 K/V + 更早 chunk + 【本 chunk 的视频 K/V】(视频链刚写进去)")

# ---------------------------------------------------------------- cache band
ax.add_patch(Rectangle((X0 - 9.0, Y_CACHE - 3.6), x_end_a - X0 + 10.4 - GAP, 7.6,
                       fc="#f4f6f8", ec=GREY, lw=1.0))
ax.text(X0 - 10.0, Y_CACHE + 0.2, "KV cache\n(滑动窗口,\n按 id 淘汰最旧)", ha="right", va="center",
        fontsize=9, color=DARK)
blocks = [("观测 K/V\n(update_cache=2\nis_pred=False)", GREEN, 0.0, 9.0),
          ("更早 chunk 的\n视频+动作 K/V", GREY, 9.0, 9.0),
          ("本 chunk 视频 K/V\n(update_cache=1)", RED, 18.0, 9.0),
          ("本 chunk 动作 K/V\n(update_cache=1)", BLUE, 27.0, 9.0)]
x = X0 - 9.0
for label, col, _, w in blocks:
    ax.add_patch(Rectangle((x, Y_CACHE - 2.2), w - 0.7, 5.2, fc=col, ec="w", lw=1.0, alpha=0.85))
    ax.text(x + (w - 0.7) / 2, Y_CACHE + 0.4, label, ha="center", va="center", color="w", fontsize=7.2)
    x += w
ax.text(x + 0.6, Y_CACHE + 0.4, "…→ 下一 chunk", ha="left", va="center", fontsize=8.6, color=DARK)
ax.text(X0 - 9.6, Y_CACHE - 5.6, "写入时刻:观测(编码时一次) → 视频链最后一步 → 动作链最后一步;"
        "三者顺序决定了动作能看本 chunk 视频、视频看不到本 chunk 动作",
        ha="left", va="center", fontsize=8.4, color="#5d6d7e")

# ---------------------------------------------------------------- notes
notes = [
    (f"$\\sigma$: 1.00 → 0.00(每步用各自的 scheduler;demo 5/10 步,LIBERO 20/50 步)", 8.6, DARK),
    ("最后一步的 t=0 是 padding 出来的:只写 cache、不调用 scheduler.step  →  保证写进去的是干净 K/V", 8.6, DARK),
    ("两条链的噪声是两次独立 randn;CFG 也分开:视频 guidance_scale=5,动作 action_guidance_scale=1", 8.6, DARK),
    ("每个前向都把自己的 K/V 临时写进 cache 并立刻读回(update_cache=0 → 用完即删),所以本 chunk 内的 token 能互相看见", 8.6, DARK),
    ("cache 容量 = (attn_window//2) 个视频块 + (attn_window//2) 个动作块;满了就按 id 淘汰最旧(LIBERO attn_window=30)", 8.6, DARK),
    ("训练掩码与这里严格对应:noisy动作→clean视频(同 chunk)允许;noisy视频→clean动作(同 chunk)禁止", 8.6, "#7e5109"),
]
y = 11.4
for txt, fs, col in notes:
    ax.text(X0 - 9.6, y, "•  " + txt, fontsize=fs, color=col, va="center")
    y -= 2.6

ax.text(X0 - 9.6, 59.5, "两条扩散链的时序:视频先跑完,动作再跑 —— 通过共享 KV cache 单向耦合",
        fontsize=13, color=DARK, fontweight="bold")
ax.text(X0 - 9.6, 56.6, "一个 chunk(= frame_chunk_size 个 latent 帧 / 动作帧)内发生的事情;每个方块 = 一次 transformer 前向",
        fontsize=9.2, color="#7f8c8d")

fig.savefig(OUT, bbox_inches="tight")
print("wrote", OUT)
print(f"视频链 {len(vid)} 次前向(含写 cache 的那次),动作链 {len(act)} 次前向")
