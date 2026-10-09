"""What is inside the BlockMask, and what attention looks like in the two scenarios.

BlockMask is nothing but a boolean function
        mask_mod(b, h, q_idx, kv_idx) -> True/False
materialised as a 2-D image over the whole sequence.  In training the sequence holds
all four pieces, so the image has structure; in inference the sequence holds only the
current chunk and the "history" is the KV cache, so the image is a plain rectangle.
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

OUT = "/Users/kevin/code/lingbot-va/.claude/scratch/rf/story_blockmask.png"

# ---------------------------------------------------------------- rebuild the real mask
L_F, L_H, L_W, P = 4, 8, 16, (1, 2, 2)
A_F, A_H, A_W = 4, 4, 1
CHUNK, WINDOW = 2, 64                       # chunk_size / window_size (both are random per step)
PT = (L_H // P[1]) * (L_W // P[2])          # tokens per latent frame
LT = L_F * PT                               # 128
AT = A_F * A_H * A_W                        # 16

lat_fr = np.repeat(np.arange(L_F), PT)
act_fr = np.repeat(np.arange(A_F), A_H * A_W)
seg = np.concatenate([np.zeros(LT), np.ones(LT), 2 * np.ones(AT), 3 * np.ones(AT)]).astype(int)
noise = np.concatenate([np.zeros(LT), np.ones(LT), np.zeros(AT), np.ones(AT)]).astype(int)
frame = np.concatenate([lat_fr // CHUNK * 2] * 2 + [act_fr // CHUNK * 2 + 1] * 2)
N = len(seg)
q = np.arange(N)[:, None]
k = np.arange(N)[None, :]
m_cc = (noise[q] == 1) & (noise[k] == 1) & (frame[k] <= frame[q])
m_nc = (noise[q] == 0) & (noise[k] == 1) & (frame[k] < frame[q])
m_nn = (noise[q] == 0) & (noise[k] == 0) & (frame[k] == frame[q])
M = (m_cc | m_nc | m_nn) & (np.abs(frame[q] - frame[k]) <= WINDOW)
M = M.astype(float)

NAMES = ["noisy 视频\n(128)", "clean 视频\n(128)", "noisy 动作\n(16)", "clean 动作\n(16)"]
BOUND = [0, LT, 2 * LT, 2 * LT + AT, N]
COLS = ["#c0392b", "#e08a80", "#2471a3", "#8fb8d6"]
print(f"BlockMask: {N}×{N} 张量里 True 的比例 = {M.mean()*100:.1f}%"
      f"(chunk_size={CHUNK}, window_size={WINDOW})")
for a in range(4):
    row = "  ".join(f"{M[BOUND[a]:BOUND[a+1], BOUND[b]:BOUND[b+1]].mean()*100:5.1f}%" for b in range(4))
    print(f"  {NAMES[a].splitlines()[0]:12s} → [ {row} ]")

fig = plt.figure(figsize=(15.8, 8.4), dpi=150)
gs = fig.add_gridspec(1, 2, width_ratios=[0.92, 1.08], wspace=0.22, left=0.085, right=0.985,
                      top=0.84, bottom=0.235)

# ---------------------------------------------------------------- (左) 训练:真实的 BlockMask
ax = fig.add_subplot(gs[0, 0])
ax.imshow(M, cmap="Greys_r", origin="upper", interpolation="nearest", vmin=0, vmax=1)
for b in BOUND[1:-1]:
    ax.axhline(b - 0.5, color="#c0392b", lw=1.0, ls="--")
    ax.axvline(b - 0.5, color="#c0392b", lw=1.0, ls="--")
ax.set_xticks(BOUND[1:-1])
ax.set_xticklabels([f"{b}" for b in BOUND[1:-1]], fontsize=7.5)
ax.set_yticks([(BOUND[i] + BOUND[i + 1]) / 2 for i in range(4)])
ax.set_yticklabels(NAMES, fontsize=8.0)
ax.set_xlabel("键 kv_idx(被读);红线 = 段边界,段名见左轴", fontsize=9.5)
ax.set_ylabel("查询 q_idx(去读)", fontsize=10)
ax.set_title("训练:BlockMask = 一张 288×288 的黑白图\n白 = 允许看,黑 = 禁止(四段全在同一条序列里)",
             fontsize=11.5)

marks = [("①", 2, LT * 0.45, 2 * LT + AT * 0.5, "#1e8449"),
         ("②", 2, LT + PT * 0.6, PT * 0.6, "#111"),
         ("③", 2, PT * 0.6, PT * 0.6, "#111"),
         ("④", 2, 2 * LT + AT * 0.5, LT + 4, "#111"),
         ("⑤", 2, LT * 0.5, N - AT * 0.5, "#111")]
for mk, sz, x, y, col in marks:
    ax.text(x, y, mk, fontsize=13, color=col, ha="center", va="center",
            bbox=dict(fc="w", ec=col, lw=1.1, boxstyle="circle,pad=0.18"))


# ---------------------------------------------------------------- (右) 推理:一整块矩形
ax = fig.add_subplot(gs[0, 1])
groups = [("观测 chunk\nupdate_cache=2\n视频 128 + 动作 16", 144, "#1e8449"),
          ("更早 chunk\n(按 id 淘汰)\n视频 128 + 动作 16", 144, "#7f8c8d"),
          ("本步临时写入\nupdate_cache=0\n视频 128(或动作 16)", 128, "#c0392b")]
x = 0.0
for label, w, col in groups:
    ax.add_patch(Rectangle((x, 0), w - 6, 128, fc=col, ec="w", lw=1.0, alpha=0.72))
    ax.text(x + (w - 6) / 2, -46, label, ha="center", va="center", fontsize=8.4, color=col)
    ax.text(x + (w - 6) / 2, 64, "全可见", ha="center", va="center", fontsize=10, color="w")
    x += w
ax.add_patch(Rectangle((0, 0), x - 6, 128, fc="none", ec="#2c3e50", lw=2.2))
ax.text(x / 2, 100, "q = 当前 chunk 的 128 个视频 token(动作循环时是 16 个)",
        ha="center", fontsize=9.6, color="#2c3e50")
ax.set_xlim(-8, x + 6); ax.set_ylim(150, -70)
ax.set_yticks([]); ax.set_xticks([])
ax.set_xlabel("键 = cache 里所有有效槽位 + 本步临时写入的自己(没有 mask,一律可见)", fontsize=10)
ax.set_title("推理:没有 mask,注意力是一整块矩形\n可见性 = 「哪些槽位还有效」——未来根本不在 cache 里",
             fontsize=11.5)

fig.text(0.085, 0.145,
         "整张 BlockMask 的 True 只占 %.1f%%(chunk_size=%d, window_size=%d);" % (M.mean() * 100, CHUNK, WINDOW)
         + "窗口滑动的效果 = 再叠一条沿对角线的带(这里 window 够大,没截到)",
         ha="left", va="top", fontsize=9.2, color="#5d6d7e")
fig.text(0.085, 0.105,
         "① noisy 动作 → clean 视频(同 chunk):2c < 2c+1 ✓ 允许 —— 全图唯一一条同 chunk 的跨模态通路(动作以当前真值视频为条件)\n"
         "② noisy 视频 → clean 视频(同 chunk):2c < 2c ✗ 禁止 —— 这里是黑的:不能抄自己答案\n"
         "③ 同一段内部:沿对角线的小方块(同一个 chunk 的帧互相可见),跨 chunk 才是因果的\n"
         "④ clean 视频 → noisy 动作:clean ← noisy 一律禁止(干净 token 不能被脏 token 影响)\n"
         "⑤ 更早 chunk 的 clean 视频 / clean 动作:可见(替代推理时 cache 里的历史)",
         ha="left", va="top", fontsize=9.2, color="#34495e")
fig.suptitle("BlockMask 的内容,以及训练 / 推理两种场景下注意力实际在算什么", fontsize=13,
             y=0.965)
fig.savefig(OUT, bbox_inches="tight")
print("wrote", OUT)
