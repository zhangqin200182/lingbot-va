"""Segment-level visibility between the four pieces of the training sequence.

    hidden = cat([noisy视频, clean视频, noisy动作, clean动作])

Rules copied verbatim from FlexAttnFunc._get_mask_mod:
    clean -> clean : frame[kv] <= frame[q]           (and noise_ids == 1 both)
    noisy -> clean : frame[kv] <  frame[q]
    noisy -> noisy : frame[kv] == frame[q]
    ... all ANDed with the sliding window |frame[q]-frame[kv]| <= window_size
with frame_ids = [latent_frame//chunk*2]*2 + [action_frame//chunk*2 + 1]*2
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
from matplotlib.patches import Rectangle

OUT = "/Users/kevin/code/lingbot-va/.claude/scratch/rf/story_clean_segments.png"

SEGS = ["noisy 视频", "clean 视频", "noisy 动作", "clean 动作"]
NOISE = [0, 1, 0, 1]          # noise_ids


def frame_of(seg, chunk):
    """frame_id of segment `seg` when it belongs to chunk index `chunk`"""
    return chunk * 2 + (1 if seg in (2, 3) else 0)


def visible(q_seg, kv_seg, q_chunk, kv_chunk, window=64):
    fq, fk = frame_of(q_seg, q_chunk), frame_of(kv_seg, kv_chunk)
    nq, nk = NOISE[q_seg], NOISE[kv_seg]
    if nq == 1 and nk == 1:
        ok = fk <= fq
    elif nq == 0 and nk == 1:
        ok = fk < fq
    elif nq == 0 and nk == 0:
        ok = fk == fq
    else:
        ok = False
    return ok and abs(fq - fk) <= window


def why(q_seg, kv_seg, q_chunk, kv_chunk):
    fq, fk = frame_of(q_seg, q_chunk), frame_of(kv_seg, kv_chunk)
    nq, nk = NOISE[q_seg], NOISE[kv_seg]
    if nq == 1 and nk == 1:
        return f"{fk} ≤ {fq}" + ("  ✓" if fk <= fq else "  ✗")
    if nq == 0 and nk == 1:
        return f"{fk} < {fq}" + ("  ✓" if fk < fq else "  ✗")
    if nq == 0 and nk == 0:
        return f"{fk} = {fq}" + ("  ✓" if fk == fq else "  ✗")
    return "clean ← noisy  ✗"


fig = plt.figure(figsize=(14.2, 6.4), dpi=150)
gs = fig.add_gridspec(1, 2, wspace=0.16, left=0.115, right=0.985, top=0.80, bottom=0.20)

for col, (kv_chunk_rel, title) in enumerate([(0, "键来自【同一个 chunk】"),
                                             (-1, "键来自【上一个 chunk】")]):
    ax = fig.add_subplot(gs[0, col])
    ax.set_xlim(-0.5, 3.5); ax.set_ylim(3.5, -1.35); ax.axis("off")
    for i, s in enumerate(SEGS):
        ax.text(-0.62, i, s, ha="right", va="center", fontsize=10.5,
                color="#c0392b" if i in (0, 1) else "#2471a3")
    for j, s in enumerate(SEGS):
        ax.text(j, -0.75, s, ha="center", va="center", fontsize=10.5,
                color="#c0392b" if j in (0, 1) else "#2471a3")
    ax.text(-0.62, -0.75, "查询 ↓ / 键 →", ha="right", va="center", fontsize=9, color="#7f8c8d")
    for i in range(4):
        for j in range(4):
            ok = visible(i, j, 1, 1 + kv_chunk_rel)
            txt = why(i, j, 1, 1 + kv_chunk_rel)
            ax.add_patch(Rectangle((j - 0.47, i - 0.42), 0.94, 0.84,
                                   fc="#e8f6ef" if ok else "#fdecea",
                                   ec="#1e8449" if ok else "#c0392b", lw=1.4))
            ax.text(j, i - 0.10, "✓ 可见" if ok else "✗ 不可见", ha="center", va="center",
                    fontsize=10, color="#1e8449" if ok else "#c0392b", fontweight="bold")
            ax.text(j, i + 0.24, txt, ha="center", va="center", fontsize=7.4, color="#5d6d7e")
    ax.set_title(title, fontsize=11.5)
    if col == 0:
        ax.text(-0.62, -1.22, "行 = 查询(第 c 个 chunk),列 = 被读的键", fontsize=9, color="#7f8c8d")

fig.suptitle("clean 段的作用:它们是「已知上下文」token,由掩码决定谁能读谁\n"
             "关键差异来源:视频 frame_id 为偶数 2c、动作为奇数 2c+1",
             fontsize=12)
fig.text(0.5, 0.045,
         "① noisy 动作 能读【同 chunk 的 clean 视频】(2c < 2c+1) → 动作以当前真值视频为条件  "
         "② noisy 视频 读不到【同 chunk 的 clean 视频】(2c < 2c 不成立)→ 防止抄答案\n"
         "③ 两条链都只能读【更早 chunk】的对方模态 → clean 段同时充当历史(替代推理时的 KV cache)  "
         "④ clean→clean 全通(因果),clean 段自己的信息也能流动",
         ha="center", fontsize=9.4, color="#34495e")
fig.savefig(OUT, bbox_inches="tight")

print("段级可见性(查询 = 第 c 个 chunk 的 noisy 段):")
for i, s in enumerate(SEGS):
    row = "  ".join(f"{'✓' if visible(i,j,1,1) else '✗'}({SEGS[j][:5]},同c)" for j in range(4))
    print(f"  {s:10s} → {row}")
print("wrote", OUT)
