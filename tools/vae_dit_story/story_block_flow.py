"""Inside one WanTransformerBlock: the same block, two very different scenarios.

Training   (train.py: load_transformer(..., attn_mode="flex"))
    ONE sequence holding all four pieces -> BlockMask decides who may read whom.
Inference  (wan_va_server.py: attn_mode="torch")
    only the current chunk of ONE modality per call -> a KV cache holds the history,
    no mask at all; visibility = "which cache slots are still occupied".
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
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

OUT = "/Users/kevin/code/lingbot-va/.claude/scratch/rf/story_block_flow.png"
RED, BLUE, GREEN, DARK, GREY, ORANGE = "#c0392b", "#2471a3", "#1e8449", "#2c3e50", "#95a5a6", "#b9770e"

fig, ax = plt.subplots(figsize=(15.4, 9.0), dpi=150)
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")


def box(x, y, w, h, fc, ec, lw=1.3, r=0.9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.3,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw))


def arrow(x1, y1, x2, y2, color=DARK, lw=1.6, style="-|>", ms=13, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=ms,
                                 lw=lw, color=color, linestyle=ls))


def block_box(x, y, w, h, tag, color):
    box(x, y, w, h, "#fbfcfd", color, lw=1.8)
    ax.text(x + w / 2, y + h - 1.6, tag, ha="center", va="center", fontsize=11.5,
            color=color, fontweight="bold")
    steps = [("① norm1 + AdaLN 调制(每个 token 用自己的 t)", DARK),
             ("② attn1  自注意力  + RoPE", RED),
             ("③ attn2  对文本的交叉注意力", DARK),
             ("④ FFN + 残差", DARK)]
    yy = y + h - 5.4
    for s, c in steps:
        ax.text(x + 2.0, yy, s, ha="left", va="center", fontsize=9.4, color=c)
        yy -= 3.4


# ============================================================ 训练
ax.text(2, 96.5, "训练:一次前向,一条序列装下全部四段", fontsize=13.5, color=DARK, fontweight="bold")
ax.text(2, 93.2, "train.py → load_transformer(..., attn_mode=\"flex\")  ⇒  attn1 走 FlexAttnFunc + BlockMask",
        fontsize=9.6, color=GREY)

seq = [("noisy 视频\n128 token\nt = 逐帧随机", RED, 0.94), ("clean 视频\n128\n t=0 或 0.5~1", RED, 0.55),
       ("noisy 动作\n16\nt = 逐帧随机", BLUE, 0.94), ("clean 动作\n16\nt=0", BLUE, 0.55)]
x = 2.5
for label, col, alpha in seq:
    box(x, 79.0, 12.0, 10.0, col, "w", lw=1.0)
    ax.text(x + 6.0, 84.0, label, ha="center", va="center", fontsize=9.0, color="w")
    x += 12.6
ax.text(2.5, 76.4, "hidden_states = cat([...], dim=1)   →  共 288 个 token,padding 到 384",
        fontsize=9.2, color=DARK)
ax.text(2.5, 74.2, "full_grid_id = cat([视频 grid]*2 + [动作 grid]*2)  →  clean 段与 noisy 段共享同一位置",
        fontsize=9.2, color=DARK)
ax.text(2.5, 72.0, "init_mask(...)  用 chunk_size / window_size 造出 BlockMask(noisy↔clean↔noisy 三条规则)",
        fontsize=9.2, color=ORANGE)

arrow(53.5, 84.0, 57.5, 84.0, DARK, 1.8, ms=15)
block_box(58.0, 68.0, 30.0, 21.0, "WanTransformerBlock  ×30", DARK)

box(90.5, 79.0, 8.0, 10.0, "#eef3f7", DARK)
ax.text(94.5, 84.0, "取 noisy\n两段的\n输出", ha="center", va="center", fontsize=9.0, color=DARK)
arrow(88.3, 84.0, 90.2, 84.0, DARK, 1.6, ms=13)

box(2.5, 61.0, 47.0, 5.6, "#fdecea", RED, lw=1.2)
ax.text(26.0, 63.8, "attn1:query = 384 个 token,k/v = 同一个序列(被 BlockMask 过滤)",
        ha="center", va="center", fontsize=9.8, color=RED)
ax.text(2.5, 58.2, "没有 cache(attn_caches 为空 ⇒ kv_cache = None);可见性 100% 由掩码决定",
        fontsize=9.2, color=GREY)
arrow(26.0, 61.0, 26.0, 68.0 - 0.0, RED, 1.4, ms=12, ls="--")

box(52.5, 61.0, 45.5, 5.6, "#eef7f1", GREEN, lw=1.2)
ax.text(75.2, 63.8, "loss = latent_loss(noisy视频) + action_loss(noisy动作)  →  一次 backward",
        ha="center", va="center", fontsize=9.8, color=GREEN)

# ============================================================ 推理
ax.plot([1, 99], [54.5, 54.5], color="#dfe6ec", lw=1.4)
ax.text(2, 52.0, "推理:每一步、每个模态各一次前向;序列里只有当前 chunk,历史全在 cache 里",
        fontsize=13.5, color=DARK, fontweight="bold")
ax.text(2, 49.2, "wan_va_server.py → attn_mode=\"torch\"  ⇒  attn1 走 custom_sdpa,不带任何掩码",
        fontsize=9.6, color=GREY)

ax.text(3.0, 46.2, "视频循环的第 i 步(action_mode=False)", fontsize=9.4, color=RED)
ax.text(3.0, 36.0, "动作循环的第 i 步(action_mode=True)", fontsize=9.4, color=BLUE)
box(3.0, 39.4, 26.0, 6.2, RED, "w", lw=1.0)
ax.text(16.0, 42.5, "当前 chunk 的视频 128 token\n(第 0 帧换成观测、t=0)", ha="center", va="center",
        fontsize=8.8, color="w")
box(3.0, 29.2, 26.0, 6.6, BLUE, "w", lw=1.0)
ax.text(16.0, 32.5, "当前 chunk 的动作 16 token\n(第 0 帧置 0、t=0)", ha="center", va="center",
        fontsize=8.8, color="w")
arrow(29.5, 42.5, 33.0, 42.5, DARK, 1.8, ms=14)
arrow(29.5, 32.5, 33.0, 32.5, DARK, 1.8, ms=14)

block_box(33.5, 25.0, 27.0, 21.0, "同一个 WanTransformerBlock ×30", DARK)

# cache panel
box(64.0, 20.0, 34.0, 27.0, "#f7f9fb", GREY, lw=1.2)
ax.text(81.0, 45.4, "KV cache(每个 block 一份)", ha="center", fontsize=10.4, color=DARK,
        fontweight="bold")
items = [("观测 K/V(update_cache=2)", GREEN),
         ("更早 chunk 的视频 K/V", GREY),
         ("更早 chunk 的动作 K/V", GREY),
         ("…按 id 淘汰最旧的…", "#dfe6ec"),
         ("本 chunk 视频 K/V(末步 update_cache=1)", RED),
         ("本 chunk 动作 K/V(末步 update_cache=1)", BLUE)]
yy = 42.0
for label, col in items:
    box(65.0, yy - 1.5, 32.0, 3.0, col, "w", lw=0.8)
    ax.text(81.0, yy, label, ha="center", va="center", fontsize=8.6,
            color=DARK if col == "#dfe6ec" else "w")
    yy -= 3.7
arrow(60.8, 40.0, 63.6, 36.0, DARK, 1.5, ms=12)
arrow(63.6, 32.0, 60.8, 35.0, DARK, 1.5, ms=12)
ax.text(62.0, 43.6, "读 k/v\n= 有效槽位", ha="center", fontsize=8.6, color=DARK)

box(2.5, 18.5, 59.0, 5.6, "#fdecea", RED, lw=1.2)
ax.text(32.0, 21.3, "attn1:query = 当前 chunk;k/v = cache 里所有有效槽位(含刚写入的自己)",
        ha="center", va="center", fontsize=9.8, color=RED)
ax.text(2.5, 15.6, "没有 mask(attn_mode=\"torch\");可见性由「哪些槽位还有效」决定 —— 未来根本不在 cache 里",
        fontsize=9.2, color=GREY)
ax.text(2.5, 12.6, "update_cache=0:临时写入 → 参与注意力 → restore 删掉   |   =1:留下(供后续 chunk 读)   "
                   "|   =2:留下且标记 is_pred=False(观测)", fontsize=9.2, color=ORANGE)

box(2.5, 4.5, 95.0, 6.4, "#eef7f1", GREEN, lw=1.2)
ax.text(50.0, 7.7, "输出只取当前模态:proj_out → 反 patchify → scheduler.step   或   "
                   "action_proj_out → action_scheduler.step(两步都用各自的 σ 网格)",
        ha="center", va="center", fontsize=9.8, color=GREEN)
fig.savefig(OUT, bbox_inches="tight")
print("wrote", OUT)
