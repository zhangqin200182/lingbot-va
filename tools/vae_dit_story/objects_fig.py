r"""一个点"看到"的到底是什么?四个对象别混。

  先验    p_data(x₀)            数据分布(复杂)
  似然    p(x_t|x₀) = N(x_t;(1−t)x₀,t²I)   ← 高斯"几何"
  后验    p(x₀|x_t) ∝ 先验 × 似然           ← 【决定方向】(加权平均的权重)
  边缘    p_t(x_t) = ∫先验×似然 dx₀          ← 【这个点上的总密度】(你之前说的"叠加")

  常见写错:p(x₀)·p(x₀|x_t)  ✗ —— 先验被算了两次,而且后验里已经含先验
  正确:    p(x₀|x_t) ∝ p_data(x₀)·p(x_t|x₀)
"""
import matplotlib; matplotlib.use("Agg")
from matplotlib import font_manager
for _f in ["/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
           "/System/Library/Fonts/STHeiti Medium.ttc"]:
    try:
        font_manager.fontManager.addfont(_f)
        matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=_f).get_name(); break
    except Exception: continue
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
RED,BLUE,GREEN,DARK,GREY,ORANGE,PURPLE=("#c0392b","#2471a3","#1e8449","#2c3e50","#95a5a6","#b9770e","#7d3c98")
RF="/Users/kevin/code/lingbot-va/.claude/scratch/rf/"
def box(ax,x,y,w,h,title,body,fc,ec,fs=8.4):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.012",fc=fc,ec=ec,lw=1.6))
    ax.text(x+w/2,y+h-0.055,title,ha="center",va="top",fontsize=9.4,color=ec)
    ax.text(x+w/2,y+h-0.135,body,ha="center",va="top",fontsize=fs,color=DARK)

fig=plt.figure(figsize=(15.2,5.7),dpi=150)
# (a) 四个对象
ax=fig.add_axes([0.02,0.05,0.47,0.88]); ax.axis("off")
ax.text(0.5,1.00,"(a) 四个对象,各是什么",ha="center",va="top",fontsize=12,color=DARK)
box(ax,0.02,0.66,0.45,0.26,"① 先验  $p_{\\rm data}(x_0)$",
    "数据分布本身(复杂、学到的)\n= 候选的\"底色\"\n→ 也就是 $p_{\\rm data}(\\cdot|c)$","#f7f9fb",GREY)
box(ax,0.51,0.66,0.47,0.26,"② 似然  $p(x_t|x_0)$",
    "$=N(x_t;(1-t)x_0,\\,t^2I)$  ← 高斯!\n= 几何:\"这个候选离 $x_t$ 多近\"\n【与 $c$ 无关】","#eaf2fa",BLUE)
box(ax,0.02,0.34,0.45,0.26,"③ 后验  $p(x_0|x_t)$",
    "$\\propto$ ① × ②(归一化到 $x_0$ 上)\n= 一个【关于 $x_0$ 的函数】\n→ 这就是点\"看到\"的分配","#eef7f1",GREEN)
box(ax,0.51,0.34,0.47,0.26,"④ 边缘  $p_t(x_t)$",
    "$=\\int p_{\\rm data}(x_0)\\,N(x_t;(1-t)x_0,t^2I)\\,dx_0$\n= 这个【点上的总密度】\n← 你之前说的\"叠加\"","#fffbe6",ORANGE)
ax.add_patch(FancyArrowPatch((0.25,0.655),(0.25,0.615),arrowstyle="-|>",mutation_scale=11,lw=1.6,color=DARK))
ax.add_patch(FancyArrowPatch((0.75,0.655),(0.75,0.615),arrowstyle="-|>",mutation_scale=11,lw=1.6,color=DARK))
ax.text(0.25,0.305,"×  并归一化  →  【决定方向】",fontsize=9.0,color=GREEN,ha="center")
ax.text(0.75,0.305,"对 $x_0$ 积分  →  【决定 score】",fontsize=9.0,color=ORANGE,ha="center")
ax.text(0.5,0.18,"同一个被积函数  $p_{\\rm data}(x_0)\\cdot N(\\cdot)$,\n两种用法:\n"
        "· 对 $x_0$ 归一化 → 得到候选之间的分配 → 方向\n"
        "· 对 $x_0$ 积分 → 得到点上的密度 → score",
        fontsize=8.8,color=DARK,ha="center",va="top",bbox=dict(fc="#f7f9fb",ec="#c8d4e0"))
ax.set_xlim(0,1); ax.set_ylim(0.02,1.02)
# (b) 你写的两个候选
ax2=fig.add_axes([0.52,0.05,0.46,0.88]); ax2.axis("off")
ax2.text(0.5,1.00,"(b) 你的两个说法,哪个对",ha="center",va="top",fontsize=12,color=DARK)
ax2.text(0.0,0.92,"说法 A:\"是 $p(x_0|x_t)$ 的值\"",fontsize=10.0,color=GREEN,va="top")
ax2.text(0.03,0.855,"✓ 对。这就是点看到的【分配】——\n"
         "　 对每个候选 $x_0^{(i)}$ 给一个数,\n"
         "　 数之间的【比例】决定权重。",
         fontsize=8.8,color=DARK,va="top",bbox=dict(fc="#eef7f1",ec=GREEN))
ax2.text(0.0,0.68,"说法 B:\"是 $p(x_0)\\cdot p(x_0|x_t)$ 的值\"",fontsize=10.0,color=RED,va="top")
ax2.text(0.03,0.615,"✗ 不对,两个错:\n"
         "　 ① 先验被算了两遍:后验里已含 $p(x_0)$\n"
         "　 ② 乘完得到 $p(x_0)^2p(x_t|x_0)/p_t(x_t)$,\n"
         "　　　这不是任何标准分布\n\n"
         "　 你想说的应该是:【先验 × 似然】\n"
         "　 $p(x_0|x_t)\\ \\propto\\ p_{\\rm data}(x_0)\\cdot p(x_t|x_0)$\n"
         "　 　 　 　 　 　 ↑①　　　↑②(高斯)",
         fontsize=8.8,color=DARK,va="top",bbox=dict(fc="#fdecea",ec="#e6a9a0"))
ax2.text(0.03,0.30,"验证(数值,固定 $x_t=0.1,t=0.5$,两原子 $x_0=\\pm1$):\n"
         "　先验 0.5/0.5,高斯(按比例) 0.5794/0.3884\n"
         "　先验×似然 = 0.2897 / 0.1942 → 归一化 → 0.5987 / 0.4013\n"
         "　这就是 $p(x_0|x_t)$;若乘两遍先验:\n"
         "　0.25×0.5794 / 0.25×0.3884 → 0.5987 / 0.4013\n"
         "　(这里先验相等,恰好看不出差别;先验不等时就会歪)",
         fontsize=8.2,color=DARK,va="top",bbox=dict(fc="#fffbe6",ec="#e0c060"))
ax2.set_xlim(0,1); ax2.set_ylim(0.02,1.02)
fig.suptitle("一个点\"看到\"的是【后验】$p(x_0|x_t)$(∝ 先验×似然);而\"点上的总密度\"是【边缘】$p_t(x_t)$ —— 两者别混",
             fontsize=11.0,y=0.99)
fig.savefig(RF+"fg_four_objects.png",bbox_inches="tight"); print("wrote",RF+"fg_four_objects.png")
