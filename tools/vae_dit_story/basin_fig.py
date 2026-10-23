r"""ε 决定结局;c 移动盆地边界。
   ODE:x₀ = F_c(ε) ⇒ H(x₀|ε,c)=0,熵只有"ε 抽样"一个来源
   "锁定"= ε 落进哪个盆地;盆地是【场】的性质 ⇒ c 移动的就是这条边界
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
import matplotlib.pyplot as plt, numpy as np
RED,BLUE,GREEN,DARK,GREY,ORANGE,PURPLE=("#c0392b","#2471a3","#1e8449","#2c3e50","#95a5a6","#b9770e","#7d3c98")
RF="/Users/kevin/code/lingbot-va/.claude/scratch/rf/"
MU=np.array([1.0,-1.0])
def make_field(w):
    w=np.asarray(w,float)
    def post(x,t):
        lw=np.log(w)[None,:]-0.5*np.log(2*np.pi*t**2)-(x[:,None]-(1-t)*MU[None,:])**2/(2*t**2)
        lw-=lw.max(1,keepdims=True); p=np.exp(lw); return p/p.sum(1,keepdims=True)
    return lambda x,t:(x-(post(x,t)@MU))/max(t,1e-6)
def endpoint(vf,e,N=400):
    SIG=np.linspace(1,0,N+1); x=float(e)
    for i in range(N): x=x+vf(np.array([x]),SIG[i])[0]*(SIG[i+1]-SIG[i])
    return x
def boundary(w,lo=-8,hi=8):
    vf=make_field(w)
    for _ in range(50):
        mid=(lo+hi)/2
        if endpoint(vf,mid)>0: hi=mid
        else: lo=mid
    return (lo+hi)/2

fig=plt.figure(figsize=(15.2,5.5),dpi=150)
# (a) 阶梯图:终点 vs ε
ax=fig.add_axes([0.055,0.16,0.34,0.68])
eps=np.linspace(-4,4,161)
for w,col,lab in [((0.5,0.5),BLUE,"无条件(0.5/0.5)"),((0.9,0.1),ORANGE,"c 强(0.9/0.1)"),((0.1,0.9),GREEN,"c 反(0.1/0.9)")]:
    vf=make_field(w); ys=[endpoint(vf,e) for e in eps]
    ax.step(eps,ys,where="mid",color=col,lw=2.4,label=lab)
    b=boundary(w); ax.axvline(b,color=col,ls=":",lw=1.4)
    ax.plot([b],[-1.0],marker="v",ms=8,color=col)
ax.set_xlabel("初始噪声 $\\epsilon$(同一次抽样的位置)",fontsize=10.4)
ax.set_ylabel("终点 $x_0$",fontsize=10.4)
ax.set_title("(a) 同一个 $\\epsilon$ 轴,三个条件:\n盆地边界被推走了",fontsize=11.2)
ax.legend(fontsize=8.6,frameon=False,loc="center right")
ax.tick_params(labelsize=9); ax.set_xlim(-4,4); ax.set_ylim(-1.25,1.25)
ax.text(-3.9,-1.18,"▼ = 盆地边界 $\\epsilon^*$",fontsize=8.4,color=GREY)
# (b) 边界 vs 条件强度
ax2=fig.add_axes([0.45,0.16,0.24,0.68])
ws=np.linspace(0.5,0.995,40); bs=[boundary((w,1-w)) for w in ws]
ax2.plot(ws,bs,color=PURPLE,lw=2.8)
for w,lab in [(0.5,"0.5/0.5\n$\\epsilon^*=0$"),(0.7,"0.7/0.3\n−0.52"),(0.9,"0.9/0.1\n−1.28"),(0.99,"0.99/0.01\n−2.33")]:
    b=boundary((w,1-w)); ax2.plot([w],[b],"o",ms=7,color=RED)
    ax2.annotate(lab,xy=(w,b),xytext=(w+0.012,b+0.28),fontsize=7.6,color=RED)
ax2.set_xlabel("条件强度:权重 $w_{+1}$",fontsize=10.2)
ax2.set_ylabel("盆地边界 $\\epsilon^*$",fontsize=10.2)
ax2.set_title("(b) 条件越强,边界越往\n劣势模式推",fontsize=11.0)
ax2.tick_params(labelsize=9)
# (c) 结论
ax3=fig.add_axes([0.72,0.10,0.27,0.80]); ax3.axis("off")
ax3.text(0.5,1.00,"(c) 所以怎么理解\"ε 决定\"",ha="center",va="top",fontsize=11.4,color=DARK)
ax3.text(0.0,0.92,"$x_0=F_c(\\epsilon)$ ⇒ $H(x_0|\\epsilon,c)=0$\n"
         "$H(x_0|c)=H(p_{\\rm data}(\\cdot|c))$\n\n"
         "⇒ 熵只有【ε 抽样】一个来源",
         fontsize=9.0,color=BLUE,va="top",bbox=dict(fc="#eaf2fa",ec=BLUE))
ax3.text(0.0,0.70,"但要区分两种\"决定\":\n\n"
         "① 因果上:✓ ε 抽到瞬间结局已定\n"
         "② \"ε 里编码着构图\":✗\n"
         "　 ε 是【种子】,不是编码;\n"
         "　 构图在轨迹中被【解出来】",
         fontsize=8.6,color=DARK,va="top",bbox=dict(fc="#fffbe6",ec="#e0c060"))
ax3.text(0.0,0.40,"而 $c$ 的作用精确地说:\n"
         "　【移动盆地边界】\n\n"
         "$0.5/0.5$ → $\\epsilon^*=0$\n"
         "$0.9/0.1$ → $\\epsilon^*=-1.28$\n\n"
         "⇒ 同一个 ε 可落到相反模式\n"
         "　(ε=+0.15:±1 都出现过)",
         fontsize=8.6,color=GREEN,va="top",bbox=dict(fc="#eef7f1",ec=GREEN))
ax3.text(0.0,0.05,"SDE 下失效:每步噪声可推过边界\n⇒ 熵的第二个来源回来了\n(这正是 GRPO 要 SDE 的原因)",
         fontsize=8.2,color=RED,va="bottom",bbox=dict(fc="#fdecea",ec="#e6a9a0"))
fig.suptitle("ε 一抽完结局就定了(ODE);而\"锁定在高噪声\"是说:锁定 = ε 落进哪个盆地,盆地的边界是【场】的性质,所以 c 移动的就是它",
             fontsize=10.8,y=0.99)
fig.savefig(RF+"fg_basin_boundary.png",bbox_inches="tight"); print("wrote",RF+"fg_basin_boundary.png")
