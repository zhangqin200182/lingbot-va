"""What the two chains are actually trained on.

Per training step each modality draws its OWN per-frame sigma from its OWN scheduler:
    train_scheduler_latent (shift = snr_shift)      -> video
    train_scheduler_action (shift = action_snr_shift) -> action
and multiplies its MSE by the scheduler's own BSMNTW loss weight (a bell in t, mean-normalised to 1).
This figure shows, for each of them: the sampled sigma density, the loss weight, and their product.
"""
import importlib.util
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
import torch

spec = importlib.util.spec_from_file_location(
    "sched", "/Users/kevin/code/lingbot-va/wan_va/utils/scheduler.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
FlowMatchScheduler = mod.FlowMatchScheduler

OUT = "/Users/kevin/code/lingbot-va/.claude/scratch/rf/story_train_sigma.png"
N = 1000
CASES = [("视频链  shift = snr_shift = 5.0", 5.0, "#c0392b"),
         ("动作链  shift = action_snr_shift = 0.05 (LIBERO)", 0.05, "#2471a3"),
         ("动作链  shift = action_snr_shift = 1.0 (demo)", 1.0, "#7fb3d5")]

torch.manual_seed(0)
u = torch.rand(400000)
tid = (u * N).clamp(0, N - 1).long()

fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4), dpi=150)
grid = torch.linspace(0.001, 1.0, 400)
for ax, (name, shift, col) in zip(axes, CASES):
    s = FlowMatchScheduler(shift=shift, sigma_min=0.0, extra_one_step=True)
    s.set_timesteps(N, training=True)
    sig = s.sigmas
    # sigma actually drawn (uniform over grid indices), and the weight attached to it
    drawn = sig[tid]
    # weight as a function of sigma: interpolate the bell onto the sigma axis
    order = sig.argsort()
    w_of_sig = s.linear_timesteps_weights[order]
    s_sorted = sig[order]
    h = torch.histc(drawn, bins=80, min=0, max=1) / len(drawn) * 80      # density (per unit sigma)
    ctr = (torch.arange(80) + 0.5) / 80
    ax.bar(ctr, h, width=1 / 80 * 0.9, color=col, alpha=0.55, label="抽到的 σ 密度")
    ax.set_xlabel("$\\sigma$  (噪声水平)", fontsize=10)
    ax.set_ylabel("密度", fontsize=9.5, color=col)
    ax.tick_params(labelsize=8)
    ax.set_xlim(0, 1)
    ax2 = ax.twinx()
    ax2.plot(grid, torch.from_numpy(
        __import__("numpy").interp(grid.numpy(), s_sorted.numpy(), w_of_sig.numpy())),
        color="#1e8449", lw=2.2, label="损失权重 (BSMNTW)")
    ax2.set_ylabel("损失权重", fontsize=9.5, color="#1e8449")
    ax2.tick_params(labelsize=8, colors="#1e8449")
    med = float(drawn.median())
    ax.axvline(med, color=col, ls="--", lw=1.4)
    ax.text(min(med + 0.02, 0.62), ax.get_ylim()[1] * 0.70, f"中位 σ={med:.3f}", fontsize=8.8, color=col)
    p_lo = float((drawn < 0.1).float().mean()) * 100
    p_hi = float((drawn > 0.7).float().mean()) * 100
    ax.text(0.50, ax.get_ylim()[1] * 0.40, f"P(σ<0.1) = {p_lo:.0f}%\nP(σ>0.7) = {p_hi:.0f}%",
            fontsize=8.6, color=col, ha="left")
    ax.set_title(name, fontsize=10)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.8, frameon=False, loc="upper center")
    print(f"{name:52s} 中位σ={med:.3f}  P(σ<0.1)={p_lo:5.1f}%  P(σ>0.7)={p_hi:5.1f}%")

fig.suptitle("两条链各自独立抽 σ:每个训练步、每一帧、每个模态都独立采样;损失权重也各自一套(BSMNTW,均值归一化到 1)",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(OUT, bbox_inches="tight")
print("\nwrote", OUT)
