"""Probe the Wan2.2 VAE spatial receptive-field / influence geometry.

Two independent models of the same object:

  A. autograd ground truth on a *randomly initialised* Wan2.2 VAE
     (exact ERF of that particular untrained network), plus a forward
     perturbation test for the decoder direction.

  B. a weight-free path-count linearisation ("geometry DP") built from the real
     module parameters (kernel / stride / padding / branch structure).  This is
     what the interactive visualisation runs in the browser.

Everything is spatial: T = 1 frame, 128x128 px -> patchify 2x2 -> 64x64x12 ->
encoder -> 8x8x48 latent -> decoder -> 64x64x3 -> unpatchify -> 128x128.
Temporal receptive field is reported separately (analytic, 1-D).
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import torch
from diffusers.models.autoencoders.autoencoder_kl_wan import (
    WanCausalConv3d,
    WanDecoder3d,
    WanEncoder3d,
    WanResidualBlock,
    WanResidualDownBlock,
    WanResidualUpBlock,
    patchify,
)

torch.manual_seed(0)
torch.set_num_threads(8)

IMG = 128
PATCH = 2
PG = IMG // PATCH          # 64 patchified grid
GRID = 8                   # latent grid
OUT = os.path.dirname(os.path.abspath(__file__))


def build(vae_path=None):
    """Random-init Wan2.2 VAE, or the encoder/decoder of a loaded checkpoint."""
    global PATCH, PG, GRID
    if vae_path:
        from diffusers import AutoencoderKLWan
        vae = AutoencoderKLWan.from_pretrained(vae_path, torch_dtype=torch.float32)
        PATCH = int(getattr(vae.config, "patch_size", None) or PATCH)
        PG = IMG // PATCH
        GRID = PG // 8
        assert PG % 8 == 0, PG
        print(f"loaded VAE from {vae_path}: patch_size={PATCH} latent grid={GRID}")
        return vae.encoder.eval(), vae.decoder.eval()
    enc = WanEncoder3d(
        in_channels=3 * PATCH * PATCH, dim=160, z_dim=96, dim_mult=[1, 2, 4, 4],
        num_res_blocks=2, attn_scales=[], temperal_downsample=[True, True, False],
        dropout=0.0, is_residual=True).eval()
    dec = WanDecoder3d(
        dim=256, z_dim=48, dim_mult=[1, 2, 4, 4], num_res_blocks=2, attn_scales=[],
        temperal_upsample=[False, True, True], dropout=0.0, out_channels=3,
        is_residual=True).eval()
    return enc, dec


# ==========================================================================
# geometry: field propagation primitives
# ==========================================================================
def conv_f(f, k, s, p):
    hin, win = f.shape
    ho = (hin + 2 * p - k) // s + 1
    wo = (win + 2 * p - k) // s + 1
    out = np.zeros((ho, wo))
    for i in range(ho):
        for j in range(wo):
            acc = 0.0
            for di in range(k):
                ii = i * s + di - p
                if 0 <= ii < hin:
                    for dj in range(k):
                        jj = j * s + dj - p
                        if 0 <= jj < win:
                            acc += f[ii, jj]
            out[i, j] = acc
    return out


def conv_t(f, k, s, p, hin, win):
    ho, wo = f.shape
    out = np.zeros((hin, win))
    for i in range(ho):
        for j in range(wo):
            v = f[i, j]
            if v == 0.0:
                continue
            for di in range(k):
                ii = i * s + di - p
                if 0 <= ii < hin:
                    for dj in range(k):
                        jj = j * s + dj - p
                        if 0 <= jj < win:
                            out[ii, jj] += v
    return out


def block_sum(f, s):
    """transpose of nearest x s upsampling = sum each s x s block."""
    h, w = f.shape
    h -= h % s
    w -= w % s
    if h == 0 or w == 0:
        return f.sum() * np.ones((max(h // s, 1), max(w // s, 1)))
    return f[:h, :w].reshape(h // s, s, w // s, s).sum(axis=(1, 3))


def block_repeat(f, s):
    return np.repeat(np.repeat(f, s, 0), s, 1)


def fit(f, g):
    out = np.zeros(g)
    h, w = min(g[0], f.shape[0]), min(g[1], f.shape[1])
    out[:h, :w] = f[:h, :w]
    return out


def resblock_t(f, scale=1.0):
    """transpose of a WanResidualBlock (two 3x3 convs + spatial residual)."""
    g = f + scale * f
    g = conv_t(g, 3, 1, 1, g.shape[0], g.shape[1])
    return conv_t(g, 3, 1, 1, g.shape[0], g.shape[1])


def resblock_f(f, scale=1.0):
    g = conv_f(f, 3, 1, 1)
    g = conv_f(g, 3, 1, 1)
    return g + scale * f


def zeropad_conv_t(f, hin, win):
    g = conv_t(f, 3, 2, 0, hin + 1, win + 1)
    return g[:hin, :win]


def zeropad_conv_f(f, ho, wo):
    padded = np.zeros((f.shape[0] + 1, f.shape[1] + 1))
    padded[: f.shape[0], : f.shape[1]] = f
    return conv_f(padded, 3, 2, 0)[:ho, :wo]


def avgdown_t(f, fs, weight, gin):
    out = np.zeros(gin)
    for oi in range(f.shape[0]):
        for oj in range(f.shape[1]):
            v = f[oi, oj] * weight
            if v == 0.0:
                continue
            for di in range(fs):
                ii = oi * fs + di
                if ii >= gin[0]:
                    break
                for dj in range(fs):
                    jj = oj * fs + dj
                    if jj < gin[1]:
                        out[ii, jj] += v
    return out


def avgdown_f(f, fs, weight, gout):
    out = np.zeros(gout)
    for i in range(f.shape[0]):
        for j in range(f.shape[1]):
            oi, oj = i // fs, j // fs
            if oi < gout[0] and oj < gout[1]:
                out[oi, oj] += f[i, j] * weight
    return out


def attn_mix(f):
    """uniform attention + residual: transpose == forward == f + mean(f)."""
    return f + f.mean()


# ==========================================================================
# op lists mirroring the real forward passes
# ==========================================================================
class Op:
    __slots__ = ("idx", "name", "kind", "gin", "gout", "stage", "p")

    def __init__(self, idx, name, kind, gin, gout, stage, **p):
        self.idx, self.name, self.kind, self.gin, self.gout = idx, name, kind, gin, gout
        self.stage, self.p = stage, p

    def forward(self, f):
        k, p = self.kind, self.p
        if k == "conv":
            return conv_f(f, p["k"], p["s"], p["pad"])
        if k == "attn":
            return attn_mix(f)
        if k == "res":
            return resblock_f(f, p.get("scale", 1.0))
        if k == "resup":
            out = f
            for _ in range(p["nres"]):
                out = resblock_f(out, p.get("scale", 1.0))
            if p["upsample"]:
                out = conv_f(block_repeat(out, 2), 3, 1, 1)
            if p["shortcut"]:
                sc = fit(block_repeat(f, p["fs"]), self.gout)
                out = fit(out, self.gout) + sc
            return fit(out, self.gout)
        if k == "resdown":
            out = f
            for _ in range(p["nres"]):
                out = resblock_f(out, p.get("scale", 1.0))
            if p["downsample"]:
                out = zeropad_conv_f(out, self.gout[0], self.gout[1])
            if p["shortcut"]:
                out = fit(out, self.gout) + avgdown_f(f, p["fs"], p["weight"], self.gout)
            return fit(out, self.gout)
        raise ValueError(k)

    def backward(self, f):
        k, p = self.kind, self.p
        if k == "conv":
            return conv_t(f, p["k"], p["s"], p["pad"], self.gin[0], self.gin[1])
        if k == "attn":
            return attn_mix(f)
        if k == "res":
            return resblock_t(f, p.get("scale", 1.0))
        if k == "resup":
            out = np.zeros(self.gin)
            # main arm: resnets (reverse order, same geometry) then upsampler
            m = f
            for _ in range(p["nres"]):
                m = resblock_t(m, p.get("scale", 1.0))
            if p["upsample"]:
                m = conv_t(m, 3, 1, 1, self.gin[0] * 2, self.gin[1] * 2)
                m = block_sum(m, 2)
            out += fit(m, self.gin)
            # shortcut arm: DupUp3D (spatial repeat -> block sum)
            if p["shortcut"]:
                sc = fit(f, self.gout)
                sc = block_sum(sc, p["fs"])
                out += fit(sc, self.gin)
            return out
        if k == "resdown":
            out = np.zeros(self.gin)
            # main arm: downsampler then resnets
            m = f
            if p["downsample"]:
                m = zeropad_conv_t(m, m.shape[0] * 2, m.shape[1] * 2)
            for _ in range(p["nres"]):
                m = resblock_t(m, p.get("scale", 1.0))
            out += fit(m, self.gin)
            # shortcut arm: AvgDown3D
            if p["shortcut"]:
                out += avgdown_t(f, p["fs"], p["weight"], self.gin)
            return out
        raise ValueError(k)


def build_ops_enc(enc, gin_img):
    ops, g, idx = [], (gin_img, gin_img), 0
    ops.append(Op(idx, "conv_in", "conv", g, g, "conv_in", k=3, s=1, pad=1)); idx += 1
    for i, blk in enumerate(enc.down_blocks):
        assert isinstance(blk, WanResidualDownBlock)
        ds = blk.downsampler
        gout = (g[0] // 2, g[1] // 2) if ds is not None else g
        fs = 2 if ds is not None else 1
        ac = blk.avg_shortcut
        weight = 1.0 / ac.group_size
        ops.append(Op(idx, f"down{i}", "resdown", g, gout, f"down{i}",
                      nres=len(blk.resnets), downsample=ds is not None, shortcut=True,
                      fs=fs, weight=weight, scale=1.0)); idx += 1
        g = gout
    ops.append(Op(idx, "mid_res0", "res", g, g, "mid_res0", scale=1.0)); idx += 1
    ops.append(Op(idx, "mid_attn", "attn", g, g, "mid_attn")); idx += 1
    ops.append(Op(idx, "mid_res1", "res", g, g, "mid_res1", scale=1.0)); idx += 1
    ops.append(Op(idx, "conv_out", "conv", g, g, "conv_out", k=3, s=1, pad=1))
    return ops


def build_ops_dec(dec, gin_latent):
    ops, g, idx = [], (gin_latent, gin_latent), 0
    ops.append(Op(idx, "conv_in", "conv", g, g, "conv_in", k=3, s=1, pad=1)); idx += 1
    ops.append(Op(idx, "mid_res0", "res", g, g, "mid_res0", scale=1.0)); idx += 1
    ops.append(Op(idx, "mid_attn", "attn", g, g, "mid_attn")); idx += 1
    ops.append(Op(idx, "mid_res1", "res", g, g, "mid_res1", scale=1.0)); idx += 1
    for i, blk in enumerate(dec.up_blocks):
        assert isinstance(blk, WanResidualUpBlock)
        up = blk.upsampler is not None
        gout = (g[0] * 2, g[1] * 2) if up else g
        ac = blk.avg_shortcut
        fs = ac.factor_s if ac is not None else 1
        ops.append(Op(idx, f"up{i}", "resup", g, gout, f"up{i}", nres=len(blk.resnets),
                      upsample=up, shortcut=ac is not None, fs=fs, scale=1.0)); idx += 1
        g = gout
    ops.append(Op(idx, "conv_out", "conv", g, g, "conv_out", k=3, s=1, pad=1))
    return ops


# ==========================================================================
# helpers
# ==========================================================================
def field2d(t):
    return t.detach().abs().sum(dim=1)[0, 0].numpy().astype(np.float64)


def expand_patch(f):
    return block_repeat(f, PATCH)


def stage_target(field_grid, grid, r, c):
    """position in `grid` that corresponds to latent cell (r,c).

    floor(v + 0.5) (not round) so that the JS engine in rf_engine.js agrees
    bit-for-bit with this reference implementation.
    """
    ri = int(math.floor((r + 0.5) * grid[0] / GRID - 0.5 + 0.5))
    ci = int(math.floor((c + 0.5) * grid[1] / GRID - 0.5 + 0.5))
    return min(max(ri, 0), grid[0] - 1), min(max(ci, 0), grid[1] - 1)


def onehot(grid, pos):
    f = np.zeros(grid)
    f[pos] = 1.0
    return f


def rf_at_stage(ops, si, r, c):
    """input-space (128x128) RF of the unit at the output of ops[si] near latent (r,c)."""
    f = onehot(ops[si].gout, stage_target(None, ops[si].gout, r, c))
    for k in range(si, -1, -1):
        f = ops[k].backward(f)
    return expand_patch(f)          # patch grid -> image px


ENC_STAGE_MODULES = {}


def stage_modules(enc, dec):
    """map op name -> module whose forward output is the op output."""
    m = {
        "conv_in": enc.conv_in, "mid_res0": enc.mid_block.resnets[0],
        "mid_attn": enc.mid_block.attentions[0], "mid_res1": enc.mid_block.resnets[1],
        "conv_out": enc.conv_out,
    }
    for i, blk in enumerate(enc.down_blocks):
        m[f"down{i}"] = blk
    for i, blk in enumerate(dec.up_blocks):
        m[f"up{i}"] = blk
    return m


def autograd_rf_stage(enc, mod, grid, r, c, x_img):
    """true RF (in 128x128 input px) of one unit at `mod`'s output, random weights."""
    store = []
    h = mod.register_forward_hook(lambda m, i, o: store.append(o))
    xp = patchify(x_img, PATCH).requires_grad_(True)
    z = enc(xp)
    h.remove()
    pi, pj = stage_target(None, grid, r, c)
    g, = torch.autograd.grad(store[0][0, :, 0, pi, pj].sum(), xp)
    return expand_patch(field2d(g))


def influence_at_stage(ops, si, r, c):
    """output-space field of latent cell (r,c) after ops[0..si] (decoder direction)."""
    f = onehot(ops[0].gin, (r, c))
    for k in range(0, si + 1):
        f = ops[k].forward(f)
    return f


def summary(f, label=""):
    tot = float(f.sum())
    if tot <= 0:
        return dict(label=label, total=0.0)
    fy, fx = np.nonzero(f)
    ii = np.arange(f.shape[0])[:, None]
    jj = np.arange(f.shape[1])[None, :]
    cy = float((f * ii).sum() / tot)
    cx = float((f * jj).sum() / tot)
    vy = float((f * (ii - cy) ** 2).sum() / tot)
    vx = float((f * (jj - cx) ** 2).sum() / tot)
    sig = math.sqrt((vy + vx) / 2)
    dist = np.sqrt((ii - cy) ** 2 + (jj - cx) ** 2)
    order = np.argsort(dist.ravel())
    cum = np.cumsum(f.ravel()[order]) / tot
    r90 = float(dist.ravel()[order][np.searchsorted(cum, 0.90)])
    r50 = float(dist.ravel()[order][np.searchsorted(cum, 0.50)])
    peak = float(f.max())
    return dict(label=label, total=tot, peak=peak,
                extent=[int(fy.max() - fy.min() + 1), int(fx.max() - fx.min() + 1)],
                area1e3=int((f > peak * 1e-3).sum()),
                area_frac=float((f > peak * 1e-3).mean()),
                cy=cy, cx=cx, sigma=sig, r50=r50, r90=r90,
                peak_frac=float(peak / tot * f.size))


def cosine(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if na == 0 or nb == 0 else float(a @ b / (na * nb))


def radial_profile(f, center, nbins=40, rmax=None):
    ii = np.arange(f.shape[0])[:, None]
    jj = np.arange(f.shape[1])[None, :]
    d = np.sqrt((ii - center[0]) ** 2 + (jj - center[1]) ** 2)
    rmax = rmax or d.max()
    edges = np.linspace(0, rmax, nbins + 1)
    prof = np.zeros(nbins)
    cnt = np.zeros(nbins)
    idx = np.clip(np.digitize(d.ravel(), edges) - 1, 0, nbins - 1)
    np.add.at(prof, idx, f.ravel())
    np.add.at(cnt, idx, 1.0)
    prof = np.where(cnt > 0, prof / np.maximum(cnt, 1), 0.0)
    return prof.tolist(), edges.tolist()


# ==========================================================================
def main(vae_path=None):
    enc, dec = build(vae_path)
    ZDIM = int(dec.conv_in.in_channels)   # latent channels of this VAE (48 for Wan2.2)
    print('latent channels:', ZDIM)
    ops_enc = build_ops_enc(enc, PG)
    ops_dec = build_ops_dec(dec, GRID)
    print("encoder ops:", [(o.name, o.gin, o.gout) for o in ops_enc])
    print("decoder ops:", [(o.name, o.gin, o.gout) for o in ops_dec])

    # ---------------- autograd ground truth: encoder RF ----------------
    store = []
    hooks = [
        enc.conv_in.register_forward_hook(lambda m, i, o: store.append(("conv_in", o))),
        enc.mid_block.resnets[0].register_forward_hook(lambda m, i, o: store.append(("mid_res0", o))),
        enc.mid_block.attentions[0].register_forward_hook(lambda m, i, o: store.append(("mid_attn", o))),
        enc.mid_block.resnets[1].register_forward_hook(lambda m, i, o: store.append(("mid_res1", o))),
        enc.conv_out.register_forward_hook(lambda m, i, o: store.append(("conv_out", o))),
    ]
    for i, blk in enumerate(enc.down_blocks):
        hooks.append(blk.register_forward_hook(
            (lambda i: lambda m, inp, o: store.append((f"down{i}", o)))(i)))

    x_img = torch.randn(1, 3, 1, IMG, IMG)
    xp = patchify(x_img, PATCH).requires_grad_(True)
    z96 = enc(xp)
    grads = torch.autograd.grad(z96[0, :, 0, GRID // 2, GRID // 2].sum(),
                               [t for _, t in store] + [xp])
    for h in hooks:
        h.remove()
    enc_true = {name: field2d(g) for (name, _), g in zip(store, grads[:-1])}
    enc_true_input = expand_patch(field2d(grads[-1]))

    # perturbation cross-check: bump the whole latent cell
    xp2 = xp.detach().clone()
    z96b = enc(xp2)
    d = (z96 - z96b)
    print("latent cell perturbation self-check |d|:", float(d.abs().max()))

    dp_input = rf_at_stage(ops_enc, len(ops_enc) - 1, GRID // 2, GRID // 2)
    sa, sb = summary(enc_true_input, "autograd"), summary(dp_input, "geometry")
    print()
    print("=" * 78)
    print("ENCODER RF of latent cell (4,4) -> 128x128 input pixels")
    print("=" * 78)
    for s in (sa, sb):
        print(f"  {s['label']:9s} extent={s['extent']} area(>0.1%peak)={s['area_frac']:.1%} "
              f"sigma={s['sigma']:.2f}px r50={s['r50']:.1f} r90={s['r90']:.1f} "
              f"centre=({s['cy']:.1f},{s['cx']:.1f})")
    print(f"  cosine(autograd, DP) = {cosine(enc_true_input, dp_input):.4f}")

    prof_true, edges = radial_profile(enc_true_input, (sa["cy"], sa["cx"]), 20, 64.0)
    prof_dp, _ = radial_profile(dp_input, (sb["cy"], sb["cx"]), 20, 64.0)

    # ---------------- RF growth with depth: DP vs autograd ----------------
    mods = stage_modules(enc, dec)
    enc_rows = []
    for si, op in enumerate(ops_enc):
        dp = rf_at_stage(ops_enc, si, GRID // 2, GRID // 2)
        au = autograd_rf_stage(enc, mods[op.name], op.gout, GRID // 2, GRID // 2, x_img)
        sdp, sau = summary(dp), summary(au)
        enc_rows.append(dict(idx=si, name=op.name, grid=list(op.gout),
                             jump=IMG // op.gout[0],
                             dp=sdp, autograd=sau, cosine=cosine(dp, au)))
        print(f"  {op.name:9s} grid={op.gout[0]:2d} jump={IMG // op.gout[0]:2d}px  "
              f"DP sigma={sdp['sigma']:6.2f} r50={sdp['r50']:5.1f} r90={sdp['r90']:5.1f} "
              f"area={sdp['area_frac']:5.1%} | autograd sigma={sau['sigma']:6.2f} "
              f"r50={sau['r50']:5.1f} r90={sau['r90']:5.1f} cos={cosine(dp, au):.3f}")

    dec_rows = []
    for si, op in enumerate(ops_dec):
        dp = influence_at_stage(ops_dec, si, GRID // 2, GRID // 2)
        # expand to image px for comparability
        dpi = block_repeat(dp, IMG // op.gout[0])   # -> 128x128 image px
        sdp = summary(dpi)
        dec_rows.append(dict(idx=si, name=op.name, grid=list(op.gout),
                             jump=IMG // op.gout[0], dp=sdp))
        print(f"  dec {op.name:9s} grid={op.gout[0]:2d} jump={IMG // op.gout[0]:3d}px  "
              f"DP extent={sdp['extent']} sigma={sdp['sigma']:6.2f} r90={sdp['r90']:5.1f} "
              f"area={sdp['area_frac']:5.1%}")

    # ---------------- decoder ----------------
    def run_dec(z):
        return dec(z, feat_cache=[None] * 34, feat_idx=[0], first_chunk=True)

    z = torch.randn(1, ZDIM, 1, GRID, GRID)
    with torch.no_grad():
        base = run_dec(z)
        z2 = z.clone()
        z2[0, :, 0, GRID // 2, GRID // 2] += 1.0
        pert = run_dec(z2)
    dec_true = expand_patch(field2d(base - pert))
    dp_dec = expand_patch(influence_at_stage(ops_dec, len(ops_dec) - 1, GRID // 2, GRID // 2))
    sc, sd = summary(dec_true, "perturb"), summary(dp_dec, "geometry")
    print()
    print("=" * 78)
    print("DECODER influence of latent cell (4,4) on the 128x128 output")
    print("=" * 78)
    for s in (sc, sd):
        print(f"  {s['label']:9s} extent={s['extent']} area={s['area_frac']:.1%} "
              f"sigma={s['sigma']:.2f}px r50={s['r50']:.1f} r90={s['r90']:.1f}")
    print(f"  cosine(perturbation, DP) = {cosine(dec_true, dp_dec):.4f}")

    # decoder dependency: one output pixel <- latent
    py, px = 72, 72            # image pixel
    fc = [None] * 34
    z3 = torch.randn(1, ZDIM, 1, GRID, GRID, requires_grad=True)
    out = dec(z3, feat_cache=fc, feat_idx=[0], first_chunk=True)
    g, = torch.autograd.grad(out[0, :, 0, py // PATCH, px // PATCH].sum(), z3)
    dep_true = field2d(g)
    f = onehot((PG, PG), (py // PATCH, px // PATCH))
    for op in reversed(ops_dec):
        f = op.backward(f)
    dep_dp = f
    print(f"  dependency: pixel({py},{px}) <- latent  cosine={cosine(dep_true, dep_dp):.4f} "
          f"nonzero_true={(dep_true > dep_true.max() * 1e-3).sum()}/{GRID * GRID} "
          f"nonzero_dp={(dep_dp > dep_dp.max() * 1e-3).sum()}/{GRID * GRID}")

    # ---------------- neighbour similarity ----------------
    fields = {}
    for r in range(GRID):
        for c in range(GRID):
            fields[(r, c)] = rf_at_stage(ops_enc, len(ops_enc) - 1, r, c)
    anchor = fields[(GRID // 2, GRID // 2)]
    sim = np.zeros((GRID, GRID))
    for r in range(GRID):
        for c in range(GRID):
            sim[r, c] = cosine(anchor, fields[(r, c)])
    print()
    print("cosine similarity of the input-space RF: anchor cell (4,4) vs every latent cell")
    for r in range(GRID):
        print("   " + " ".join(f"{sim[r, c]:.3f}" for c in range(GRID)))
    ring = {}
    for r in range(GRID):
        for c in range(GRID):
            d = max(abs(r - GRID // 2), abs(c - GRID // 2))
            ring.setdefault(d, []).append(sim[r, c])
    ring_mean = {int(k): float(np.mean(v)) for k, v in sorted(ring.items())}
    print("  Chebyshev ring -> mean similarity:", {k: round(v, 4) for k, v in ring_mean.items()})
    print("  adjacent (orthogonal neighbour) similarity:",
          round(cosine(anchor, fields[(GRID // 2, GRID // 2 + 1)]), 4))

    # difference field (what one latent cell knows that its neighbour does not)
    diff = anchor - fields[(GRID // 2, GRID // 2 + 1)]
    diff_pos = np.clip(diff, 0, None)
    ds = summary(diff_pos, "anchor-minus-right-neighbour")
    print("  unique (positive) part of the anchor, extent:", ds["extent"], "sigma",
          round(ds["sigma"], 2))

    # ---------------- temporal RF (analytic) ----------------
    plan = [("conv_in", 3, 1)]
    for i, blk in enumerate(enc.down_blocks):
        for r in range(len(blk.resnets) * 2):
            plan.append((f"down{i}.res", 3, 1))
        if blk.downsampler is not None and blk.downsampler.mode == "downsample3d":
            plan.append((f"down{i}.ds", 3, 2))
    plan += [("mid_res0", 3, 1), ("mid_res1", 3, 1), ("conv_out", 3, 1)]
    t_rf, t_jump, curve = 1.0, 1.0, []
    for name, k, s in plan:
        t_rf += (k - 1) * t_jump
        t_jump *= s
        curve.append([name, t_rf, t_jump])
    print(f"\ntemporal RF of one latent frame: {t_rf:.0f} input frames (stride {t_jump:.0f})")

    # ---------------- export ----------------
    ops_export = []
    for o in ops_enc:
        ops_export.append(dict(dir="enc", idx=o.idx, name=o.name, kind=o.kind,
                               gin=list(o.gin), gout=list(o.gout), p=o.p))
    for o in ops_dec:
        ops_export.append(dict(dir="dec", idx=o.idx, name=o.name, kind=o.kind,
                               gin=list(o.gin), gout=list(o.gout), p=o.p))
    out = dict(
        image=IMG, patch=PATCH, patch_grid=PG, grid=GRID, z_dim=48,
        ops=ops_export,
        validation=dict(
            cosine_enc=cosine(enc_true_input, dp_input),
            cosine_dec=cosine(dec_true, dp_dec),
            cosine_dep=cosine(dep_true, dep_dp),
            enc_autograd=sa, enc_dp=sb, dec_perturb=sc, dec_dp=sd,
            enc_rows=enc_rows, dec_rows=dec_rows,
            radial=dict(edges=edges, rmax=64.0, autograd=prof_true, geometry=prof_dp),
        ),
        weights=(vae_path or f"random-init (torch seed {torch.initial_seed()})"),
        sim=sim.tolist(), ring=ring_mean,
        adjacent_sim=cosine(anchor, fields[(GRID // 2, GRID // 2 + 1)]),
        temporal=dict(rf=t_rf, stride=t_jump, curve=curve),
    )
    with open(os.path.join(OUT, "probe_summary.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", os.path.join(OUT, "probe_summary.json"))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vae-path", default=None,
                    help="directory of a Wan VAE in AutoencoderKLWan layout (e.g. <model>/vae). "
                         "Without it the script uses randomly initialised weights.")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    main(vae_path=a.vae_path)
