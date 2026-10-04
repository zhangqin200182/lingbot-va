/* Weight-free receptive-field / influence geometry for the Wan2.2 VAE.
 *
 * Mirrors .claude/scratch/rf/probe_rf.py exactly.  All fields are Float64
 * row-major arrays wrapped as {h, w, a}; the same file runs in node (tests)
 * and in the browser (visualisation).
 */
(function (root) {
  'use strict';

  function F(h, w) { return { h: h, w: w, a: new Float64Array(h * w) }; }
  function at(f, i, j) { return f.a[i * f.w + j]; }
  function set(f, i, j, v) { f.a[i * f.w + j] = v; }
  function add(f, i, j, v) { f.a[i * f.w + j] += v; }
  function copy(f) { return { h: f.h, w: f.w, a: f.a.slice() }; }
  function scale(f, s) { const g = copy(f); for (let i = 0; i < g.a.length; i++) g.a[i] *= s; return g; }
  function onehot(h, w, pi, pj) { const f = F(h, w); set(f, pi, pj, 1); return f; }

  function convF(f, k, s, p, ho, wo) {
    const out = F(ho, wo);
    for (let i = 0; i < ho; i++) {
      for (let j = 0; j < wo; j++) {
        let acc = 0;
        for (let di = 0; di < k; di++) {
          const ii = i * s + di - p;
          if (ii < 0 || ii >= f.h) continue;
          for (let dj = 0; dj < k; dj++) {
            const jj = j * s + dj - p;
            if (jj < 0 || jj >= f.w) continue;
            acc += at(f, ii, jj);
          }
        }
        set(out, i, j, acc);
      }
    }
    return out;
  }

  function convT(f, k, s, p, hin, win) {
    const out = F(hin, win);
    for (let i = 0; i < f.h; i++) {
      for (let j = 0; j < f.w; j++) {
        const v = at(f, i, j);
        if (v === 0) continue;
        for (let di = 0; di < k; di++) {
          const ii = i * s + di - p;
          if (ii < 0 || ii >= hin) continue;
          for (let dj = 0; dj < k; dj++) {
            const jj = j * s + dj - p;
            if (jj < 0 || jj >= win) continue;
            add(out, ii, jj, v);
          }
        }
      }
    }
    return out;
  }

  function blockSum(f, s) {
    let h = f.h - (f.h % s), w = f.w - (f.w % s);
    if (h === 0 || w === 0) {
      const out = F(Math.max(h / s | 0, 1), Math.max(w / s | 0, 1));
      let tot = 0; for (let i = 0; i < f.a.length; i++) tot += f.a[i];
      out.a.fill(tot); return out;
    }
    const out = F(h / s, w / s);
    for (let i = 0; i < h; i++) for (let j = 0; j < w; j++)
      add(out, (i / s) | 0, (j / s) | 0, at(f, i, j));
    return out;
  }

  function blockRepeat(f, s) {
    const out = F(f.h * s, f.w * s);
    for (let i = 0; i < f.h; i++) for (let j = 0; j < f.w; j++) {
      const v = at(f, i, j);
      for (let di = 0; di < s; di++) for (let dj = 0; dj < s; dj++)
        set(out, i * s + di, j * s + dj, v);
    }
    return out;
  }

  function fit(f, h, w) {
    const out = F(h, w);
    const hh = Math.min(h, f.h), ww = Math.min(w, f.w);
    for (let i = 0; i < hh; i++) for (let j = 0; j < ww; j++) set(out, i, j, at(f, i, j));
    return out;
  }

  function resblockF(f, sc) {
    let g = convF(f, 3, 1, 1, f.h, f.w);
    g = convF(g, 3, 1, 1, g.h, g.w);
    const out = copy(g);
    for (let i = 0; i < out.a.length; i++) out.a[i] += sc * f.a[i];
    return out;
  }

  function resblockT(f, sc) {
    let g = copy(f);
    for (let i = 0; i < g.a.length; i++) g.a[i] *= (1 + sc);
    g = convT(g, 3, 1, 1, g.h, g.w);
    return convT(g, 3, 1, 1, g.h, g.w);
  }

  function zeropadConvT(f, hin, win) {
    const g = convT(f, 3, 2, 0, hin + 1, win + 1);
    return fit(g, hin, win);
  }

  function zeropadConvF(f, ho, wo) {
    const padded = F(f.h + 1, f.w + 1);
    for (let i = 0; i < f.h; i++) for (let j = 0; j < f.w; j++) set(padded, i, j, at(f, i, j));
    return convF(padded, 3, 2, 0, ho, wo);
  }

  function avgdownT(f, fs, weight, gin, win) {
    const out = F(gin, win);
    for (let oi = 0; oi < f.h; oi++) for (let oj = 0; oj < f.w; oj++) {
      const v = at(f, oi, oj) * weight;
      if (v === 0) continue;
      for (let di = 0; di < fs; di++) {
        const ii = oi * fs + di;
        if (ii >= gin) break;
        for (let dj = 0; dj < fs; dj++) {
          const jj = oj * fs + dj;
          if (jj < win) add(out, ii, jj, v);
        }
      }
    }
    return out;
  }

  function avgdownF(f, fs, weight, gh, gw) {
    const out = F(gh, gw);
    for (let i = 0; i < f.h; i++) for (let j = 0; j < f.w; j++) {
      const oi = (i / fs) | 0, oj = (j / fs) | 0;
      if (oi < gh && oj < gw) add(out, oi, oj, at(f, i, j) * weight);
    }
    return out;
  }

  function attnMix(f) {
    let tot = 0; for (let i = 0; i < f.a.length; i++) tot += f.a[i];
    const m = tot / f.a.length, out = copy(f);
    for (let i = 0; i < out.a.length; i++) out.a[i] += m;
    return out;
  }

  /* ---------------- op level ---------------- */
  function opForward(op, f) {
    const p = op.p, k = op.kind;
    if (k === 'conv') return convF(f, p.k, p.s, p.pad, op.gout[0], op.gout[1]);
    if (k === 'attn') return attnMix(f);
    if (k === 'res') return resblockF(f, p.scale === undefined ? 1 : p.scale);
    if (k === 'resup') {
      let out = f;
      const nres = p.nres;
      for (let r = 0; r < nres; r++) out = resblockF(out, p.scale === undefined ? 1 : p.scale);
      if (p.upsample) out = convF(blockRepeat(out, 2), 3, 1, 1, op.gout[0], op.gout[1]);
      if (p.shortcut) {
        const sc = fit(blockRepeat(f, p.fs), op.gout[0], op.gout[1]);
        out = fit(out, op.gout[0], op.gout[1]);
        for (let i = 0; i < out.a.length; i++) out.a[i] += sc.a[i];
      }
      return fit(out, op.gout[0], op.gout[1]);
    }
    if (k === 'resdown') {
      let out = f;
      for (let r = 0; r < p.nres; r++) out = resblockF(out, p.scale === undefined ? 1 : p.scale);
      if (p.downsample) out = zeropadConvF(out, op.gout[0], op.gout[1]);
      if (p.shortcut) {
        const sc = avgdownF(f, p.fs, p.weight, op.gout[0], op.gout[1]);
        out = fit(out, op.gout[0], op.gout[1]);
        for (let i = 0; i < out.a.length; i++) out.a[i] += sc.a[i];
      }
      return fit(out, op.gout[0], op.gout[1]);
    }
    throw new Error('bad kind ' + k);
  }

  function opBackward(op, f) {
    const p = op.p, k = op.kind;
    if (k === 'conv') return convT(f, p.k, p.s, p.pad, op.gin[0], op.gin[1]);
    if (k === 'attn') return attnMix(f);
    if (k === 'res') return resblockT(f, p.scale === undefined ? 1 : p.scale);
    if (k === 'resup') {
      let m = f;
      for (let r = 0; r < p.nres; r++) m = resblockT(m, p.scale === undefined ? 1 : p.scale);
      if (p.upsample) {
        m = convT(m, 3, 1, 1, op.gin[0] * 2, op.gin[1] * 2);
        m = blockSum(m, 2);
      }
      const out = fit(m, op.gin[0], op.gin[1]);
      if (p.shortcut) {
        const sc = blockSum(fit(f, op.gout[0], op.gout[1]), p.fs);
        const s2 = fit(sc, op.gin[0], op.gin[1]);
        for (let i = 0; i < out.a.length; i++) out.a[i] += s2.a[i];
      }
      return out;
    }
    if (k === 'resdown') {
      let m = f;
      if (p.downsample) m = zeropadConvT(m, m.h * 2, m.w * 2);
      for (let r = 0; r < p.nres; r++) m = resblockT(m, p.scale === undefined ? 1 : p.scale);
      const out = fit(m, op.gin[0], op.gin[1]);
      if (p.shortcut) {
        const sc = avgdownT(f, p.fs, p.weight, op.gin[0], op.gin[1]);
        for (let i = 0; i < out.a.length; i++) out.a[i] += sc.a[i];
      }
      return out;
    }
    throw new Error('bad kind ' + k);
  }

  /* ---------------- query helpers ---------------- */
  function stageTarget(grid, r, c, GRID) {
    const v0 = Math.floor((r + 0.5) * grid[0] / GRID - 0.5 + 0.5);
    const v1 = Math.floor((c + 0.5) * grid[1] / GRID - 0.5 + 0.5);
    return [Math.min(Math.max(v0, 0), grid[0] - 1), Math.min(Math.max(v1, 0), grid[1] - 1)];
  }

  /* encoder: RF (in image pixels) of the unit at the output of ops[si] */
  function rfAtStage(encOps, si, r, c, GRID, patch) {
    const op = encOps[si];
    const t = stageTarget(op.gout, r, c, GRID);
    let f = onehot(op.gout[0], op.gout[1], t[0], t[1]);
    for (let k = si; k >= 0; k--) f = opBackward(encOps[k], f);
    return blockRepeat(f, patch);            // patch grid -> image pixels
  }

  /* RF of the input pixel itself (baseline) */
  function rfOfPixel(r, c, GRID, patch) {
    return onehot(GRID * patch, GRID * patch, r * patch, c * patch);
  }

  function influenceAt(decOps, si, r, c, IMG) {
    let f = onehot(decOps[0].gin[0], decOps[0].gin[1], r, c);
    for (let k = 0; k <= si; k++) f = opForward(decOps[k], f);
    const g = decOps[si].gout[0];
    return blockRepeat(f, Math.round(IMG / g));
  }

  /* decoder: dependency of image pixel (pi,pj) on the latent grid */
  function dependencyAt(decOps, pi, pj) {
    const gout = decOps[decOps.length - 1].gout;
    let f = onehot(gout[0], gout[1], pi, pj);
    for (let k = decOps.length - 1; k >= 0; k--) f = opBackward(decOps[k], f);
    return f;
  }

  /* ---------------- statistics ---------------- */
  function total(f) { let t = 0; for (let i = 0; i < f.a.length; i++) t += f.a[i]; return t; }

  function summary(f) {
    const tot = total(f);
    if (tot <= 0) return { total: 0 };
    let peak = 0;
    for (let i = 0; i < f.a.length; i++) if (f.a[i] > peak) peak = f.a[i];
    let cy = 0, cx = 0;
    for (let i = 0; i < f.h; i++) for (let j = 0; j < f.w; j++) {
      const v = at(f, i, j); cy += v * i; cx += v * j;
    }
    cy /= tot; cx /= tot;
    let vy = 0, vx = 0;
    for (let i = 0; i < f.h; i++) for (let j = 0; j < f.w; j++) {
      const v = at(f, i, j);
      vy += v * (i - cy) * (i - cy); vx += v * (j - cx) * (j - cx);
    }
    vy /= tot; vx /= tot;
    const sigma = Math.sqrt((vy + vx) / 2);
    // radii by mass
    const dists = [], vals = [];
    for (let i = 0; i < f.h; i++) for (let j = 0; j < f.w; j++) {
      dists.push(Math.hypot(i - cy, j - cx)); vals.push(at(f, i, j));
    }
    const order = dists.map((d, i) => i).sort((a, b) => dists[a] - dists[b]);
    let cum = 0, r50 = 0, r90 = 0, r99 = 0;
    for (let n = 0; n < order.length; n++) {
      cum += vals[order[n]] / tot;
      if (r50 === 0 && cum >= 0.5) r50 = dists[order[n]];
      if (r90 === 0 && cum >= 0.9) r90 = dists[order[n]];
      if (r99 === 0 && cum >= 0.99) r99 = dists[order[n]];
    }
    let area = 0, minI = f.h, maxI = -1, minJ = f.w, maxJ = -1;
    const thr = peak * 1e-3;
    for (let i = 0; i < f.h; i++) for (let j = 0; j < f.w; j++) {
      if (at(f, i, j) > thr) {
        area++;
        if (i < minI) minI = i; if (i > maxI) maxI = i;
        if (j < minJ) minJ = j; if (j > maxJ) maxJ = j;
      }
    }
    return {
      total: tot, peak: peak, cy: cy, cx: cx, sigma: sigma,
      r50: r50, r90: r90, r99: r99,
      area: area, areaFrac: area / (f.h * f.w),
      extent: maxI < 0 ? [0, 0] : [maxI - minI + 1, maxJ - minJ + 1],
      centre: peak / tot * f.a.length,
    };
  }

  function cosine(a, b) {
    let dot = 0, na = 0, nb = 0;
    for (let i = 0; i < a.a.length; i++) {
      dot += a.a[i] * b.a[i]; na += a.a[i] * a.a[i]; nb += b.a[i] * b.a[i];
    }
    if (na === 0 || nb === 0) return 0;
    return dot / Math.sqrt(na * nb);
  }

  function radial(f, cy, cx, nbins, rmax) {
    const prof = new Float64Array(nbins), cnt = new Float64Array(nbins);
    rmax = rmax || Math.hypot(Math.max(cy, f.h - 1 - cy), Math.max(cx, f.w - 1 - cx));
    for (let i = 0; i < f.h; i++) for (let j = 0; j < f.w; j++) {
      const d = Math.hypot(i - cy, j - cx);
      let bin = Math.floor(d / rmax * nbins);
      if (bin >= nbins) bin = nbins - 1;
      prof[bin] += at(f, i, j); cnt[bin] += 1;
    }
    for (let b = 0; b < nbins; b++) if (cnt[b] > 0) prof[b] /= cnt[b];
    return prof;
  }

  function normalise(f) {
    let m = 0; for (let i = 0; i < f.a.length; i++) if (f.a[i] > m) m = f.a[i];
    const out = copy(f);
    if (m > 0) for (let i = 0; i < out.a.length; i++) out.a[i] /= m;
    return out;
  }

  root.RF = {
    F: F, at: at, set: set, copy: copy, onehot: onehot, scale: scale,
    convF: convF, convT: convT, blockSum: blockSum, blockRepeat: blockRepeat, fit: fit,
    resblockF: resblockF, resblockT: resblockT, zeropadConvF: zeropadConvF, zeropadConvT: zeropadConvT,
    avgdownF: avgdownF, avgdownT: avgdownT, attnMix: attnMix,
    opForward: opForward, opBackward: opBackward,
    stageTarget: stageTarget, rfAtStage: rfAtStage, rfOfPixel: rfOfPixel,
    influenceAt: influenceAt, dependencyAt: dependencyAt,
    summary: summary, cosine: cosine, radial: radial, normalise: normalise, total: total,
  };
})(typeof globalThis !== 'undefined' ? globalThis : this);
