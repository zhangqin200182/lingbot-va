const fs = require('fs');
const path = require('path');
require(path.join(__dirname, 'rf_engine.js'));
const ref = JSON.parse(fs.readFileSync(path.join(__dirname, 'ref_cases.json'), 'utf8'));
const R = globalThis.RF;
let worstRel = 0, rows = [];
for (const c of ref.cases) {
  let got;
  if (c.kind === 'enc_rf') got = R.rfAtStage(ref.ops_enc, c.si, c.anchor[0], c.anchor[1], ref.grid, ref.patch);
  else if (c.kind === 'dec_infl') got = R.influenceAt(ref.ops_dec, c.si, c.anchor[0], c.anchor[1], ref.image);
  else got = R.dependencyAt(ref.ops_dec, c.pixel[0], c.pixel[1]);
  const g = got.a;
  if (g.length !== c.vals.length) { rows.push([c.kind, 'LEN MISMATCH', g.length, c.vals.length]); continue; }
  let maxd = 0, mx = 0, dot = 0, na = 0, nb = 0;
  for (let i = 0; i < g.length; i++) {
    const d = Math.abs(g[i] - c.vals[i]);
    if (d > maxd) maxd = d;
    if (Math.abs(c.vals[i]) > mx) mx = Math.abs(c.vals[i]);
    dot += g[i] * c.vals[i]; na += g[i] * g[i]; nb += c.vals[i] * c.vals[i];
  }
  const rel = mx > 0 ? maxd / mx : 0;
  if (rel > worstRel) worstRel = rel;
  const tag = c.kind === 'dec_dep' ? 'px' + c.pixel : 'cell' + c.anchor + '#' + c.si;
  rows.push([c.kind.padEnd(8), tag.padEnd(14), 'rel_maxdiff=' + rel.toExponential(2),
             'cos=' + (dot / Math.sqrt(na * nb || 1)).toFixed(12)]);
}
rows.forEach(r => console.log(r.join('  ')));
console.log('\nWORST RELATIVE DIFF =', worstRel.toExponential(3), '(float64 round-off)');
