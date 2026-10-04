"""Dump reference DP fields from the python implementation so the JS engine
can be checked against it bit-for-bit (well, to 1e-9)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import probe_rf as P

OUT = os.path.dirname(os.path.abspath(__file__))
enc, dec = P.build()
ops_enc = P.build_ops_enc(enc, P.PG)
ops_dec = P.build_ops_dec(dec, P.GRID)

cases = []
for (r, c) in [(4, 4), (0, 0), (6, 2)]:
    for si in [0, 2, 4, 5, 8]:
        f = P.rf_at_stage(ops_enc, si, r, c)
        cases.append(dict(kind="enc_rf", anchor=[r, c], si=si, shape=list(f.shape),
                          vals=[round(float(v), 9) for v in f.ravel()]))
for si in [0, 3, 4, 8]:
    f = P.influence_at_stage(ops_dec, si, 4, 4)
    f = P.block_repeat(f, P.IMG // ops_dec[si].gout[0])
    cases.append(dict(kind="dec_infl", anchor=[4, 4], si=si, shape=list(f.shape),
                      vals=[round(float(v), 9) for v in f.ravel()]))
for (pi, pj) in [(36, 36), (10, 50)]:
    f = P.dependency_at(ops_dec, (pi, pj)) if hasattr(P, "dependency_at") else None
    f = P.onehot((P.PG, P.PG), (pi, pj))
    for op in reversed(ops_dec):
        f = op.backward(f)
    cases.append(dict(kind="dec_dep", pixel=[pi, pj], shape=list(f.shape),
                      vals=[round(float(v), 9) for v in f.ravel()]))

ops = [dict(dir=o.__class__.__name__, name=o.name, kind=o.kind, gin=list(o.gin),
            gout=list(o.gout), p=o.p) for o in ops_enc]
ops += [dict(dir="dec", name=o.name, kind=o.kind, gin=list(o.gin), gout=list(o.gout), p=o.p)
        for o in ops_dec]
json.dump(dict(image=P.IMG, patch=P.PATCH, patch_grid=P.PG, grid=P.GRID,
               ops_enc=[o for o in ops[:len(ops_enc)]], ops_dec=[o for o in ops[len(ops_enc):]],
               cases=cases),
          open(os.path.join(OUT, "ref_cases.json"), "w"))
print("wrote ref_cases.json:", len(cases), "cases")
