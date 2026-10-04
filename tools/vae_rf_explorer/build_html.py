"""Inline rf_engine.js + probe data into ui_template.html -> single-file deliverable."""
import json, os, pathlib

HERE = pathlib.Path(__file__).parent
OUT = HERE / "vae_rf_explorer.html"

ref = json.load(open(HERE / "ref_cases.json"))
probe = json.load(open(HERE / "probe_summary.json"))
data = dict(
    image=ref["image"], patch=ref["patch"], patch_grid=ref["patch_grid"], grid=ref["grid"],
    ops_enc=ref["ops_enc"], ops_dec=ref["ops_dec"],
    validation=probe["validation"], temporal=probe["temporal"],
    sim=probe["sim"], ring=probe["ring"], adjacent_sim=probe["adjacent_sim"],
    weights=probe.get("weights"),
)
tpl = (HERE / "ui_template.html").read_text()
engine = (HERE / "rf_engine.js").read_text()
html = tpl.replace("__DATA__", json.dumps(data, separators=(",", ":"))).replace("__ENGINE__", engine)
OUT.write_text(html)
print("wrote", OUT, f"{len(html)/1024:.0f} KB")
