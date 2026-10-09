"""Build the self-contained story page.

Reads  story_template.html  +  every figure in  docs/  +  the existing VAE explorer,
base64-embeds the images and inlines the explorer as a srcdoc iframe, then writes
vae_dit_story.html (a single file that works offline).

Run from this directory:   python build_story.py
"""
import base64
import io
import pathlib

from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
IMG = HERE / "docs"
EXPLORER = HERE.parent / "vae_rf_explorer" / "vae_rf_explorer.html"
OUT = HERE / "vae_dit_story.html"

SLOTS = {
    # ---- act 1: VAE
    "IMG_rf":         IMG / "story_rf_growth.png",
    "IMG_chan":       IMG / "per_channel_mu_sigma.png",
    "IMG_rfmain":     IMG / "preview-main.png",
    # ---- act 2: DiT
    "IMG_counting":   IMG / "bayes_counting_pic.png",
    "IMG_bayesinner": IMG / "bayes_inner.png",
    "IMG_average":    IMG / "why_average.png",
    "IMG_gravity":    IMG / "story_gravity.png",
    "IMG_gcontour":   IMG / "story_gravity_contour.png",
    "IMG_glens":      IMG / "story_gravity_lens.png",
    "IMG_gfield":     IMG / "story_density_field.png",
    "IMG_traj":       IMG / "traj_vs_dist.png",
    "IMG_5steps":     IMG / "step_by_step.png",
    "IMG_bell":       IMG / "bell_to_two_spikes.png",
    "IMG_arb":        IMG / "arbitrary_shapes.png",
    "IMG_eff":        IMG / "story_eff_sources.png",
    "IMG_basins":     IMG / "basins.png",
    "IMG_multi":      IMG / "multimodal.png",
    # ---- act 3: conditioning
    "IMG_cond":       IMG / "conditioning.png",
    "IMG_samexs":     IMG / "story_same_xs_diff_c.png",
    "IMG_mask":       IMG / "story_mask.png",
    # ---- act 4: the two chains
    "IMG_twochains":  IMG / "story_two_chains.png",
    "IMG_wammap":     IMG / "story_wam_map.png",
    "IMG_wamevidence": IMG / "story_wam_evidence.png",
    "IMG_tfields":    IMG / "story_two_fields.png",
    "IMG_lensrand":   IMG / "story_lens_random.png",
}

MAXW = 1250
# smooth / colourful figures: JPEG is much smaller and looks the same on screen
JPEG = {"story_gravity_contour.png", "story_two_fields.png", "story_gravity_lens.png", "story_density_field.png",
        "traj_vs_dist.png", "basins.png", "conditioning.png", "arbitrary_shapes.png",
        "bell_to_two_spikes.png", "multimodal.png"}


def embed(path: pathlib.Path, maxw: int = MAXW) -> str:
    im = Image.open(path).convert("RGB")
    if im.width > maxw:
        h = round(im.height * maxw / im.width)
        im = im.resize((maxw, h), Image.LANCZOS)
    buf = io.BytesIO()
    if path.name in JPEG:
        im.save(buf, format="JPEG", quality=92, optimize=True, subsampling=1)
        kind = "jpeg"
    else:
        im.save(buf, format="PNG", optimize=True)
        kind = "png"
    b = buf.getvalue()
    print(f"  {path.name:28s} {im.width}x{im.height}  {len(b)/1024:7.1f} KB  [{kind}]")
    return f"data:image/{kind};base64," + base64.b64encode(b).decode()


def main():
    html = (HERE / "story_template.html").read_text()

    # ---- inline the self-contained VAE explorer through srcdoc (keeps this page single-file)
    raw = EXPLORER.read_text()
    esc = raw.replace("&", "&amp;").replace('"', "&quot;")
    iframe = (f'<iframe srcdoc="{esc}" title="Wan2.2 VAE 感受野 / 影响域浏览器" '
              f'loading="lazy" referrerpolicy="no-referrer"></iframe>')
    html = html.replace("{{IFRAME_RF_EXPLORER}}", iframe)
    print(f"  embedded explorer: {len(raw)/1024:.0f} KB -> escaped {len(esc)/1024:.0f} KB")

    # ---- embed every figure
    for slot, path in SLOTS.items():
        if not path.exists():
            raise SystemExit(f"missing: {path}")
        html = html.replace("{{" + slot + "}}", embed(path))

    if "{{" in html:
        raise SystemExit("unreplaced placeholder left in the template")

    OUT.write_text(html)
    print(f"\nwrote {OUT}")
    print(f"  size: {len(html)/1024/1024:.2f} MB")


if __name__ == "__main__":
    main()
