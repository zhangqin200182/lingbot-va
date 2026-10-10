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
    "IMG_fg_baseline_removes":   IMG / "fg_baseline_removes.png",
    "IMG_fg_compensation_coef":   IMG / "fg_compensation_coef.png",
    "IMG_fg_compensation_ledger":   IMG / "fg_compensation_ledger.png",
    "IMG_fg_continuity_meaning":   IMG / "fg_continuity_meaning.png",
    "IMG_fg_equivalent_sde_family":   IMG / "fg_equivalent_sde_family.png",
    "IMG_fg_flux_bridge":   IMG / "fg_flux_bridge.png",
    "IMG_fg_flux_total":   IMG / "fg_flux_total.png",
    "IMG_fg_flux_why":   IMG / "fg_flux_why.png",
    "IMG_fg_fp_cancellation":   IMG / "fg_fp_cancellation.png",
    "IMG_fg_grpo_computation":   IMG / "fg_grpo_computation.png",
    "IMG_fg_grpo_what_changes":   IMG / "fg_grpo_what_changes.png",
    "IMG_fg_logprob_and_loss":   IMG / "fg_logprob_and_loss.png",
    "IMG_fg_marginal_bayes":   IMG / "fg_marginal_bayes.png",
    "IMG_fg_mean_vs_mode":   IMG / "fg_mean_vs_mode.png",
    "IMG_fg_no_wiggle":   IMG / "fg_no_wiggle.png",
    "IMG_fg_ode_distribution":   IMG / "fg_ode_distribution.png",
    "IMG_fg_ode_vs_sde_basic":   IMG / "fg_ode_vs_sde_basic.png",
    "IMG_fg_p_is_probability":   IMG / "fg_p_is_probability.png",
    "IMG_fg_rl_pushes_mean":   IMG / "fg_rl_pushes_mean.png",
    "IMG_fg_rl_vs_flow":   IMG / "fg_rl_vs_flow.png",
    "IMG_fg_shift_invariance":   IMG / "fg_shift_invariance.png",
    "IMG_fg_tweedie_two_identities":   IMG / "fg_tweedie_two_identities.png",
    "IMG_fg_two_cancellations":   IMG / "fg_two_cancellations.png",
    "IMG_fg_v_vs_drift_noise":   IMG / "fg_v_vs_drift_noise.png",
    "IMG_fg_v_vs_score":   IMG / "fg_v_vs_score.png",
    "IMG_fg_vstar_step":   IMG / "fg_vstar_step.png",
    "IMG_fg_vstar_table":   IMG / "fg_vstar_table.png",
    "IMG_fg_reward_terminal": IMG / "fg_reward_terminal.png",
    "IMG_fg_reward_path_vs_field": IMG / "fg_reward_path_vs_field.png",
    "IMG_fg_more_steps": IMG / "fg_more_steps.png",
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
