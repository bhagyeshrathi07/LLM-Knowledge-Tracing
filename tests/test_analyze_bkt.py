"""The extra comparisons analyze_bkt.py reports beyond the item-mean ones."""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_bkt.py"
spec = importlib.util.spec_from_file_location("analyze_bkt", SCRIPT)
analyze_bkt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyze_bkt)


def test_extra_pairs_cover_fluid_control_and_no_recency():
    present = {"bkt_k0.1_r0.9_skill", "bkt_k1_r0.9_fluid_lo-2_hi1", "bkt_k0.1_r1_skill",
               "item_mean_ewma_r0.9", "item_mean"}
    pairs = analyze_bkt.extra_pairs("bkt_k0.1_r0.9_skill", "bkt_k1_r0.9_fluid_lo-2_hi1",
                                    "bkt_k0.1_r1_skill", "item_mean_ewma_r0.9", present)
    assert [(a, b) for a, b, _ in pairs] == [
        ("bkt_k1_r0.9_fluid_lo-2_hi1", "item_mean_ewma_r0.9"),
        ("bkt_k1_r0.9_fluid_lo-2_hi1", "bkt_k0.1_r0.9_skill"),
        ("bkt_k0.1_r1_skill", "item_mean"),
    ]


def test_extra_pairs_drop_absent_predictors():
    present = {"bkt_k0.1_r0.9_skill", "bkt_k1_r0.9_fluid_lo-2_hi1", "item_mean"}
    pairs = analyze_bkt.extra_pairs("bkt_k0.1_r0.9_skill", "bkt_k1_r0.9_fluid_lo-2_hi1",
                                    None, "item_mean_ewma_r0.9", present)
    assert [(a, b) for a, b, _ in pairs] == [("bkt_k1_r0.9_fluid_lo-2_hi1", "bkt_k0.1_r0.9_skill")]


def test_select_no_recency_picks_best_rho_one_skill_setting():
    import pandas as pd
    val = pd.DataFrame({
        "predictor": ["bkt_k0.1_r0.9_skill", "bkt_k1_r1_skill", "bkt_k0.1_r1_skill",
                      "bkt_k1e+06_r1_skill", "bkt_k1_r1_fluid_lo-1_hi1", "item_mean"],
        "log_loss": [0.32, 0.38, 0.37, 0.63, 0.36, 0.39],
    })
    assert analyze_bkt.select_no_recency(val) == "bkt_k0.1_r1_skill"


def _val_table():
    import pandas as pd
    rows = []
    for kappa in (0.1, 1, 10, 1e6):
        for rho in (1.0, 0.9, 0.7):
            if kappa >= 1e5 and rho != 1.0:
                continue
            rows.append((f"bkt_k{kappa:g}_r{rho:g}_skill", 0.4 + 0.01 * rho + 0.02 * (kappa > 1)))
            if kappa < 1e5:
                rows.append((f"bkt_k{kappa:g}_r{rho:g}_fluid_lo-2_hi1", 0.41 + 0.01 * rho))
    rows += [("item_mean", 0.386), ("item_mean_ewma_r0.9", 0.332), ("skill_mean", 0.63)]
    return pd.DataFrame(rows, columns=["predictor", "log_loss"]).assign(auc=0.9)


def test_plot_validation_grid_writes_a_png(tmp_path):
    out = tmp_path / "grid.png"
    analyze_bkt.plot_validation_grid(_val_table(), "bkt_k0.1_r0.9_skill", "bkt_k1_r0.9_fluid_lo-2_hi1", out)
    assert out.exists() and out.stat().st_size > 10_000


def test_plot_horizons_writes_a_png(tmp_path):
    import pandas as pd
    rows = []
    for h in (1, 5, 20):
        for name, ll in (("item_mean", 0.37 + 0.002 * h), ("item_mean_ewma_r0.9", 0.32 + 0.002 * h),
                         ("bkt_k0.1_r0.9_skill", 0.31 + 0.002 * h), ("bkt_k1_r0.9_fluid_lo-2_hi1", 0.316 + 0.002 * h)):
            rows.append({"h": h, "scope": "test", "predictor": name, "log_loss": ll, "log_loss_se": 0.02,
                         "auc": 0.92 - 0.001 * h, "auc_se": 0.006})
    out = tmp_path / "horizons.png"
    analyze_bkt.plot_horizons(pd.DataFrame(rows), "bkt_k0.1_r0.9_skill", "bkt_k1_r0.9_fluid_lo-2_hi1",
                              "item_mean_ewma_r0.9", out)
    assert out.exists() and out.stat().st_size > 10_000


def test_replot_mastery_redraws_from_stored_csvs(tmp_path):
    import numpy as np
    import pandas as pd
    skills = ["high_school_biology", "moral_scenarios", "professional_law", "anatomy"]
    mastery = pd.DataFrame([(s, t, float(t > i)) for i, s in enumerate(skills) for t in range(6)],
                           columns=["skill", "checkpoint_idx", "p_learned"])
    params = pd.DataFrame({"skill": skills, "n_items": 100, "pi": 0.1, "learn": np.linspace(0.05, 0.3, 4),
                           "forget": [0.01, 0.0002, 0.0002, 0.02], "guess_s": 0.3, "slip_s": 0.3, "n_iter": 3})
    for suffix in ("", "_k1e+06_r1"):
        mastery.to_csv(tmp_path / f"bkt_mastery_olmo2-7b{suffix}.csv", index=False)
        params.to_csv(tmp_path / f"bkt_skill_params_olmo2-7b{suffix}.csv", index=False)
    written = analyze_bkt.replot_mastery(tmp_path, tmp_path / "figs")
    assert [p.name for p in written] == ["bkt_mastery_olmo2-7b.png", "bkt_mastery_olmo2-7b_k1e+06_r1.png"]
    assert all(p.exists() and p.stat().st_size > 10_000 for p in written)
