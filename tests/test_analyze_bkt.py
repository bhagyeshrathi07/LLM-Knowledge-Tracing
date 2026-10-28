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
