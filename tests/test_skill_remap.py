import numpy as np
import pandas as pd
import pytest

from llm_kt.baselines import SkillMean
from llm_kt.evaluate import TARGET_COLUMNS, evaluate
from llm_kt.metrics import auc, log_loss
from llm_kt.skill_remap import RemappedSkills, constant_skill, shuffled_skills


def items_by_subject():
    return {
        "astro": [f"a{i}" for i in range(5)],
        "bio": [f"b{i}" for i in range(3)],
        "chem": [f"c{i}" for i in range(7)],
        "law": [f"l{i}" for i in range(1)],
    }


class TestMappings:
    def test_constant(self):
        m = constant_skill(["x", "y", "x"])
        assert m == {"x": "all", "y": "all"}

    def test_shuffle_is_a_size_preserving_bijection(self):
        subj = items_by_subject()
        m = shuffled_skills(subj, seed=0)
        items = [i for v in subj.values() for i in v]
        assert sorted(m) == sorted(items)              # every item mapped exactly once
        sizes = pd.Series(m).value_counts().to_dict()
        assert sizes == {s: len(v) for s, v in subj.items()}
        # the labels are a permutation of the original labels across items
        orig = sorted(s for s, v in subj.items() for _ in v)
        assert sorted(m.values()) == orig

    def test_actually_moves_items(self):
        m = shuffled_skills(items_by_subject(), seed=0)
        orig = {i: s for s, v in items_by_subject().items() for i in v}
        assert any(m[i] != orig[i] for i in orig)

    def test_deterministic_per_seed_and_input_order(self):
        subj = items_by_subject()
        rev = {s: list(reversed(v)) for s, v in reversed(list(subj.items()))}
        assert shuffled_skills(subj, 3) == shuffled_skills(subj, 3) == shuffled_skills(rev, 3)

    def test_differs_across_seeds(self):
        subj = items_by_subject()
        maps = [shuffled_skills(subj, s) for s in range(4)]
        assert all(maps[i] != maps[j] for i in range(4) for j in range(i + 1, 4))

    def test_rejects_item_in_two_subjects(self):
        with pytest.raises(ValueError, match="more than one"):
            shuffled_skills({"x": ["i1"], "y": ["i1"]}, 0)


def toy_responses() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    skill_of = {f"i{k}": ("s0" if k < 6 else "s1") for k in range(12)}
    rows = []
    for t in range(6):
        for item, s in skill_of.items():
            rows.append({"model": "m", "checkpoint_idx": t, "checkpoint_label": f"step{t}", "step": t,
                         "tokens_seen_b": pd.NA, "benchmark": "mmlu", "skill": s, "item_id": item,
                         "correct": int(rng.random() < (0.3 if s == "s0" else 0.7) + 0.05 * t)})
    return pd.DataFrame(rows)


class Recorder:
    """Stub predictor: records the skills it is shown and returns a distinct value per row."""

    name = "rec"

    def __init__(self):
        self.seen = []

    def predict(self, history, target, horizon=1):
        self.seen.append((history[["item_id", "skill"]].copy(), target[["item_id", "skill"]].copy()))
        return (target["item_id"].str[1:].astype(int).to_numpy() + 1) / 20.0


class TestRemappedSkills:
    def test_inner_sees_remapped_labels_and_inputs_untouched(self):
        resp = toy_responses()
        mapping = {f"i{k}": f"g{k % 3}" for k in range(12)}
        rec = Recorder()
        w = RemappedSkills(rec, mapping, "_test")
        hist = resp[resp["checkpoint_idx"] <= 3]
        target = resp[resp["checkpoint_idx"] == 4][TARGET_COLUMNS].reset_index(drop=True)
        hist_before, target_before = hist.copy(), target.copy()
        w.predict(hist, target)
        h, t = rec.seen[0]
        assert (h["skill"] == h["item_id"].map(mapping)).all()
        assert (t["skill"] == t["item_id"].map(mapping)).all()
        pd.testing.assert_frame_equal(hist, hist_before)
        pd.testing.assert_frame_equal(target, target_before)
        assert w.name == "rec_test"

    def test_preserves_target_order(self):
        resp = toy_responses()
        w = RemappedSkills(Recorder(), constant_skill(resp["item_id"]), "_gskill")
        hist = resp[resp["checkpoint_idx"] <= 3]
        target = resp[resp["checkpoint_idx"] == 4][TARGET_COLUMNS].sample(frac=1, random_state=1)
        target = target.reset_index(drop=True)
        p = w.predict(hist, target)
        expected = (target["item_id"].str[1:].astype(int).to_numpy() + 1) / 20.0
        assert np.array_equal(p, expected)

    def test_unmapped_item_rejected(self):
        resp = toy_responses()
        w = RemappedSkills(Recorder(), {"i0": "x"}, "_x")
        target = resp[resp["checkpoint_idx"] == 4][TARGET_COLUMNS].reset_index(drop=True)
        with pytest.raises(ValueError, match="no mapping"):
            w.predict(resp[resp["checkpoint_idx"] <= 3], target)

    def test_harness_scores_original_subjects(self):
        # SkillMean under the constant mapping predicts the last checkpoint's overall accuracy for
        # every item; the harness must still slice and score by the real subjects s0 and s1
        resp = toy_responses()
        w = RemappedSkills(SkillMean(), constant_skill(resp["item_id"]), "_gskill")
        res = evaluate(resp, [w], min_history=2, horizon=1)
        assert w.name == "skill_mean_gskill"
        assert set(res["skill"]) == {"s0", "s1"}
        for _, r in res.iterrows():
            p = resp.loc[resp["checkpoint_idx"] == r["origin_idx"], "correct"].mean()
            tgt = resp[(resp["checkpoint_idx"] == r["target_idx"]) & (resp["skill"] == r["skill"])]
            y = tgt["correct"].to_numpy()
            assert r["n"] == len(tgt) == 6
            assert r["log_loss"] == pytest.approx(log_loss(y, np.full(len(y), p)))
            assert np.isnan(r["auc"]) or r["auc"] == pytest.approx(auc(y, np.full(len(y), p)))

    def test_harness_within_skill_auc_on_original_subjects(self):
        # a predictor whose output depends on the remapped group: within each ORIGINAL subject the
        # AUC must be computed from these predictions against that subject's outcomes
        resp = toy_responses()
        mapping = {f"i{k}": ("g0" if k % 2 else "g1") for k in range(12)}

        class GroupRank:
            name = "group_rank"

            def predict(self, history, target, horizon=1):
                k = target["item_id"].str[1:].astype(int).to_numpy()
                return np.where(target["skill"] == "g0", 0.6, 0.3) + 0.01 * k

        res = evaluate(resp, [RemappedSkills(GroupRank(), mapping, "_g")], min_history=2)
        assert set(res["skill"]) == {"s0", "s1"}
        checked = 0
        for _, r in res.iterrows():
            tgt = resp[(resp["checkpoint_idx"] == r["target_idx"]) & (resp["skill"] == r["skill"])]
            k = tgt["item_id"].str[1:].astype(int).to_numpy()
            p = np.where(tgt["item_id"].map(mapping) == "g0", 0.6, 0.3) + 0.01 * k
            y = tgt["correct"].to_numpy()
            assert r["log_loss"] == pytest.approx(log_loss(y, p))
            if not np.isnan(r["auc"]):
                assert r["auc"] == pytest.approx(auc(y, p))
                checked += 1
        assert checked > 0
