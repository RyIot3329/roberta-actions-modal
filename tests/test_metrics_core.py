"""Unit + regression tests for scripts/metrics_core.py.

Run: python3 tests/test_metrics_core.py   (or pytest tests/)
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import metrics_core as mc  # noqa: E402


def _preds(n_correct, n_wrong, site="A", seen=False, conf_correct=0.9, conf_wrong=0.4):
    out = []
    for i in range(n_correct):
        out.append({"text": f"{site}-c{i}", "site": site, "rows": 1 + i % 3, "seen_in_train": seen,
                    "actual_label": "x", "predicted_label": "x", "confidence": conf_correct})
    for i in range(n_wrong):
        out.append({"text": f"{site}-w{i}", "site": site, "rows": 1, "seen_in_train": seen,
                    "actual_label": "x", "predicted_label": "y", "confidence": conf_wrong})
    return out


def test_score_basic():
    preds = _preds(8, 2)
    m = mc.score_predictions(preds, tau=0.5)
    assert abs(m["strict"]["accuracy"] - 0.8) < 1e-9
    assert m["lenient"]["accuracy"] >= m["strict"]["accuracy"]
    assert m["slices"]["site:A"]["n"] == 10
    assert m["slices"]["seen_in_train:false"]["strict"] == 0.8
    # coverage at tau=0.5 accepts only the 8 correct ones -> precision 1.0
    assert abs(m["coverage"]["coverage_texts"] - 0.8) < 1e-9
    assert abs(m["coverage"]["precision_at_tau"] - 1.0) < 1e-9


def test_lenient_uses_accept_set():
    preds = _preds(0, 4)
    for p in preds[:2]:
        p["accept"] = ["x", "y"]
    m = mc.score_predictions(preds)
    assert m["strict"]["accuracy"] == 0.0
    assert abs(m["lenient"]["accuracy"] - 0.5) < 1e-9


def test_fit_threshold():
    preds = _preds(9, 1, conf_correct=0.9, conf_wrong=0.95)  # the wrong one is most confident
    # accepting the top-1 gives precision 0 -> below target; top-10 gives 0.9
    tau = mc.fit_acceptance_threshold(preds, target_precision=0.9)
    assert tau == 0.9
    preds = _preds(5, 5, conf_correct=0.9, conf_wrong=0.1)
    tau = mc.fit_acceptance_threshold(preds, target_precision=0.9)
    assert tau == 0.9  # accepting the 0.1 ones would drop precision to 0.5
    assert mc.fit_acceptance_threshold(_preds(0, 5), 0.9) == 1.0


def test_fingerprints_change_with_test_set():
    recs = [{"text": "a", "label": "x", "site": "S"}, {"text": "b", "label": "y", "site": "S"}]
    f1 = mc.fingerprints(recs, ["x", "y"])
    f2 = mc.fingerprints(recs[::-1], ["y", "x"])       # order-invariant
    assert mc.fingerprints_match(f1, f2)
    f3 = mc.fingerprints([{"text": "a", "label": "x", "site": "S"},
                          {"text": "b2", "label": "y", "site": "S"}], ["x", "y"])
    assert not mc.fingerprints_match(f1, f3)
    f4 = mc.fingerprints(recs, ["x", "y", "z"])         # label space change
    assert not mc.fingerprints_match(f1, f4)


def _record(preds, fp):
    return mc.build_metrics_record(mc.score_predictions(preds, tau=0.5), fp, model="m")


def test_gate_blocks_stale_baseline():
    fp_a = {"test_sha256": "1", "label_space_sha256": "1", "n_test": 1, "n_classes": 1}
    fp_b = {"test_sha256": "2", "label_space_sha256": "1", "n_test": 1, "n_classes": 1}
    d = mc.promote_decision(_record(_preds(8, 2), fp_b), _record(_preds(8, 2), fp_a))
    assert not d["passed"] and d["reason"] == "stale_baseline"


def test_gate_passes_equal_and_blocks_slice_regression():
    fp = {"test_sha256": "1", "label_space_sha256": "1", "n_test": 1, "n_classes": 1}
    base = _preds(80, 20, site="A") + _preds(60, 40, site="B")
    same = [dict(p) for p in base]
    d = mc.promote_decision(_record(same, fp), _record(base, fp), same, base)
    assert d["passed"], d
    # regress site B by flipping 3 correct predictions to wrong (3/100 = 3pp)
    worse = [dict(p) for p in base]
    flipped = 0
    for p in worse:
        if p["site"] == "B" and p["predicted_label"] == "x" and flipped < 3:
            p["predicted_label"] = "z"
            flipped += 1
    d = mc.promote_decision(_record(worse, fp), _record(base, fp), worse, base)
    assert not d["passed"]
    assert any(a["axis"] == "slices.site:B.strict" and not a["ok"] for a in d["axes"]), d["axes"]
    table = mc.format_axes_table(d)
    assert "site:B" in table and "FAILED" in table


def test_paired_bootstrap_on_archived_runs():
    """Seed 42 vs seed 43 on identical data differ by 0.5pp: the bootstrap
    must call that non-inferior at the 0.005 margin (it is seed noise)."""
    def load(path):
        txt = open(path).read()
        i = txt.find("\n{\n")
        blob = txt[i + 1:]
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            j = blob.rfind("\n}")
            return json.loads(blob[: j + 2])
    p42 = ROOT / "output/20260713_191733_deberta-v3-bms-base.txt"
    p43 = ROOT / "output/20260713_205617_deberta-v3-bms-base.txt"
    if not (p42.exists() and p43.exists()):
        print("  (archived runs missing, skipping)")
        return
    a = load(p42)["test_inference"]["predictions"]
    b = load(p43)["test_inference"]["predictions"]
    boot = mc.paired_bootstrap(a, b, B=2000)
    assert boot["n_paired"] > 800
    assert abs(boot["delta"] - (-0.0048)) < 0.002, boot
    assert boot["ci_lo"] > -0.03 and boot["ci_hi"] > 0  # noise straddles zero
    # a 2pp forced regression must be caught
    worse = [dict(p) for p in a]
    n_flip = int(0.03 * len(worse))
    flipped = 0
    for p in worse:
        if p["correct"] and flipped < n_flip:
            p["predicted_label"] = "__wrong__"
            flipped += 1
    boot2 = mc.paired_bootstrap(a, worse, B=2000)
    assert boot2["ci_hi"] < 0, boot2


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    sys.exit(1 if failed else 0)
