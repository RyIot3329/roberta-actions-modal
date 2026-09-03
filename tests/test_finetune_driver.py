"""Tests for the local (non-Modal) helpers in scripts/finetune.py:
ensemble construction, seed agreement, context-view scoring.

Run: python3 tests/test_finetune_driver.py   (or pytest tests/)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import finetune  # noqa: E402
import metrics_core as mc  # noqa: E402


def _samples(n, labels):
    return [{"text": f"t{i}", "label": labels[i % len(labels)], "label_id": i % len(labels),
             "site": "S", "rows": 1 + i, "seen_in_train": bool(i % 2), "context": "eq a"}
            for i in range(n)]


def _pred(sample, pred_id, conf, id2label):
    return {"text": sample["text"], "actual_label": sample["label"], "actual_id": sample["label_id"],
            "predicted_label": id2label[pred_id], "predicted_id": pred_id, "confidence": conf,
            "correct": pred_id == sample["label_id"], "topk_labels": [id2label[pred_id]],
            "site": sample["site"], "rows": sample["rows"], "seen_in_train": sample["seen_in_train"],
            "context": sample["context"]}


def test_ensemble_predictions_averages_calibrated_probs():
    id2label = {0: "a", 1: "b", 2: "c"}
    samples = _samples(4, ["a", "b", "c"])
    # seed 1 is confident and right on record 0, seed 2 is confident and wrong
    logits1 = np.array([[5, 0, 0], [0, 5, 0], [0, 0, 5], [5, 0, 0]], dtype=np.float32)
    logits2 = np.array([[0, 4, 0], [0, 5, 0], [0, 0, 5], [5, 0, 0]], dtype=np.float32)
    seed_results = {1: {"test_logits": logits1, "calibration_temperature": 1.0},
                    2: {"test_logits": logits2, "calibration_temperature": 1.0}}
    preds = finetune._ensemble_predictions(seed_results, samples, "test", id2label)
    assert [p["predicted_label"] for p in preds] == ["a", "b", "c", "a"]  # 5 beats 4 on record 0
    assert preds[1]["correct"] and 0.5 < preds[0]["confidence"] < 1.0
    assert preds[0]["topk_labels"][:2] == ["a", "b"]


def test_seed_agreement():
    base = [{"predicted_id": i % 3} for i in range(9)]
    other = [{"predicted_id": (i % 3) if i < 6 else 0} for i in range(9)]
    seed_results = {1: {"test_inference": {"predictions": base}},
                    2: {"test_inference": {"predictions": other}}}
    agg = finetune._seed_agreement(seed_results)
    assert abs(agg["pairwise_mean"] - 7 / 9) < 1e-9 and abs(agg["unanimous"] - 7 / 9) < 1e-9
    assert finetune._seed_agreement({1: {"test_inference": {"predictions": base}}}) is None


def test_score_context_views_alignment_and_ensemble():
    id2label = {0: "a", 1: "b"}
    samples = _samples(6, ["a", "b"])
    # context view: right on all but the last; name view: right on first two only,
    # but very confident on record 5 where context is wrong -> ensemble takes name
    ctx = [_pred(s, s["label_id"] if i < 5 else 1 - s["label_id"], 0.6, id2label)
           for i, s in enumerate(samples)]
    name = [_pred(s, s["label_id"] if i < 2 or i == 5 else 1 - s["label_id"],
                  0.9 if i == 5 else 0.3, id2label) for i, s in enumerate(samples)]
    out = finetune._score_context_views(ctx, ctx, name, name, 0.85, mc)
    assert abs(out["test"]["context"]["strict"]["accuracy"] - 5 / 6) < 1e-9
    assert abs(out["test"]["name_on_pairs"]["strict"]["accuracy"] - 3 / 6) < 1e-9
    assert abs(out["test"]["ensemble"]["strict"]["accuracy"] - 1.0) < 1e-9
    assert abs(out["ensemble_source_context_share"] - 5 / 6) < 1e-9
    summary = finetune._ctx_summary(out)
    assert set(summary) == {"name_on_pairs", "context", "ensemble"}
    try:
        finetune._score_context_views(ctx, ctx[:-1], name, name, 0.85, mc)
        raise AssertionError("misaligned views must fail")
    except AssertionError as e:
        assert "align" in str(e) or "misaligned" not in str(e)


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
