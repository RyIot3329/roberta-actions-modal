"""
Permanent linear floor: TF-IDF (char 2-5 + word 1-2) + linear SVM (SGD hinge)
trained on data/train.jsonl, scored on validation/test with the same
metrics_core scorer as the transformer. Runs in seconds on CPU.

Writes output/baseline_linear_metrics.json (best_metrics v2 schema). The
transformer should clear this floor by a comfortable margin (>=5pp strict);
a run that does not is flagged in the results PR.

Usage: python scripts/baseline_linear.py [--data-dir data] [--out output/baseline_linear_metrics.json]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import metrics_core as mc  # noqa: E402


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def predictions(clf, vec, records, id2label):
    X = vec.transform([r["text"] for r in records])
    scores = clf.decision_function(X)
    if scores.ndim == 1:
        scores = np.stack([-scores, scores], axis=1)
    order = np.argsort(-scores, axis=1)
    top = order[:, 0]
    # pseudo-confidence: logistic of the top-1 vs top-2 margin
    margin = scores[np.arange(len(top)), top] - scores[np.arange(len(top)), order[:, 1]]
    conf = 1.0 / (1.0 + np.exp(-margin))
    preds = []
    for r, t, c, o in zip(records, top, conf, order):
        p = {"text": r["text"], "actual_label": r["label"], "predicted_label": id2label[int(t)],
             "confidence": float(c), "topk_labels": [id2label[int(j)] for j in o[:10]]}
        for k in ("site", "rows", "seen_in_train", "accept"):
            if k in r:
                p[k] = r[k]
        preds.append(p)
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="output/baseline_linear_metrics.json")
    ap.add_argument("--predictions-out", default=None,
                    help="Optional JSONL of test predictions (not committed by default)")
    args = ap.parse_args()

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import SGDClassifier
    from sklearn.pipeline import make_union

    train = load_jsonl(os.path.join(args.data_dir, "train.jsonl"))
    val = load_jsonl(os.path.join(args.data_dir, "validation.jsonl"))
    test = load_jsonl(os.path.join(args.data_dir, "test.jsonl"))
    with open(os.path.join(args.data_dir, "label_mapping.json")) as f:
        mapping = json.load(f)
    labels = sorted(mapping["label2id"])

    t0 = time.time()
    vec = make_union(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True),
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True),
    )
    X = vec.fit_transform([r["text"] for r in train])
    y = [r["label"] for r in train]
    clf = SGDClassifier(loss="hinge", alpha=2e-5, max_iter=40, tol=1e-4,
                        n_jobs=min(8, os.cpu_count() or 1), random_state=0).fit(X, y)
    id2label = {i: c for i, c in enumerate(clf.classes_)}
    fit_s = time.time() - t0

    val_preds = predictions(clf, vec, val, id2label)
    test_preds = predictions(clf, vec, test, id2label)
    tau = mc.fit_acceptance_threshold(val_preds, mc.DEFAULT_TARGET_PRECISION)
    val_metrics = mc.score_predictions(val_preds, tau=tau)
    test_metrics = mc.score_predictions(test_preds, tau=tau)
    fp = mc.fingerprints(test, labels)

    record = mc.build_metrics_record(
        test_metrics, fp, model="tfidf+sgd-hinge", git_sha=git_sha(),
        timestamp=datetime.now().isoformat(),
        extra={"validation_metrics": {k: v for k, v in val_metrics.items() if k != "coverage_curve"},
               "train_records": len(train), "fit_seconds": round(fit_s, 1)},
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(record, f, indent=2)
    if args.predictions_out:
        mc.write_predictions_jsonl(test_preds, args.predictions_out)

    s, l = test_metrics["strict"], test_metrics["lenient"]
    print(f"Linear floor (fit {fit_s:.0f}s, {len(train)} train texts, {len(labels)} classes)")
    print(f"  validation strict {val_metrics['strict']['accuracy']:.1%}  test strict {s['accuracy']:.1%} "
          f"lenient {l['accuracy']:.1%}  log1p-rows {s['log1p_rows_accuracy']:.1%}  "
          f"rows {s['rows_accuracy']:.1%}  f1w {s['f1_weighted']:.4f}")
    for name, sl in test_metrics["slices"].items():
        print(f"  {name:<24} n={sl['n']:<4} strict {sl['strict']:.1%}  lenient {sl['lenient']:.1%}")
    cov = test_metrics["coverage"]
    print(f"  coverage@tau={cov['tau']:.3f} (val-fitted for {mc.DEFAULT_TARGET_PRECISION:.0%} precision): "
          f"{cov['coverage_texts']:.1%} of texts, precision {cov['precision_at_tau']:.1%}")
    print(f"  top-3 {test_metrics['topk']['top3']:.1%}  top-5 {test_metrics['topk']['top5']:.1%}")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
