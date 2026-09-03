"""
Shared scoring for the autotagger: slices, lenient accept sets, fingerprints,
coverage at a fixed precision, paired bootstrap, and the composite gate.

Every consumer (finetune.py, rescore_baseline.py, baseline_linear.py,
evaluate_external.py) scores prediction records through this module so their
numbers are comparable. Pure Python + numpy + scikit-learn; no torch.

A prediction record is a dict with at least:
    text, actual_label, predicted_label, confidence
and optionally:
    site, rows, seen_in_train, accept (list of acceptable labels),
    topk_labels (labels ranked by probability, best first)

`output/best_metrics.json` v2 (written by build_metrics_record):
    schema_version, model, hf_repo, git_sha, timestamp, seeds, selected_seed,
    fingerprints{test_sha256, label_space_sha256, n_test, n_classes},
    predictions_path, metrics{strict, lenient, slices, coverage, topk,
    calibration}, gate{primary, margin, bootstrap_B}
"""

import hashlib
import json
import math
from collections import defaultdict

import numpy as np

SCHEMA_VERSION = 2
PRIMARY_METRIC = "strict.accuracy"
DEFAULT_MARGIN = 0.005          # ~ the measured single-seed spread on 830 texts
DEFAULT_BOOTSTRAP_B = 10000
DEFAULT_TARGET_PRECISION = 0.85   # 0.90 is unreachable at useful coverage today (val-fitted tau 0.94 -> 0.4% coverage)
THRESHOLD_GRID = [round(x, 2) for x in np.arange(0.05, 1.0, 0.05)]

# Secondary axes (dotted paths into the metrics dict) that must not regress by
# more than the margin. Slices are added dynamically from the baseline.
SECONDARY_AXES = (
    "lenient.accuracy",
    "strict.log1p_rows_accuracy",
    "coverage.coverage_log1p_rows",
)


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------

def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fingerprints(test_records, labels) -> dict:
    """Identify the scoreboard: the exact held-out (text, label, site) set and
    the label space. A gate comparison is only meaningful when both match."""
    lines = sorted(f"{r['text']}\t{r['label']}\t{r.get('site', '')}" for r in test_records)
    label_list = sorted(set(labels))
    return {
        "test_sha256": _sha256("\n".join(lines)),
        "label_space_sha256": _sha256("\n".join(label_list)),
        "n_test": len(test_records),
        "n_classes": len(label_list),
    }


def fingerprints_match(a: dict, b: dict) -> bool:
    return bool(a and b
                and a.get("test_sha256") == b.get("test_sha256")
                and a.get("label_space_sha256") == b.get("label_space_sha256"))


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _accept_set(pred: dict) -> set:
    accept = set(pred.get("accept") or [])
    accept.add(pred["actual_label"])
    return accept


def _weights(preds, kind: str) -> np.ndarray:
    if kind == "rows":
        return np.array([float(p.get("rows", 1) or 1) for p in preds])
    if kind == "log1p_rows":
        return np.array([math.log1p(float(p.get("rows", 1) or 1)) for p in preds])
    return np.ones(len(preds))


def _f1(actual, predicted, average: str) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(actual, predicted, average=average, zero_division=0))


def _ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    if len(conf) == 0:
        return float("nan")
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            ece += mask.mean() * abs(conf[mask].mean() - correct[mask].mean())
    return float(ece)


def fit_acceptance_threshold(preds, target_precision: float = DEFAULT_TARGET_PRECISION) -> float:
    """Smallest confidence threshold at which strict precision among the
    accepted predictions reaches `target_precision` (fit on VALIDATION only).
    Returns 1.0 when no threshold reaches the target."""
    if not preds:
        return 1.0
    order = sorted(preds, key=lambda p: -float(p["confidence"]))
    correct = np.array([p["predicted_label"] == p["actual_label"] for p in order], dtype=float)
    conf = np.array([float(p["confidence"]) for p in order])
    cum_precision = np.cumsum(correct) / np.arange(1, len(order) + 1)
    best = 1.0
    for i in range(len(order)):
        # accepting everything down to conf[i] (inclusive) yields cum_precision[i]
        if cum_precision[i] >= target_precision:
            best = float(conf[i])
    return best


def coverage_at(preds, tau: float) -> dict:
    conf = np.array([float(p["confidence"]) for p in preds])
    correct = np.array([p["predicted_label"] == p["actual_label"] for p in preds], dtype=float)
    accepted = conf >= tau
    w = _weights(preds, "log1p_rows")
    out = {
        "tau": float(tau),
        "coverage_texts": float(accepted.mean()) if len(preds) else float("nan"),
        "coverage_log1p_rows": float(w[accepted].sum() / w.sum()) if w.sum() else float("nan"),
        "precision_at_tau": float(correct[accepted].mean()) if accepted.any() else float("nan"),
    }
    return out


def coverage_curve(preds) -> list:
    curve = []
    for tau in THRESHOLD_GRID:
        c = coverage_at(preds, tau)
        curve.append({"tau": tau, "coverage": round(c["coverage_texts"], 4),
                      "precision": None if math.isnan(c["precision_at_tau"])
                      else round(c["precision_at_tau"], 4)})
    return curve


def score_predictions(preds, tau=None, tau_source: str = "validation") -> dict:
    """Strict + lenient accuracy, row-aware accuracy, slices, top-k, ECE,
    coverage at `tau` (fit elsewhere, on validation). Robust to missing
    optional fields."""
    preds = list(preds)
    if not preds:
        raise ValueError("no predictions to score")
    actual = [p["actual_label"] for p in preds]
    predicted = [p["predicted_label"] for p in preds]
    strict = np.array([a == b for a, b in zip(actual, predicted)], dtype=float)
    lenient = np.array([p["predicted_label"] in _accept_set(p) for p in preds], dtype=float)
    conf = np.array([float(p.get("confidence", 0.0)) for p in preds])
    w_rows = _weights(preds, "rows")
    w_log = _weights(preds, "log1p_rows")

    def block(correct):
        return {
            "accuracy": float(correct.mean()),
            "rows_accuracy": float((correct * w_rows).sum() / w_rows.sum()),
            "log1p_rows_accuracy": float((correct * w_log).sum() / w_log.sum()),
            "n": int(len(correct)),
        }

    metrics = {
        "strict": {**block(strict),
                   "f1_weighted": _f1(actual, predicted, "weighted"),
                   "f1_macro": _f1(actual, predicted, "macro")},
        "lenient": block(lenient),
    }

    # Slices: per site and seen/unseen
    slices = {}
    groups = defaultdict(list)
    for i, p in enumerate(preds):
        if p.get("site") is not None:
            groups[f"site:{p['site']}"].append(i)
        if p.get("seen_in_train") is not None:
            groups[f"seen_in_train:{str(bool(p['seen_in_train'])).lower()}"].append(i)
    for name, idx in sorted(groups.items()):
        idx = np.array(idx)
        slices[name] = {
            "n": int(len(idx)),
            "strict": float(strict[idx].mean()),
            "lenient": float(lenient[idx].mean()),
            "log1p_rows_strict": float((strict[idx] * w_log[idx]).sum() / w_log[idx].sum()),
        }
    metrics["slices"] = slices

    # Top-k when ranked labels are available
    if all(p.get("topk_labels") for p in preds):
        topk = {}
        for k in (3, 5, 10):
            hits = [p["actual_label"] in list(p["topk_labels"])[:k] for p in preds]
            topk[f"top{k}"] = float(np.mean(hits))
        metrics["topk"] = topk

    metrics["calibration"] = {
        "ece": _ece(conf, strict),
        "mean_confidence": float(conf.mean()),
        "mean_confidence_correct": float(conf[strict == 1].mean()) if strict.any() else float("nan"),
        "mean_confidence_wrong": float(conf[strict == 0].mean()) if (strict == 0).any() else float("nan"),
    }
    if tau is not None:
        metrics["coverage"] = {**coverage_at(preds, tau), "tau_source": tau_source}
    metrics["coverage_curve"] = coverage_curve(preds)
    return metrics


# ---------------------------------------------------------------------------
# Paired bootstrap + gate
# ---------------------------------------------------------------------------

def _join(preds_a, preds_b, key: str):
    a = {p[key]: p for p in preds_a}
    b = {p[key]: p for p in preds_b}
    common = sorted(set(a) & set(b))
    return [a[k] for k in common], [b[k] for k in common], len(a), len(b)


def paired_bootstrap(preds_baseline, preds_candidate, key: str = "text",
                     B: int = DEFAULT_BOOTSTRAP_B, seed: int = 0,
                     lenient: bool = False) -> dict:
    """Bootstrap the accuracy delta (candidate - baseline) over the shared
    texts, resampling texts once per draw and applying the same draw to both
    models (paired). Returns the point delta, the 95% CI and P(delta < 0)."""
    base, cand, n_a, n_b = _join(preds_baseline, preds_candidate, key)
    if not base:
        raise ValueError("no overlapping records between baseline and candidate")
    if lenient:
        ca = np.array([p["predicted_label"] in _accept_set(p) for p in base], dtype=float)
        cb = np.array([p["predicted_label"] in _accept_set(p) for p in cand], dtype=float)
    else:
        ca = np.array([p["predicted_label"] == p["actual_label"] for p in base], dtype=float)
        cb = np.array([p["predicted_label"] == p["actual_label"] for p in cand], dtype=float)
    diff = cb - ca
    rng = np.random.default_rng(seed)
    n = len(diff)
    idx = rng.integers(0, n, size=(B, n))
    deltas = diff[idx].mean(axis=1)
    return {
        "n_paired": int(n),
        "n_baseline": int(n_a),
        "n_candidate": int(n_b),
        "delta": float(diff.mean()),
        "ci_lo": float(np.percentile(deltas, 2.5)),
        "ci_hi": float(np.percentile(deltas, 97.5)),
        "p_worse": float((deltas < 0).mean()),
        "B": int(B),
    }


def _get(d: dict, path: str, default=None):
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def promote_decision(candidate: dict, baseline: dict,
                     candidate_preds=None, baseline_preds=None,
                     margin: float = DEFAULT_MARGIN, B: int = DEFAULT_BOOTSTRAP_B) -> dict:
    """Composite non-inferiority gate.

    candidate/baseline are metrics records (build_metrics_record output).
    Passes iff fingerprints match, the paired-bootstrap CI of the primary
    delta excludes a drop larger than `margin`, and every secondary axis
    (lenient, log1p-rows, coverage at 90%% precision, each slice) has a point
    delta better than -margin. Without prediction files the primary check
    falls back to the point delta.
    """
    axes = []
    reasons = []

    if not fingerprints_match(candidate.get("fingerprints"), baseline.get("fingerprints")):
        return {
            "passed": False,
            "reason": "stale_baseline",
            "detail": ("test set or label space changed since the baseline was written; "
                       "run scripts/rescore_baseline.py to re-seed it"),
            "axes": axes,
        }

    cm, bm = candidate.get("metrics", {}), baseline.get("metrics", {})
    primary_c = _get(cm, PRIMARY_METRIC)
    primary_b = _get(bm, PRIMARY_METRIC)
    boot = None
    if candidate_preds and baseline_preds:
        boot = paired_bootstrap(baseline_preds, candidate_preds, B=B)
        primary_ok = boot["ci_lo"] > -margin
        axes.append({"axis": PRIMARY_METRIC + " (paired bootstrap ci_lo)",
                     "baseline": primary_b, "candidate": primary_c,
                     "delta": boot["delta"], "ci_lo": boot["ci_lo"], "ci_hi": boot["ci_hi"],
                     "threshold": -margin, "ok": bool(primary_ok)})
    else:
        primary_ok = (primary_c is not None and primary_b is not None
                      and primary_c - primary_b > -margin)
        axes.append({"axis": PRIMARY_METRIC + " (point delta)", "baseline": primary_b,
                     "candidate": primary_c,
                     "delta": None if primary_c is None or primary_b is None else primary_c - primary_b,
                     "threshold": -margin, "ok": bool(primary_ok)})
    if not primary_ok:
        reasons.append("primary regressed")

    secondary = list(SECONDARY_AXES)
    secondary += [f"slices.{name}.strict" for name in sorted(_get(bm, "slices", {}) or {})]
    for axis in secondary:
        b_val, c_val = _get(bm, axis), _get(cm, axis)
        if b_val is None or c_val is None or (isinstance(b_val, float) and math.isnan(b_val)):
            axes.append({"axis": axis, "baseline": b_val, "candidate": c_val,
                         "delta": None, "threshold": -margin, "ok": True, "note": "not comparable"})
            continue
        delta = c_val - b_val
        ok = delta > -margin
        axes.append({"axis": axis, "baseline": b_val, "candidate": c_val,
                     "delta": delta, "threshold": -margin, "ok": bool(ok)})
        if not ok:
            reasons.append(f"{axis} regressed by {-delta:.4f}")

    # Promotion needs strict non-inferiority everywhere AND a strict-or-equal
    # primary: an equal primary is allowed only when nothing regressed.
    passed = primary_ok and all(a["ok"] for a in axes)
    return {
        "passed": bool(passed),
        "reason": "ok" if passed else "; ".join(reasons),
        "primary": PRIMARY_METRIC,
        "margin": margin,
        "bootstrap": boot,
        "axes": axes,
    }


def format_axes_table(decision: dict) -> str:
    """Markdown table of the gate axes for PR bodies / results files."""
    lines = ["| axis | baseline | candidate | delta | ok |", "|---|---|---|---|---|"]
    for a in decision.get("axes", []):
        def fmt(v):
            return "" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))
        lines.append(f"| {a['axis']} | {fmt(a.get('baseline'))} | {fmt(a.get('candidate'))} | "
                     f"{fmt(a.get('delta'))} | {'yes' if a.get('ok') else 'NO'}"
                     f"{' (' + a['note'] + ')' if a.get('note') else ''} |")
    header = f"Gate: {'PASSED' if decision.get('passed') else 'FAILED'} -- {decision.get('reason', '')}"
    if decision.get("bootstrap"):
        b = decision["bootstrap"]
        header += (f" | paired bootstrap on {b['n_paired']} texts: delta {b['delta']:+.4f} "
                   f"[{b['ci_lo']:+.4f}, {b['ci_hi']:+.4f}], P(worse) {b['p_worse']:.2f}")
    return header + "\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Records and I/O
# ---------------------------------------------------------------------------

def build_metrics_record(metrics: dict, fingerprints_: dict, model: str, hf_repo=None,
                         git_sha=None, timestamp=None, seeds=None, selected_seed=None,
                         predictions_path=None, extra=None) -> dict:
    record = {
        "schema_version": SCHEMA_VERSION,
        "model": model,
        "hf_repo": hf_repo,
        "git_sha": git_sha,
        "timestamp": timestamp,
        "seeds": seeds,
        "selected_seed": selected_seed,
        "fingerprints": fingerprints_,
        "predictions_path": predictions_path,
        "metrics": {k: v for k, v in metrics.items() if k != "coverage_curve"},
        "gate": {"primary": PRIMARY_METRIC, "margin": DEFAULT_MARGIN,
                 "bootstrap_B": DEFAULT_BOOTSTRAP_B},
    }
    if extra:
        record.update(extra)
    return record


def write_predictions_jsonl(preds, path):
    keys = ("text", "site", "rows", "seen_in_train", "actual_label", "predicted_label",
            "confidence", "accept", "topk_labels")
    with open(path, "w", encoding="utf-8") as f:
        for p in preds:
            row = {k: p[k] for k in keys if k in p}
            row["correct"] = p["predicted_label"] == p["actual_label"]
            row["correct_lenient"] = p["predicted_label"] in _accept_set(p)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_predictions_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def legacy_baseline(path_json: str):
    """Read a v1 best_metrics.json (test_f1_weighted/test_accuracy only)."""
    with open(path_json) as f:
        data = json.load(f)
    if data.get("schema_version", 1) >= 2:
        return data
    return {
        "schema_version": 1,
        "model": data.get("model"),
        "fingerprints": None,
        "metrics": {"strict": {"accuracy": data.get("test_accuracy"),
                               "f1_weighted": data.get("test_f1_weighted")}},
    }
