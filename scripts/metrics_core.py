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
# coverage at the validation-fitted threshold is REPORTED but not gated: on
# ~1k validation texts the fitted tau swings by tenths and the coverage with it
# (0.4% .. 47% between otherwise similar models), so it cannot block a promotion.
SECONDARY_AXES = (
    "lenient.accuracy",
    "strict.log1p_rows_accuracy",
)


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------

def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fingerprints(test_records, labels, pair_records=None) -> dict:
    """Identify the scoreboard: the exact held-out (text, label, site) set, the
    label space and, when present, the (text, context, label, site) pair set.
    A gate comparison is only meaningful when they match."""
    lines = sorted(f"{r['text']}\t{r['label']}\t{r.get('site', '')}" for r in test_records)
    label_list = sorted(set(labels))
    out = {
        "test_sha256": _sha256("\n".join(lines)),
        "label_space_sha256": _sha256("\n".join(label_list)),
        "n_test": len(test_records),
        "n_classes": len(label_list),
    }
    if pair_records:
        pairs = sorted(f"{r['text']}\t{r.get('context', '')}\t{r['label']}\t{r.get('site', '')}"
                       for r in pair_records)
        out["pairs_sha256"] = _sha256("\n".join(pairs))
        out["n_pairs"] = len(pair_records)
    return out


def fingerprints_match(a: dict, b: dict) -> bool:
    if not (a and b):
        return False
    if a.get("test_sha256") != b.get("test_sha256") \
            or a.get("label_space_sha256") != b.get("label_space_sha256"):
        return False
    # Pair sets must match when both sides have one
    if a.get("pairs_sha256") and b.get("pairs_sha256") and a["pairs_sha256"] != b["pairs_sha256"]:
        return False
    return True


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
    # The same normalized name can occur in two held-out sites with different
    # golds, so records are paired on (key, site) when a site is present
    def k(p):
        return (p[key], p.get("context", ""), p.get("site", ""))
    a = {k(p): p for p in preds_a}
    b = {k(p): p for p in preds_b}
    common = sorted(set(a) & set(b))
    return [a[c] for c in common], [b[c] for c in common], len(a), len(b)


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


def _binomial_margin(p, n, base_margin):
    """Non-inferiority margin widened to two binomial standard errors for
    small slices (a 230-text slice moves by ~3pp between seeds)."""
    if not n or p is None:
        return base_margin
    p = min(max(float(p), 0.01), 0.99)
    return max(base_margin, 2.0 * math.sqrt(p * (1 - p) / n))


def promote_decision(candidate: dict, baseline: dict,
                     candidate_preds=None, baseline_preds=None,
                     candidate_pairs=None, baseline_pairs=None,
                     margin: float = DEFAULT_MARGIN, B: int = DEFAULT_BOOTSTRAP_B) -> dict:
    """Composite non-inferiority gate.

    Primary: when both records carry the operational pair view
    (metrics.pairs_ensemble = max-confidence of name and context views on the
    held-out (name, context) pairs) it is the primary, judged by a paired
    bootstrap on the pair predictions; otherwise the per-name name-only strict
    accuracy is (bootstrap on the per-name predictions). Secondary axes must not
    regress beyond the margin: name-only strict accuracy (compared with the
    baseline run's SEED MEAN when it recorded one -- the deployed seed alone is
    a lucky draw), lenient accuracy, log1p-rows accuracy, and every per-name
    slice with a two-standard-error margin. Fingerprints must match.
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
    use_pairs = bool(_get(cm, "pairs_ensemble.accuracy") is not None
                     and _get(bm, "pairs_ensemble.accuracy") is not None)
    primary_axis = "pairs_ensemble.accuracy" if use_pairs else PRIMARY_METRIC
    primary_c, primary_b = _get(cm, primary_axis), _get(bm, primary_axis)
    preds_c = candidate_pairs if use_pairs else candidate_preds
    preds_b = baseline_pairs if use_pairs else baseline_preds
    boot = None
    if preds_c and preds_b:
        boot = paired_bootstrap(preds_b, preds_c, B=B)
        primary_ok = boot["ci_lo"] > -margin
        axes.append({"axis": primary_axis + " (paired bootstrap ci_lo)",
                     "baseline": primary_b, "candidate": primary_c,
                     "delta": boot["delta"], "ci_lo": boot["ci_lo"], "ci_hi": boot["ci_hi"],
                     "threshold": -margin, "ok": bool(primary_ok)})
    else:
        primary_ok = (primary_c is not None and primary_b is not None
                      and primary_c - primary_b > -margin)
        axes.append({"axis": primary_axis + " (point delta)", "baseline": primary_b,
                     "candidate": primary_c,
                     "delta": None if primary_c is None or primary_b is None else primary_c - primary_b,
                     "threshold": -margin, "ok": bool(primary_ok)})
    if not primary_ok:
        reasons.append("primary regressed")

    # Name-only strict accuracy: against the baseline run's seed mean when known
    if use_pairs:
        summary = baseline.get("seed_summary") or {}
        seed_tests = [v.get("test_strict") for v in summary.values() if v.get("test_strict") is not None]
        b_name = sum(seed_tests) / len(seed_tests) if seed_tests else _get(bm, PRIMARY_METRIC)
        c_summary = candidate.get("seed_summary") or {}
        c_tests = [v.get("test_strict") for v in c_summary.values() if v.get("test_strict") is not None]
        c_name = sum(c_tests) / len(c_tests) if c_tests else _get(cm, PRIMARY_METRIC)
        label = PRIMARY_METRIC + (" (seed means)" if seed_tests and c_tests else "")
        if b_name is not None and c_name is not None:
            ok = c_name - b_name > -margin
            axes.append({"axis": label, "baseline": b_name, "candidate": c_name,
                         "delta": c_name - b_name, "threshold": -margin, "ok": bool(ok)})
            if not ok:
                reasons.append(f"{label} regressed by {b_name - c_name:.4f}")

    secondary = list(SECONDARY_AXES)
    slice_names = sorted(_get(bm, "slices", {}) or {})
    for axis in secondary + [f"slices.{name}.strict" for name in slice_names]:
        b_val, c_val = _get(bm, axis), _get(cm, axis)
        if b_val is None or c_val is None or (isinstance(b_val, float) and math.isnan(b_val)):
            axes.append({"axis": axis, "baseline": b_val, "candidate": c_val,
                         "delta": None, "threshold": -margin, "ok": True, "note": "not comparable"})
            continue
        m = margin
        if axis.startswith("slices."):
            n = _get(bm, axis.rsplit(".", 1)[0] + ".n")
            m = _binomial_margin(b_val, n, margin)
        delta = c_val - b_val
        ok = delta > -m
        axes.append({"axis": axis, "baseline": b_val, "candidate": c_val,
                     "delta": delta, "threshold": -m, "ok": bool(ok)})
        if not ok:
            reasons.append(f"{axis} regressed by {-delta:.4f}")

    passed = primary_ok and all(a["ok"] for a in axes)
    return {
        "passed": bool(passed),
        "reason": "ok" if passed else "; ".join(reasons),
        "primary": primary_axis,
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
