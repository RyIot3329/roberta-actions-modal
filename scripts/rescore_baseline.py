"""
Re-seed the quality-gate baseline from the DEPLOYED model.

The gate compares a candidate against output/best_metrics.json. That file is
only meaningful when it was computed on the same scoreboard (same held-out
records, same label space). When the data pipeline changes the test set or
the label space (new sites, label-space cleanup), run this script: it scores
the currently deployed Hub model on the CURRENT data/test.jsonl with the
shared metrics_core scorer and writes a fingerprinted v2 baseline plus the
per-record predictions the paired bootstrap needs.

Usage:
    python scripts/rescore_baseline.py --check          # exit 3 if baseline is stale/missing
    python scripts/rescore_baseline.py                  # rescore RyIoT33/haystack-autotagging
    python scripts/rescore_baseline.py --model-id IOTechSystems/haystack-autotagging --offline

Token: HF_TOKEN env var or HUGGING_FACE_TOKEN in .env (repo root).
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import metrics_core as mc  # noqa: E402

DEFAULT_REPO = "RyIoT33/haystack-autotagging"
BATCH_SIZE = 256


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def current_fingerprints(test_path, label_mapping_path, pairs_path="data/test_ctx.jsonl"):
    test = load_jsonl(test_path)
    with open(label_mapping_path) as f:
        labels = list(json.load(f)["label2id"])
    pairs = load_jsonl(pairs_path) if os.path.exists(pairs_path) else None
    return mc.fingerprints(test, labels, pair_records=pairs), test, labels


def check(args) -> int:
    fp, _, _ = current_fingerprints(args.test, args.label_mapping)
    if not os.path.exists(args.out):
        print(f"No baseline at {args.out}: STALE (missing)")
        return 3
    with open(args.out) as f:
        baseline = json.load(f)
    if baseline.get("schema_version", 1) < 2 or not baseline.get("fingerprints"):
        print(f"Baseline at {args.out} is schema v1 (no fingerprints): STALE")
        return 3
    if mc.fingerprints_match(fp, baseline["fingerprints"]):
        if fp.get("pairs_sha256") and not baseline["fingerprints"].get("pairs_sha256"):
            print(f"Baseline is STALE: the scoreboard gained a (name, context) pair test "
                  f"({fp['n_pairs']} pairs) the baseline never scored")
            return 3
        print(f"Baseline matches the current scoreboard "
              f"({fp['n_test']} test records, {fp['n_classes']} classes"
              f"{', ' + str(fp['n_pairs']) + ' pairs' if fp.get('n_pairs') else ''}): OK")
        return 0
    b = baseline["fingerprints"]
    print(f"Baseline is STALE: test {b.get('n_test')}->{fp['n_test']} records "
          f"(sha {str(b.get('test_sha256'))[:8]}->{fp['test_sha256'][:8]}), "
          f"label space {b.get('n_classes')}->{fp['n_classes']} classes")
    return 3


def load_model(model_id, token, offline):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    kwargs = {"local_files_only": True} if offline else {"token": token}
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
        model = AutoModelForSequenceClassification.from_pretrained(model_id, **kwargs)
    except Exception as e:  # noqa: BLE001
        if offline:
            raise
        print(f"WARNING: Hub fetch failed ({type(e).__name__}: {e}); retrying from local cache")
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_id, local_files_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return tokenizer, model.to(device).eval(), device


def predict(records, tokenizer, model, device, id2label, temperature, max_len=32,
            with_context=False):
    import torch
    preds = []
    texts = [(f"{r['text']} | {r['context']}" if with_context and r.get("context") else r["text"])
             for r in records]
    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=max_len,
                               return_tensors="pt").to(device)
            probs = torch.softmax(model(**inputs).logits / temperature, dim=1)
            top = probs.topk(10, dim=1)
            for r, ids, confs in zip(records[i:i + BATCH_SIZE], top.indices.tolist(), top.values.tolist()):
                p = {"text": r["text"], "actual_label": r["label"],
                     "predicted_label": id2label[ids[0]], "confidence": float(confs[0]),
                     "topk_labels": [id2label[j] for j in ids]}
                for k in ("site", "rows", "seen_in_train", "accept", "context", "pair_seen_in_train"):
                    if k in r:
                        p[k] = r[k]
                preds.append(p)
    return preds


def ensemble(name_preds, ctx_preds):
    """Max-confidence of the name-only and context views, record by record."""
    out = []
    for n, c in zip(name_preds, ctx_preds):
        best = c if c["confidence"] > n["confidence"] else n
        q = dict(best)
        q["source"] = "context" if best is c else "name"
        out.append(q)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default=DEFAULT_REPO, help="Hub repo id or local model dir")
    ap.add_argument("--offline", action="store_true", help="Use the local HF cache only")
    ap.add_argument("--check", action="store_true",
                    help="Only check whether the baseline matches the current scoreboard (exit 3 if not)")
    ap.add_argument("--test", default="data/test.jsonl")
    ap.add_argument("--val", default="data/validation.jsonl")
    ap.add_argument("--label-mapping", default="data/label_mapping.json")
    ap.add_argument("--out", default="output/best_metrics.json")
    ap.add_argument("--predictions-out", default="output/best_predictions.jsonl")
    ap.add_argument("--max-seq-length", type=int, default=32)
    args = ap.parse_args()

    if args.check:
        sys.exit(check(args))

    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_TOKEN")

    # Alias credit: a model trained on a pre-cleanup label space predicts old
    # names; canonicalizing them scores it fairly on the current scoreboard
    from convert_to_jsonl import build_label_canonicalizer
    canon = build_label_canonicalizer("data/eo66.xlsx", "data/target_audit.csv")

    fp, test, labels = current_fingerprints(args.test, args.label_mapping)
    val = load_jsonl(args.val)
    print(f"Scoreboard: {fp['n_test']} test records, {fp['n_classes']} classes "
          f"(test sha {fp['test_sha256'][:12]})")

    print(f"Loading model {args.model_id}{' (offline)' if args.offline else ''}...")
    tokenizer, model, device = load_model(args.model_id, token, args.offline)
    id2label = {int(k): canon(v) for k, v in model.config.id2label.items()}
    temperature = float(getattr(model.config, "calibration_temperature", None) or 1.0)
    snapshot = getattr(getattr(model, "config", None), "_commit_hash", None)
    print(f"  {len(id2label)} classes, calibration temperature {temperature:.3f}, "
          f"commit {snapshot or 'n/a'}, device {device}")

    val_preds = predict(val, tokenizer, model, device, id2label, temperature, args.max_seq_length)
    tau = mc.fit_acceptance_threshold(val_preds, mc.DEFAULT_TARGET_PRECISION)
    val_metrics = mc.score_predictions(val_preds, tau=tau)
    test_preds = predict(test, tokenizer, model, device, id2label, temperature, args.max_seq_length)
    test_metrics = mc.score_predictions(test_preds, tau=tau)

    # Operational pair view: name-only on the (name, context) pairs, plus the
    # context view and their max-confidence ensemble when the model was trained
    # with context (config context_trained); a name-only model's pair view is
    # its name-only predictions
    pairs_preds = None
    pairs_path = "data/test_ctx.jsonl"
    if os.path.exists(pairs_path):
        pairs = load_jsonl(pairs_path)
        max_len = max(args.max_seq_length, 64)
        name_on_pairs = predict(pairs, tokenizer, model, device, id2label, temperature, max_len)
        context_trained = bool(getattr(model.config, "context_trained", False))
        if context_trained:
            ctx_view = predict(pairs, tokenizer, model, device, id2label, temperature, max_len,
                               with_context=True)
            pairs_preds = ensemble(name_on_pairs, ctx_view)
            test_metrics["pairs_context"] = {k: v for k, v in mc.score_predictions(ctx_view, tau=tau).items()
                                             if k != "coverage_curve"}
        else:
            pairs_preds = name_on_pairs
        test_metrics["pairs_name"] = {k: v for k, v in mc.score_predictions(name_on_pairs, tau=tau).items()
                                      if k != "coverage_curve"}
        test_metrics["pairs_ensemble"] = {k: v for k, v in mc.score_predictions(pairs_preds, tau=tau).items()
                                          if k != "coverage_curve"}
        print(f"  pair view ({len(pairs)} pairs, context_trained={context_trained}): "
              f"name-only {test_metrics['pairs_name']['strict']['accuracy']:.2%}, "
              f"operational {test_metrics['pairs_ensemble']['strict']['accuracy']:.2%}")

    record = mc.build_metrics_record(
        test_metrics, fp, model=args.model_id, hf_repo=args.model_id, git_sha=git_sha(),
        timestamp=datetime.now().isoformat(), seeds=None, selected_seed=None,
        predictions_path=args.predictions_out,
        extra={"reseeded_from": "deployed_hf_model", "model_commit": snapshot,
               "calibration_temperature": temperature,
               "validation_metrics": {k: v for k, v in val_metrics.items() if k != "coverage_curve"}},
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if pairs_preds is not None:
        pairs_out = os.path.join(os.path.dirname(args.predictions_out) or ".", "best_predictions_pairs.jsonl")
        record["predictions_pairs_path"] = pairs_out
        mc.write_predictions_jsonl(pairs_preds, pairs_out)
    with open(args.out, "w") as f:
        json.dump(record, f, indent=2)
    mc.write_predictions_jsonl(test_preds, args.predictions_out)

    s = test_metrics["strict"]
    print(f"\nRe-seeded baseline from {args.model_id}:")
    print(f"  test strict {s['accuracy']:.2%} (f1w {s['f1_weighted']:.4f}), "
          f"lenient {test_metrics['lenient']['accuracy']:.2%}, log1p-rows {s['log1p_rows_accuracy']:.2%}, "
          f"rows {s['rows_accuracy']:.2%}")
    for name, sl in test_metrics["slices"].items():
        print(f"  {name:<24} n={sl['n']:<4} strict {sl['strict']:.1%}  lenient {sl['lenient']:.1%}")
    cov = test_metrics["coverage"]
    print(f"  coverage@tau={cov['tau']:.3f} (validation-fitted, {mc.DEFAULT_TARGET_PRECISION:.0%} precision): "
          f"{cov['coverage_texts']:.1%} of texts at precision {cov['precision_at_tau']:.1%}")
    print(f"  top-3 {test_metrics['topk']['top3']:.1%}  top-5 {test_metrics['topk']['top5']:.1%}")
    print(f"Wrote {args.out} and {args.predictions_out}")


if __name__ == "__main__":
    main()
