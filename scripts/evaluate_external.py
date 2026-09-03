"""
Evaluate the fine-tuned model against an external labeled dataset.
==================================================================

Pulls the model from Hugging Face (token from .env), extracts point names,
normalizes them exactly like the training pipeline, canonicalizes gold
labels (eo66 numbered variants + target_audit merges/renames) and scores
predictions. Predictions are passed through the same canonicalizer so
models trained on pre-audit label names get alias credit.

Name + description ensemble: when a record carries a description (e.g.
BACnet's description property), the model predicts BOTH the normalized
name and the normalized description, and the higher-confidence prediction
wins. Confidences are temperature-calibrated (the model stores T fitted on
validation), so the two views are comparable probabilities. This rescues
semantically empty names like "AV 00" whose description ("Heating Signal")
carries the meaning, without changing the model's input contract -- the
production edge tagger should apply the same rule.

Inputs (auto-detected):
  - data/real_points.csv produced by scripts/extract_real_data.py
    (columns site,name,description,label_raw,label,rows) -- filter with
    --site; `rows` weights the row-level metrics
  - any CSV with a point-path column and a label column
    (--text-column / --label-column / optional --desc-column)

Usage:
    python scripts/evaluate_external.py --csv data/real_points.csv --site Motorola_Points
    python scripts/evaluate_external.py \
        --csv ~/Downloads/n4_points.csv \
        --text-column "pointPath from BAS" \
        --label-column "EO66 Point"

Outputs output/external_eval_<name>.csv (full predictions with per-view
confidences and the winning source) and output/review_queue_<name>.csv
(unique names below the auto-accept threshold, ordered by row count --
review once, apply to all rows).
"""

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from dotenv import load_dotenv
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Reuse the exact normalization + label canonicalization of the pipeline
sys.path.insert(0, str(Path(__file__).parent))
from clean_data import normalize_text
from convert_to_jsonl import build_label_canonicalizer
import metrics_core as mc

HF_REPO = "RyIoT33/haystack-autotagging"
BATCH_SIZE = 256


def load_dataset(args):
    """Return a dataframe with point_name, text, desc_text, label, weight."""
    df = pd.read_csv(args.csv, low_memory=False)

    if {'site', 'name', 'label'}.issubset(df.columns):  # real_points.csv
        if args.site:
            df = df[df['site'] == args.site]
            if df.empty:
                print(f"ERROR: no rows for site '{args.site}' "
                      f"(available: {sorted(pd.read_csv(args.csv)['site'].unique())})")
                sys.exit(1)
        df = df.rename(columns={'name': 'point_name'})
        df['weight'] = df['rows'] if 'rows' in df.columns else 1
        desc = df['description'] if 'description' in df.columns else ''
        source = f"{args.csv}" + (f" [site={args.site}]" if args.site else "")
    else:
        df = df.dropna(subset=[args.text_column, args.label_column])
        df['point_name'] = (df[args.text_column].astype(str)
                            .str.rstrip('/').str.split('/').str[-1])
        df['label'] = df[args.label_column].astype(str).str.strip()
        df['weight'] = 1
        desc = df[args.desc_column] if args.desc_column and args.desc_column in df.columns else ''
        source = args.csv

    df['text'] = df['point_name'].astype(str).map(normalize_text)
    df['desc_text'] = pd.Series(desc, index=df.index).fillna('').astype(str).map(
        lambda s: normalize_text(s) if s.strip() else '')
    df = df[(df['text'].str.len() > 0) & (df['label'].astype(str).str.len() > 0)]
    n_desc = (df['desc_text'].str.len() > 0).sum()
    print(f"Loaded {source}: {len(df)} records, {int(df['weight'].sum())} weighted rows, "
          f"{df['text'].nunique()} unique names, {n_desc} records with descriptions")
    return df


def build_accept_sets(args, canon):
    """Lenient credit: labels that TRAIN sites (every site other than the one
    being scored and the frozen held-out sites) used for the identical
    normalized name, plus approved label equivalences."""
    from convert_to_jsonl import VAL_SITES, TEST_SITES, load_equivalences
    accept = {}
    try:
        rp = pd.read_csv("data/real_points.csv")
    except FileNotFoundError:
        return accept
    excluded = set(VAL_SITES) | set(TEST_SITES) | ({args.site} if args.site else set())
    rp = rp[~rp["site"].isin(excluded)]
    rp["text"] = rp["name"].astype(str).map(normalize_text)
    rp["label"] = rp["label"].astype(str).str.strip().map(canon)
    for text, label in zip(rp["text"], rp["label"]):
        accept.setdefault(text, set()).add(label)
    equiv = load_equivalences("data/label_equivalences.csv", canon)
    if equiv:
        for text in list(accept):
            for label in list(accept[text]):
                accept[text] |= equiv.get(label, set())
    return accept


def run_probe(args):
    """Score the frozen 110-row live-service probe as a regression suite.

    Each row carries the raw name, its normalized form, a pre-registered
    gold, `acceptables` (pipe-separated twins/upgrades credited on review)
    and the verdict the deployed model earned. Reports the verdict transition
    matrix against that stored verdict; any correct* -> wrong-* move is a
    regression. OOD rows are excluded from accuracy and report abstention at
    the model's stored acceptance threshold.
    """
    load_dotenv(Path(__file__).parent.parent / ".env")
    token = os.environ.get("HUGGING_FACE_TOKEN") or os.environ.get("HF_TOKEN")
    probe = pd.read_csv(args.probe)
    needed = {"name", "gold", "verdict"}
    if not needed.issubset(probe.columns):
        print(f"ERROR: probe file lacks columns {sorted(needed - set(probe.columns))}")
        sys.exit(1)
    canon = build_label_canonicalizer("data/eo66.xlsx", "data/target_audit.csv")
    probe["text"] = probe["name"].astype(str).map(normalize_text)
    probe["gold_c"] = probe["gold"].fillna("").astype(str).str.strip().map(lambda g: canon(g) if g else "")
    equiv = None
    try:
        from convert_to_jsonl import load_equivalences
        equiv = load_equivalences("data/label_equivalences.csv", canon)
    except Exception:  # noqa: BLE001
        equiv = {}

    def accept_set(row):
        acc = {row["gold_c"]} if row["gold_c"] else set()
        raw = row.get("acceptables")
        if isinstance(raw, str) and raw.strip():
            acc |= {canon(a.strip()) for a in raw.split("|") if a.strip()}
        # A stored correct-upgrade / correct-twin verdict means the reviewer
        # accepted the deployed model's prediction: it is part of the gold set
        stored_pred = row.get("predicted")
        if (str(row.get("verdict", "")).startswith("correct") and isinstance(stored_pred, str)
                and stored_pred.strip()):
            acc.add(canon(stored_pred.strip()))
        for label in list(acc):
            acc |= equiv.get(label, set())
        return acc

    probe["accept"] = probe.apply(accept_set, axis=1)

    try:
        if args.offline:
            raise RuntimeError("offline requested")
        tokenizer = AutoTokenizer.from_pretrained(args.hf_repo, token=token)
        model = AutoModelForSequenceClassification.from_pretrained(args.hf_repo, token=token)
    except Exception as e:  # noqa: BLE001
        if not args.offline:
            print(f"WARNING: Hub fetch failed ({type(e).__name__}); using local cache")
        tokenizer = AutoTokenizer.from_pretrained(args.hf_repo, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(args.hf_repo, local_files_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    id2label = {int(k): canon(v) for k, v in model.config.id2label.items()}
    temperature = float(getattr(model.config, "calibration_temperature", None) or 1.0)
    tau = getattr(model.config, "acceptance_threshold", None)
    tau = float(tau) if tau else args.threshold
    texts = probe["text"].tolist()
    preds, confs = [], []
    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=32,
                               return_tensors="pt").to(device)
            probs = torch.softmax(model(**inputs).logits / temperature, dim=1)
            c, ids = probs.max(dim=1)
            preds += [id2label[j] for j in ids.tolist()]
            confs += c.tolist()
    probe["stored_predicted"] = probe.get("predicted")
    probe["predicted"] = preds
    probe["confidence"] = confs

    def new_verdict(row):
        old = str(row["verdict"])
        if old == "ood":
            return "ood-abstained" if row["confidence"] < tau else "ood-tagged"
        if row["predicted"] == row["gold_c"]:
            return "correct"
        if row["predicted"] in row["accept"]:
            return "correct-twin"
        return "wrong"

    probe["new_verdict"] = probe.apply(new_verdict, axis=1)
    old_ok = probe["verdict"].astype(str).str.startswith("correct")
    new_ok = probe["new_verdict"].str.startswith("correct")
    scoreable = probe["verdict"].astype(str) != "ood"
    regressions = probe[scoreable & old_ok & ~new_ok]
    fixes = probe[scoreable & ~old_ok & new_ok]
    print("=" * 60)
    print(f"PROBE {args.probe}: {len(probe)} rows, model {args.hf_repo} (T={temperature:.3f}, tau={tau:.3f})")
    print("=" * 60)
    print(f"Scoreable rows (non-OOD): {int(scoreable.sum())}  "
          f"stored correct* {int((scoreable & old_ok).sum())} -> now correct* {int((scoreable & new_ok).sum())} "
          f"({(scoreable & new_ok).sum() / scoreable.sum():.1%})")
    print(f"OOD rows: {int((~scoreable).sum())}  abstained at tau: "
          f"{int((probe['new_verdict'] == 'ood-abstained').sum())}")
    trans = Counter(zip(probe["verdict"].astype(str), probe["new_verdict"]))
    print("Verdict transitions (stored -> now):")
    for (a, b), n in sorted(trans.items()):
        print(f"  {a:<16} -> {b:<14} {n}")
    if len(regressions):
        print(f"\nREGRESSIONS ({len(regressions)}):")
        for r in regressions.itertuples():
            print(f"  {r.name!r} ({r.text}): gold {r.gold_c}, now {r.predicted} @ {r.confidence:.2f}")
    if len(fixes):
        print(f"\nFixed ({len(fixes)}):")
        for r in fixes.itertuples():
            print(f"  {r.name!r} ({r.text}): gold {r.gold_c}, now {r.predicted} @ {r.confidence:.2f}")
    os.makedirs("output", exist_ok=True)
    out = f"output/probe_eval_{args.hf_repo.split('/')[-1]}.csv"
    probe.drop(columns=["accept"]).to_csv(out, index=False)
    print(f"\nSaved: {out}")
    return 1 if len(regressions) else 0


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on an external dataset")
    parser.add_argument("--csv", default="data/real_points.csv")
    parser.add_argument("--site", default=None,
                        help="Site filter when --csv is a real_points.csv export")
    parser.add_argument("--text-column", default="pointPath from BAS")
    parser.add_argument("--label-column", default="EO66 Point")
    parser.add_argument("--desc-column", default=None,
                        help="Optional description column for legacy CSV inputs")
    parser.add_argument("--hf-repo", default=HF_REPO)
    parser.add_argument("--threshold", type=float, default=0.8,
                        help="Auto-accept confidence threshold. Tune on the "
                             "validation site per model version: smoothed+"
                             "calibrated models concentrate below ~0.9 (the "
                             "2026-06-12 run: 0.8 accepts ~75%% of rows at "
                             "~90%% precision), while legacy raw-softmax "
                             "models needed 0.999")
    parser.add_argument("--no-ensemble", action="store_true",
                        help="Score names only, ignoring descriptions")
    parser.add_argument("--lenient", action="store_true",
                        help="Also report lenient accuracy: credit any label a TRAIN site "
                             "used for the identical normalized name (data/real_points.csv, "
                             "sites outside --site/val/test) plus approved "
                             "data/label_equivalences.csv rows")
    parser.add_argument("--probe", nargs="?", const="output/live_probe_20260713_rows.csv",
                        default=None, metavar="CSV",
                        help="Score the frozen live-probe regression suite instead of a site "
                             "export (columns: name,normalized,gold,acceptables,verdict,...)")
    parser.add_argument("--offline", action="store_true",
                        help="Load the model from the local HF cache only")
    args = parser.parse_args()

    if args.probe:
        return run_probe(args)

    # Token from .env at repo root
    load_dotenv(Path(__file__).parent.parent / ".env")
    token = os.environ.get("HUGGING_FACE_TOKEN") or os.environ.get("HF_TOKEN")
    if not token and not args.offline:
        print("ERROR: no HUGGING_FACE_TOKEN / HF_TOKEN in .env (or pass --offline)")
        sys.exit(1)

    df = load_dataset(args)
    if args.no_ensemble:
        df['desc_text'] = ''

    # Canonicalize gold labels the same way the training pipeline does
    canon = build_label_canonicalizer('data/eo66.xlsx', 'data/target_audit.csv')
    df["label"] = df["label"].astype(str).str.strip().map(canon)

    print(f"\nDownloading model: {args.hf_repo}")
    try:
        if args.offline:
            raise RuntimeError("offline requested")
        tokenizer = AutoTokenizer.from_pretrained(args.hf_repo, token=token)
        model = AutoModelForSequenceClassification.from_pretrained(args.hf_repo, token=token)
    except Exception as e:
        print(f"WARNING: Hub fetch failed ({type(e).__name__}); retrying from local cache "
              f"-- the cached snapshot may be STALE")
        tokenizer = AutoTokenizer.from_pretrained(args.hf_repo, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(args.hf_repo, local_files_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    # Alias credit: a model trained before the audit merges predicts the old
    # names; canonicalizing its outputs scores them fairly
    id2label = {int(k): canon(v) for k, v in model.config.id2label.items()}
    model_labels = set(id2label.values())
    # Calibration: models trained with temperature scaling carry T in their
    # config; dividing logits by it makes confidences honest probabilities
    # (and makes name-vs-description confidences comparable for the ensemble)
    temperature = float(getattr(model.config, "calibration_temperature", None) or 1.0)
    print(f"Model loaded on {device}: {len(model_labels)} classes (after canonicalization), "
          f"calibration temperature {temperature:.3f}"
          + ("" if temperature != 1.0 else " (none stored; raw softmax)"))

    # Dataset labels the model has never seen are impossible to get right
    data_labels = set(df["label"])
    unknown_labels = data_labels - model_labels
    unknown_mask = df["label"].isin(unknown_labels)
    print(f"\nLabels in dataset: {len(data_labels)} "
          f"({len(data_labels & model_labels)} known to model, {len(unknown_labels)} not)")
    if unknown_labels:
        w = df.loc[unknown_mask, 'weight'].sum() / df['weight'].sum()
        print(f"Rows with labels outside the model's taxonomy: {w:.1%} of weighted rows "
              f"-- counted as errors, listed separately below")

    # Predict every unique view (names and descriptions) once
    unique_texts = sorted(set(df["text"]) | {t for t in df["desc_text"] if t})
    print(f"\nRunning inference on {len(unique_texts)} unique strings "
          f"(names + descriptions), batch={BATCH_SIZE}...")

    pred_label, pred_conf = {}, {}
    with torch.no_grad():
        for i in range(0, len(unique_texts), BATCH_SIZE):
            batch = unique_texts[i:i + BATCH_SIZE]
            inputs = tokenizer(batch, padding=True, truncation=True,
                               max_length=32, return_tensors="pt").to(device)
            probs = torch.softmax(model(**inputs).logits / temperature, dim=1)
            confs, ids = probs.max(dim=1)
            for text, pid, conf in zip(batch, ids.tolist(), confs.tolist()):
                pred_label[text] = id2label[pid]
                pred_conf[text] = conf

    df["pred_name"] = df["text"].map(pred_label)
    df["conf_name"] = df["text"].map(pred_conf)
    df["pred_desc"] = df["desc_text"].map(lambda t: pred_label.get(t))
    df["conf_desc"] = df["desc_text"].map(lambda t: pred_conf.get(t, 0.0))

    # Max-confidence ensemble: the more certain view wins
    use_desc = df["conf_desc"] > df["conf_name"]
    df["predicted"] = df["pred_name"].where(~use_desc, df["pred_desc"])
    df["confidence"] = df["conf_name"].where(~use_desc, df["conf_desc"])
    df["source"] = pd.Series('name', index=df.index).where(~use_desc, 'description')
    df["correct"] = df["predicted"] == df["label"]

    # ----- Results -----
    def wacc(frame, col="correct"):
        return (frame[col] * frame["weight"]).sum() / frame["weight"].sum() \
            if frame["weight"].sum() else float("nan")

    total_w = df["weight"].sum()
    known = df[~unknown_mask]
    has_desc = df[df["desc_text"].str.len() > 0]
    uniq = df.drop_duplicates("text")

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    df["correct_name"] = df["pred_name"] == df["label"]
    print(f"Name-only accuracy (row-weighted):  {wacc(df, 'correct_name'):.2%}")
    if len(has_desc):
        has_desc = has_desc.assign(correct_desc=has_desc["pred_desc"] == has_desc["label"])
        print(f"Description-only, where present:    {wacc(has_desc, 'correct_desc'):.2%}  "
              f"({has_desc['weight'].sum() / total_w:.1%} of rows have descriptions)")
        print(f"Max-confidence ensemble:            {wacc(df):.2%}  <-- headline")
        src = df.loc[df['desc_text'].str.len() > 0, 'source'].value_counts(normalize=True)
        print(f"  ensemble chose description for {src.get('description', 0):.0%} "
              f"of described records")
    else:
        print(f"Ensemble == name-only (no descriptions in this dataset): {wacc(df):.2%}")
    print(f"  on labels the model knows:        {wacc(known):.2%}  "
          f"({known['weight'].sum() / total_w:.1%} of rows)")
    print(f"Unique-name accuracy (ensemble):    {uniq['correct'].mean():.2%}  ({len(uniq)} names)")
    if args.lenient:
        accept = build_accept_sets(args, canon)
        df["correct_lenient"] = [
            pred == gold or pred in accept.get(text, set())
            for pred, gold, text in zip(df["predicted"], df["label"], df["text"])]
        uniq = df.drop_duplicates("text")
        print(f"Lenient accuracy (accept sets):     row-weighted {wacc(df, 'correct_lenient'):.2%}, "
              f"unique-name {uniq['correct_lenient'].mean():.2%}  "
              f"({sum(1 for t in uniq['text'] if len(accept.get(t, ())) > 0)} names have alternatives)")

    hi = df[df["confidence"] >= args.threshold]
    lo = df[df["confidence"] < args.threshold]
    print(f"\nTriage at threshold {args.threshold}:")
    if len(hi):
        print(f"  auto-accept: {hi['weight'].sum() / total_w:.1%} of rows, "
              f"precision {wacc(hi):.2%} "
              f"(in-taxonomy precision {wacc(hi[~hi['label'].isin(unknown_labels)]):.2%})")
    if len(lo):
        lo_uniq = lo.drop_duplicates('text')
        print(f"  review queue: {len(lo_uniq)} unique names covering "
              f"{lo['weight'].sum() / total_w:.1%} of rows")

    wrong_uniq = uniq[~uniq["correct"]]
    confusions = Counter(zip(wrong_uniq["label"], wrong_uniq["predicted"]))
    print(f"\nTop confusions (unique point names, actual -> predicted):")
    for (actual, predicted), count in confusions.most_common(15):
        flag = " [label unknown to model]" if actual in unknown_labels else ""
        print(f"  {count:3d}x  {actual} -> {predicted}{flag}")

    if unknown_labels:
        unknown_w = (df[unknown_mask].groupby('label')['weight'].sum()
                     .sort_values(ascending=False))
        print(f"\nDataset labels the model cannot predict (taxonomy gap):")
        for label, count in unknown_w.head(15).items():
            print(f"  {int(count):5d} rows: {label}")

    # ----- Save predictions + deduplicated review queue -----
    name = args.site or Path(args.csv).stem
    os.makedirs("output", exist_ok=True)
    out_path = f"output/external_eval_{name}.csv"
    df[["point_name", "text", "desc_text", "label",
        "pred_name", "conf_name", "pred_desc", "conf_desc",
        "predicted", "confidence", "source", "correct", "weight"]].to_csv(
        out_path, index=False)
    print(f"\nFull predictions saved to: {out_path}")

    queue = (lo.groupby("text")
             .agg(rows=("weight", "sum"), predicted=("predicted", "first"),
                  confidence=("confidence", "first"), source=("source", "first"),
                  example_name=("point_name", "first"),
                  example_description=("desc_text", "first"))
             .sort_values("rows", ascending=False)
             .reset_index())
    queue_path = f"output/review_queue_{name}.csv"
    queue.to_csv(queue_path, index=False)
    print(f"Review queue (one entry per unique name, by row count): {queue_path}")


if __name__ == "__main__":
    sys.exit(main() or 0)
