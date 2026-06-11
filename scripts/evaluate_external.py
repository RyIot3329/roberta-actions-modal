"""
Evaluate the fine-tuned model against an external labeled dataset.
==================================================================

Pulls the model from Hugging Face (token from .env), extracts point names,
normalizes them exactly like the training pipeline, canonicalizes gold
labels (eo66 numbered variants + target_audit merges/renames) and scores
predictions. Predictions are passed through the same canonicalizer so
models trained on pre-audit label names get alias credit.

Inputs (auto-detected):
  - data/real_points.csv produced by scripts/extract_real_data.py
    (columns site,name,label_raw,label,rows) -- filter with --site;
    `rows` weights the row-level metrics
  - any CSV with a point-path column and a label column
    (--text-column / --label-column, like the original n4_points.csv)

Usage:
    python scripts/evaluate_external.py --csv data/real_points.csv --site N4-Integ06
    python scripts/evaluate_external.py \
        --csv ~/Downloads/n4_points.csv \
        --text-column "pointPath from BAS" \
        --label-column "EO66 Point"

Outputs output/external_eval_<name>.csv (full predictions) and
output/review_queue_<name>.csv (unique names below the auto-accept
threshold, ordered by row count -- review once, apply to all rows).
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

HF_REPO = "RyIoT33/haystack-autotagging"
BATCH_SIZE = 256


def load_dataset(args):
    """Return a dataframe with point_name, text, label, weight columns."""
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
        source = f"{args.csv}" + (f" [site={args.site}]" if args.site else "")
    else:
        df = df.dropna(subset=[args.text_column, args.label_column])
        df['point_name'] = (df[args.text_column].astype(str)
                            .str.rstrip('/').str.split('/').str[-1])
        df['label'] = df[args.label_column].astype(str).str.strip()
        df['weight'] = 1
        source = args.csv

    df['text'] = df['point_name'].astype(str).map(normalize_text)
    df = df[(df['text'].str.len() > 0) & (df['label'].astype(str).str.len() > 0)]
    print(f"Loaded {source}: {len(df)} records, "
          f"{int(df['weight'].sum())} weighted rows, {df['text'].nunique()} unique texts")
    return df


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on an external dataset")
    parser.add_argument("--csv", default="data/real_points.csv")
    parser.add_argument("--site", default=None,
                        help="Site filter when --csv is a real_points.csv export")
    parser.add_argument("--text-column", default="pointPath from BAS")
    parser.add_argument("--label-column", default="EO66 Point")
    parser.add_argument("--hf-repo", default=HF_REPO)
    parser.add_argument("--threshold", type=float, default=0.999,
                        help="Auto-accept confidence threshold (in-taxonomy row "
                             "precision was 98.7%% at 0.999 in the audit)")
    args = parser.parse_args()

    # Token from .env at repo root
    load_dotenv(Path(__file__).parent.parent / ".env")
    token = os.environ.get("HUGGING_FACE_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: no HUGGING_FACE_TOKEN / HF_TOKEN in .env")
        sys.exit(1)

    df = load_dataset(args)

    # Canonicalize gold labels the same way the training pipeline does
    canon = build_label_canonicalizer('data/eo66.xlsx', 'data/target_audit.csv')
    df["label"] = df["label"].astype(str).str.strip().map(canon)

    print(f"\nDownloading model: {args.hf_repo}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.hf_repo, token=token)
        model = AutoModelForSequenceClassification.from_pretrained(args.hf_repo, token=token)
    except Exception as e:
        print(f"WARNING: Hub fetch failed ({type(e).__name__}); retrying from local cache")
        tokenizer = AutoTokenizer.from_pretrained(args.hf_repo, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(args.hf_repo, local_files_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    # Alias credit: a model trained before the audit merges predicts the old
    # names; canonicalizing its outputs scores them fairly
    id2label = {int(k): canon(v) for k, v in model.config.id2label.items()}
    model_labels = set(id2label.values())
    print(f"Model loaded on {device}: {len(model_labels)} classes (after canonicalization)")

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

    # Predict each unique text once, then map back to rows
    unique_texts = sorted(df["text"].unique())
    print(f"\nRunning inference on {len(unique_texts)} unique point names, batch={BATCH_SIZE}...")

    pred_label, pred_conf = {}, {}
    with torch.no_grad():
        for i in range(0, len(unique_texts), BATCH_SIZE):
            batch = unique_texts[i:i + BATCH_SIZE]
            inputs = tokenizer(batch, padding=True, truncation=True,
                               max_length=32, return_tensors="pt").to(device)
            probs = torch.softmax(model(**inputs).logits, dim=1)
            confs, ids = probs.max(dim=1)
            for text, pid, conf in zip(batch, ids.tolist(), confs.tolist()):
                pred_label[text] = id2label[pid]
                pred_conf[text] = conf

    df["predicted"] = df["text"].map(pred_label)
    df["confidence"] = df["text"].map(pred_conf)
    df["correct"] = df["predicted"] == df["label"]

    # ----- Results -----
    def wacc(frame):
        return (frame["correct"] * frame["weight"]).sum() / frame["weight"].sum() \
            if frame["weight"].sum() else float("nan")

    total_w = df["weight"].sum()
    known = df[~unknown_mask]
    uniq = df.drop_duplicates("text")

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Row-weighted accuracy (all rows):   {wacc(df):.2%}  ({int(total_w)} rows)")
    print(f"  on labels the model knows:        {wacc(known):.2%}  "
          f"({known['weight'].sum() / total_w:.1%} of rows)")
    print(f"Unique-text accuracy:               {uniq['correct'].mean():.2%}  ({len(uniq)} texts)")
    print(f"  on labels the model knows:        "
          f"{uniq[~uniq['label'].isin(unknown_labels)]['correct'].mean():.2%}")

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
    df[["point_name", "text", "label", "predicted", "confidence", "correct", "weight"]] \
        .to_csv(out_path, index=False)
    print(f"\nFull predictions saved to: {out_path}")

    queue = (lo.groupby("text")
             .agg(rows=("weight", "sum"), predicted=("predicted", "first"),
                  confidence=("confidence", "first"),
                  example_name=("point_name", "first"))
             .sort_values("rows", ascending=False)
             .reset_index())
    queue_path = f"output/review_queue_{name}.csv"
    queue.to_csv(queue_path, index=False)
    print(f"Review queue (one entry per unique name, by row count): {queue_path}")


if __name__ == "__main__":
    main()
