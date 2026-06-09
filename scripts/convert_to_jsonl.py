"""
Step 3: Convert to JSONL
========================
Deduplicates cleaned data, resolves conflicting labels, and converts to
JSONL format with leakage-free train/val/test splits.

Leakage prevention: every text appears exactly once in the dataset after
deduplication, so no point name can land in more than one split.

Conflict resolution: texts mapped to more than one target keep the
majority label (by row count); ties are dropped. All conflicts are
logged to data/label_conflicts.csv for human review.

Manual overrides: if data/label_overrides.csv exists, its decisions take
precedence over majority-vote resolution. To create it, edit the
`resolution` column of label_conflicts.csv (set the correct label, or
DROP to exclude the text) and save it as data/label_overrides.csv.
Columns: `text` plus either `target` or `resolution`. Texts must be in
normalized form (lowercase, space-separated), as they appear in the
conflicts report.

Input:  data/cleaned_data.csv [+ data/label_overrides.csv]
Output: data/train.jsonl, data/validation.jsonl, data/test.jsonl,
        data/label_mapping.json, data/dataset_summary.json,
        data/label_conflicts.csv
"""

import pandas as pd
import json
from sklearn.model_selection import train_test_split
from collections import Counter
import os
import sys


DROP_VALUES = {'DROP', 'DROPPED', 'DROPPED (TIE)'}


def load_overrides(path):
    """
    Load manual label overrides as {text: target_or_None}.

    None means the text is explicitly dropped. Accepts either a `target`
    or `resolution` column so an edited label_conflicts.csv works as-is.
    """
    if not os.path.exists(path):
        return {}

    odf = pd.read_csv(path)
    if 'text' not in odf.columns:
        print(f"WARNING: {path} has no 'text' column, ignoring overrides")
        return {}
    label_col = 'target' if 'target' in odf.columns else 'resolution'
    if label_col not in odf.columns:
        print(f"WARNING: {path} needs a 'target' or 'resolution' column, ignoring overrides")
        return {}

    overrides = {}
    for row in odf.itertuples():
        text = str(getattr(row, 'text')).strip()
        label = str(getattr(row, label_col)).strip()
        if not text or not label or label.lower() == 'nan':
            continue
        overrides[text] = None if label.upper() in DROP_VALUES else label
    return overrides


def deduplicate_and_resolve(df, overrides=None):
    """
    Collapse the dataframe to one row per unique text.

    Manual overrides win over majority-vote resolution. Returns
    (deduped_df, conflicts_df) where conflicts_df describes every text
    that had more than one distinct target.
    """
    overrides = overrides or {}
    counts = df.groupby(['text', 'target']).size().reset_index(name='count')

    resolved_rows = []
    conflict_rows = []

    for text, group in counts.groupby('text'):
        if text in overrides:
            target = overrides[text]
            if len(group) > 1:
                conflict_rows.append({
                    'text': text,
                    'targets': ' | '.join(f"{r.target} ({r.count})" for r in group.itertuples()),
                    'resolution': f"{target if target is not None else 'DROPPED'} (override)",
                })
            if target is not None:
                resolved_rows.append({'text': text, 'target': target})
            continue

        if len(group) == 1:
            resolved_rows.append({'text': text, 'target': group.iloc[0]['target']})
            continue

        # Conflicting labels: majority wins, ties are dropped
        max_count = group['count'].max()
        winners = group[group['count'] == max_count]
        resolution = winners.iloc[0]['target'] if len(winners) == 1 else 'DROPPED (tie)'

        conflict_rows.append({
            'text': text,
            'targets': ' | '.join(f"{r.target} ({r.count})" for r in group.itertuples()),
            'resolution': resolution,
        })
        if len(winners) == 1:
            resolved_rows.append({'text': text, 'target': resolution})

    # Overrides for texts absent from the source data are likely typos
    # (or texts that were not normalized) -- warn but still honor them
    known_texts = set(counts['text'])
    for text, target in overrides.items():
        if text not in known_texts:
            print(f"WARNING: override text not found in data (typo or not normalized?): '{text}'")
            if target is not None:
                resolved_rows.append({'text': text, 'target': target})

    return pd.DataFrame(resolved_rows), pd.DataFrame(conflict_rows)


def convert_to_jsonl(
    input_csv='data/cleaned_data.csv',
    output_dir='data',
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    random_seed=42
):
    """
    Convert CSV to JSONL format with deduplication and train/val/test split.
    """

    os.makedirs(output_dir, exist_ok=True)

    try:
        print(f"Loading {input_csv}...")
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Error: '{input_csv}' not found.")
        sys.exit(1)

    print(f"Loaded: {len(df)} rows, {len(df.columns)} columns")

    # Strip whitespace and remove empty rows
    df['text'] = df['text'].astype(str).str.strip()
    df['target'] = df['target'].astype(str).str.strip()
    original_len = len(df)
    df = df.dropna(subset=['text', 'target'])
    df = df[df['text'].str.len() > 0]
    df = df[df['target'].str.len() > 0]
    if original_len - len(df) > 0:
        print(f"Removed {original_len - len(df)} rows with empty values")

    # Manual label overrides take precedence over majority-vote resolution
    overrides_path = os.path.join(output_dir, 'label_overrides.csv')
    overrides = load_overrides(overrides_path)
    if overrides:
        n_dropped = sum(1 for t in overrides.values() if t is None)
        print(f"Loaded {len(overrides)} manual overrides from {overrides_path} "
              f"({len(overrides) - n_dropped} relabeled, {n_dropped} dropped)")

    # Deduplicate: one row per unique text, conflicts resolved by majority
    total_rows = len(df)
    df, conflicts = deduplicate_and_resolve(df, overrides)
    print(f"Deduplicated: {total_rows} rows -> {len(df)} unique texts")
    conflicts_path = os.path.join(output_dir, 'label_conflicts.csv')
    if len(conflicts) > 0:
        overridden = conflicts['resolution'].str.endswith('(override)').sum()
        dropped = (conflicts['resolution'] == 'DROPPED (tie)').sum()
        print(f"Conflicting labels: {len(conflicts)} texts "
              f"({overridden} resolved by override, "
              f"{len(conflicts) - dropped - overridden} resolved by majority, {dropped} dropped)")
        conflicts.to_csv(conflicts_path, index=False)
        print(f"  Conflict report saved: {conflicts_path}")
    elif os.path.exists(conflicts_path):
        os.remove(conflicts_path)

    # Classes need at least 3 samples to stratify across 3 splits
    class_counts = Counter(df['target'])
    rare = {label for label, count in class_counts.items() if count < 3}
    if rare:
        print(f"Removed {len(rare)} classes with fewer than 3 unique texts: {sorted(rare)}")
        df = df[~df['target'].isin(rare)]
        class_counts = Counter(df['target'])
    print(f"Unique classes: {len(class_counts)}")

    # Create label to ID mapping
    unique_labels = sorted(df['target'].unique())
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    # Split data (texts are unique, so splits cannot share a text)
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        random_state=random_seed,
        stratify=df['target']
    )

    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio_adjusted,
        random_state=random_seed,
        stratify=train_val_df['target']
    )

    # Sanity check: no text may appear in more than one split
    train_texts = set(train_df['text'])
    val_texts = set(val_df['text'])
    test_texts = set(test_df['text'])
    assert not (train_texts & val_texts), "Leakage: train/val share texts"
    assert not (train_texts & test_texts), "Leakage: train/test share texts"
    assert not (val_texts & test_texts), "Leakage: val/test share texts"
    print("Leakage check passed: no text appears in more than one split")

    print(f"\nSplit sizes:")
    print(f"  Train:      {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Validation: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test:       {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")

    # Write JSONL files
    def write_jsonl(data, filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            for _, row in data.iterrows():
                record = {
                    'text': row['text'],
                    'label': row['target'],
                    'label_id': label2id[row['target']]
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(f"  Saved: {filepath}")

    print(f"\nWriting JSONL files...")
    write_jsonl(train_df, os.path.join(output_dir, 'train.jsonl'))
    write_jsonl(val_df, os.path.join(output_dir, 'validation.jsonl'))
    write_jsonl(test_df, os.path.join(output_dir, 'test.jsonl'))

    # Save label mappings
    label_mapping_path = os.path.join(output_dir, 'label_mapping.json')
    with open(label_mapping_path, 'w', encoding='utf-8') as f:
        json.dump({
            'label2id': label2id,
            'id2label': {str(k): v for k, v in id2label.items()},
            'num_labels': len(unique_labels)
        }, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {label_mapping_path}")

    # Create summary file
    summary_path = os.path.join(output_dir, 'dataset_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            'total_samples': len(df),
            'source_rows': total_rows,
            'duplicate_rows_removed': total_rows - len(df) - len(conflicts),
            'conflicting_texts': len(conflicts),
            'num_classes': len(unique_labels),
            'train_samples': len(train_df),
            'validation_samples': len(val_df),
            'test_samples': len(test_df),
            'class_distribution': dict(class_counts.most_common()),
            'text_length_stats': {
                'min': int(df['text'].str.len().min()),
                'max': int(df['text'].str.len().max()),
                'mean': float(df['text'].str.len().mean()),
                'median': float(df['text'].str.len().median())
            }
        }, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {summary_path}")

    print(f"\nConversion complete!")

    return train_df, val_df, test_df, label2id


if __name__ == "__main__":
    convert_to_jsonl()
