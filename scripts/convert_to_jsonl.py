"""
Step 3: Convert to JSONL (site-grouped split)
=============================================
Builds the train/validation/test JSONL files for fine-tuning.

The split is by SITE, not by row: entire real sites are held out for
validation and test so the metrics measure cross-site generalization --
the deployment scenario of tagging a building the model has never seen.
The synthetic template data plus the remaining real sites form the
training pool.

    train       synthetic (cleaned_data.csv) + real training sites
    validation  --val-sites   (drives early stopping / model selection)
    test        --test-sites  (frozen benchmark: never tune against it)

Labels are canonicalized before anything else:
  - eo66 numbered variants collapse to their base definition
    (heatingStage01 -> heatingStage) via eo66's own regex column
  - data/target_audit.csv merge/rename decisions are applied
    (dischargeFanEnable -> dischargeFan, zoneTempMaxSP -> zoneTempMaxSp, ...)

Training-pool dedup: one row per unique text. Conflicting labels are
resolved by weighted majority -- each synthetic row counts 1, each real
occurrence counts 1 (the `rows` column of real_points.csv), so real-site
evidence outweighs synthetic templates. Ties are dropped. All conflicts
are logged to data/label_conflicts.csv.

Manual overrides: data/label_overrides.csv decisions take precedence over
majority vote (same format as before: `text` plus `target`/`resolution`,
DROP to exclude). Texts are re-normalized on load, so the file keeps
working when normalization rules change.

Held-out sites keep ALL their unique texts. Texts whose (site-majority)
gold label is outside the training label space cannot be scored by the
model; they are written to data/{validation,test}_uncovered.jsonl and
counted as coverage in data/dataset_summary.json. The covered files are
what finetune.py consumes, so its metrics mean "accuracy on labels the
model can express" -- read them together with the coverage figure.
Texts that also appear in training are KEPT in val/test (recurring names
are deployment reality) and flagged seen_in_train for sliced reporting.

Input:  data/cleaned_data.csv, data/real_points.csv
        [+ data/label_overrides.csv, data/target_audit.csv, data/eo66.xlsx]
Output: data/train.jsonl, data/validation.jsonl, data/test.jsonl,
        data/validation_uncovered.jsonl, data/test_uncovered.jsonl,
        data/label_mapping.json, data/dataset_summary.json,
        data/label_conflicts.csv
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clean_data import normalize_text
from extract_real_data import build_canonicalizer

VAL_SITES = ['N4-Integ05']
TEST_SITES = ['N4-Integ06', 'Motorola_Points']

DROP_VALUES = {'DROP', 'DROPPED', 'DROPPED (TIE)'}
AUDIT_APPLY_STATUSES = {'merge_case', 'rename', 'merge_dup'}


def build_label_canonicalizer(eo66_path, audit_path):
    """Compose eo66 regex canonicalization with target_audit merge/renames."""
    eo66_canon, _ = build_canonicalizer(eo66_path)

    audit_map = {}
    if os.path.exists(audit_path):
        audit = pd.read_csv(audit_path)
        for row in audit.itertuples():
            target = str(getattr(row, 'target', '')).strip()
            proposed = getattr(row, 'proposed_target', '')
            status = str(getattr(row, 'status', '')).strip()
            if (status in AUDIT_APPLY_STATUSES and target
                    and isinstance(proposed, str) and proposed.strip()):
                audit_map[target] = proposed.strip()
        if audit_map:
            print(f"Applying {len(audit_map)} target_audit merges/renames: {audit_map}")

    def canon(label: str) -> str:
        label = audit_map.get(label, label)
        return eo66_canon(label)

    return canon


def load_overrides(path, canon):
    """
    Load manual label overrides as {normalized_text: target_or_None}.

    None means the text is explicitly dropped. Accepts either a `target`
    or `resolution` column so an edited label_conflicts.csv works as-is.
    Texts are re-normalized so the file survives normalization changes.
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
        text = normalize_text(str(getattr(row, 'text')))
        label = str(getattr(row, label_col)).strip()
        if not text or not label or label.lower() == 'nan':
            continue
        value = None if label.upper() in DROP_VALUES else canon(label)
        if text in overrides and overrides[text] != value:
            print(f"WARNING: override collision after re-normalization for '{text}': "
                  f"{overrides[text]} vs {value}; keeping the latter")
        overrides[text] = value
    return overrides


def resolve_training_pool(weights, overrides):
    """
    Collapse {text: {label: weight}} to one label per text.

    Manual overrides win; otherwise weighted majority; ties are dropped.
    Returns (resolved {text: label}, conflicts list for the CSV report).
    """
    resolved = {}
    conflicts = []

    for text, label_weights in weights.items():
        if text in overrides:
            target = overrides[text]
            if len(label_weights) > 1:
                conflicts.append({
                    'text': text,
                    'targets': ' | '.join(f"{l} ({w})" for l, w in
                                          sorted(label_weights.items(), key=lambda x: -x[1])),
                    'resolution': f"{target if target is not None else 'DROPPED'} (override)",
                })
            if target is not None:
                resolved[text] = target
            continue

        if len(label_weights) == 1:
            resolved[text] = next(iter(label_weights))
            continue

        ranked = sorted(label_weights.items(), key=lambda x: -x[1])
        is_tie = ranked[0][1] == ranked[1][1]
        resolution = 'DROPPED (tie)' if is_tie else ranked[0][0]
        conflicts.append({
            'text': text,
            'targets': ' | '.join(f"{l} ({w})" for l, w in ranked),
            'resolution': resolution,
        })
        if not is_tie:
            resolved[text] = ranked[0][0]

    # Overrides for texts absent from the source data are honored but flagged
    known = set(weights)
    for text, target in overrides.items():
        if text not in known:
            print(f"WARNING: override text not found in data (typo or stale?): '{text}'")
            if target is not None:
                resolved[text] = target

    return resolved, conflicts


def site_unique_texts(site_df):
    """Per-text weighted-majority gold label within one held-out site.

    Returns list of (text, label, rows). Texts whose gold label ties are
    genuinely ambiguous and excluded (counted by the caller via len diff).
    """
    agg = defaultdict(Counter)
    totals = Counter()
    for row in site_df.itertuples():
        agg[row.text][row.label] += row.rows
        totals[row.text] += row.rows
    out = []
    for text, label_weights in agg.items():
        ranked = label_weights.most_common()
        if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
            continue  # ambiguous gold label within the site
        out.append((text, ranked[0][0], int(totals[text])))
    return out


def write_jsonl(records, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f"  Saved: {filepath} ({len(records)} records)")


def convert_to_jsonl(
    input_csv='data/cleaned_data.csv',
    real_points_csv='data/real_points.csv',
    output_dir='data',
    val_sites=None,
    test_sites=None,
):
    val_sites = val_sites or VAL_SITES
    test_sites = test_sites or TEST_SITES
    os.makedirs(output_dir, exist_ok=True)

    canon = build_label_canonicalizer(
        os.path.join(output_dir, 'eo66.xlsx'),
        os.path.join(output_dir, 'target_audit.csv'),
    )

    # ----- Synthetic template data -----
    try:
        synth = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Error: '{input_csv}' not found. Run scripts/clean_data.py first.")
        sys.exit(1)
    synth['text'] = synth['text'].astype(str).str.strip()
    synth['target'] = synth['target'].astype(str).str.strip().map(canon)
    synth = synth[(synth['text'].str.len() > 0) & (synth['target'].str.len() > 0)]
    print(f"Synthetic data: {len(synth)} rows, {synth['text'].nunique()} unique texts")

    # ----- Real site data -----
    try:
        real = pd.read_csv(real_points_csv)
    except FileNotFoundError:
        print(f"Error: '{real_points_csv}' not found.\n"
              "Run scripts/extract_real_data.py against data/real_data/ first --\n"
              "the site-grouped split needs real sites for validation and test.")
        sys.exit(1)
    real['text'] = real['name'].astype(str).map(normalize_text)
    real['label'] = real['label'].astype(str).str.strip().map(canon)
    real = real[(real['text'].str.len() > 0) & (real['label'].str.len() > 0)]

    sites = sorted(real['site'].unique())
    missing = [s for s in val_sites + test_sites if s not in sites]
    if missing:
        print(f"Error: held-out sites not in {real_points_csv}: {missing} (have: {sites})")
        sys.exit(1)
    train_sites = [s for s in sites if s not in val_sites + test_sites]
    print(f"Sites -> train: {train_sites}, val: {val_sites}, test: {test_sites}")

    # ----- Training pool: weighted label evidence per text -----
    weights = defaultdict(Counter)
    for row in synth.itertuples():
        weights[row.text][row.target] += 1
    real_train = real[real['site'].isin(train_sites)]
    for row in real_train.itertuples():
        weights[row.text][row.label] += int(row.rows)

    overrides = load_overrides(os.path.join(output_dir, 'label_overrides.csv'), canon)
    if overrides:
        n_dropped = sum(1 for t in overrides.values() if t is None)
        print(f"Loaded {len(overrides)} manual overrides "
              f"({len(overrides) - n_dropped} relabeled, {n_dropped} dropped)")

    resolved, conflicts = resolve_training_pool(weights, overrides)
    print(f"Training pool: {len(weights)} unique texts -> {len(resolved)} after conflict resolution")

    conflicts_path = os.path.join(output_dir, 'label_conflicts.csv')
    if conflicts:
        dropped = sum(1 for c in conflicts if c['resolution'] == 'DROPPED (tie)')
        overridden = sum(1 for c in conflicts if c['resolution'].endswith('(override)'))
        print(f"Conflicting labels: {len(conflicts)} texts "
              f"({overridden} by override, {len(conflicts) - dropped - overridden} by majority, "
              f"{dropped} dropped as ties)")
        pd.DataFrame(conflicts).to_csv(conflicts_path, index=False)
        print(f"  Conflict report saved: {conflicts_path}")
    elif os.path.exists(conflicts_path):
        os.remove(conflicts_path)

    # ----- Label space comes from the training pool -----
    train_items = sorted(resolved.items())
    unique_labels = sorted({label for _, label in train_items})
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    print(f"Classes: {len(unique_labels)}")

    train_records = [
        {'text': text, 'label': label, 'label_id': label2id[label]}
        for text, label in train_items
    ]
    assert len({r['text'] for r in train_records}) == len(train_records), \
        "Training texts must be unique after dedup"

    # ----- Held-out sites -----
    train_texts = set(resolved)

    def build_split(split_sites, name):
        covered, uncovered = [], []
        ambiguous = 0
        for site in split_sites:
            site_df = real[real['site'] == site]
            uniq = site_unique_texts(site_df)
            ambiguous += site_df['text'].nunique() - len(uniq)
            for text, label, rows in uniq:
                record = {
                    'text': text,
                    'label': label,
                    'site': site,
                    'rows': rows,
                    'seen_in_train': text in train_texts,
                }
                if label in label2id:
                    record['label_id'] = label2id[label]
                    covered.append(record)
                else:
                    uncovered.append(record)
        total = len(covered) + len(uncovered)
        seen = sum(1 for r in covered + uncovered if r['seen_in_train'])
        print(f"{name}: {total} unique texts from {split_sites} "
              f"({len(covered)} covered = {len(covered) / total:.1%}, "
              f"{seen} seen in train, {ambiguous} ambiguous-gold dropped)")
        return covered, uncovered, ambiguous

    val_records, val_uncovered, val_ambiguous = build_split(val_sites, 'Validation')
    test_records, test_uncovered, test_ambiguous = build_split(test_sites, 'Test')

    # ----- Write outputs -----
    print("\nWriting JSONL files...")
    write_jsonl(train_records, os.path.join(output_dir, 'train.jsonl'))
    write_jsonl(val_records, os.path.join(output_dir, 'validation.jsonl'))
    write_jsonl(test_records, os.path.join(output_dir, 'test.jsonl'))
    write_jsonl(val_uncovered, os.path.join(output_dir, 'validation_uncovered.jsonl'))
    write_jsonl(test_uncovered, os.path.join(output_dir, 'test_uncovered.jsonl'))

    label_mapping_path = os.path.join(output_dir, 'label_mapping.json')
    with open(label_mapping_path, 'w', encoding='utf-8') as f:
        json.dump({
            'label2id': label2id,
            'id2label': {str(i): l for l, i in label2id.items()},
            'num_labels': len(unique_labels),
        }, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {label_mapping_path}")

    class_counts = Counter(label for _, label in train_items)
    text_lengths = pd.Series([len(t) for t, _ in train_items])

    def split_summary(records, uncovered, ambiguous, split_sites):
        total = len(records) + len(uncovered)
        return {
            'sites': split_sites,
            'unique_texts': total,
            'covered': len(records),
            'coverage_pct': round(100 * len(records) / total, 1) if total else None,
            'rows_total': sum(r['rows'] for r in records + uncovered),
            'seen_in_train': sum(1 for r in records + uncovered if r['seen_in_train']),
            'ambiguous_gold_dropped': ambiguous,
        }

    summary_path = os.path.join(output_dir, 'dataset_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            'split_strategy': 'site-grouped (whole sites held out for val/test)',
            'num_classes': len(unique_labels),
            'train': {
                'unique_texts': len(train_records),
                'synthetic_rows': len(synth),
                'real_sites': train_sites,
                'real_rows_weight': int(real_train['rows'].sum()),
                'conflicting_texts': len(conflicts),
                'classes_with_lt3_texts': sum(1 for c in class_counts.values() if c < 3),
            },
            'validation': split_summary(val_records, val_uncovered, val_ambiguous, val_sites),
            'test': split_summary(test_records, test_uncovered, test_ambiguous, test_sites),
            'class_distribution': dict(class_counts.most_common()),
            'text_length_stats': {
                'min': int(text_lengths.min()),
                'max': int(text_lengths.max()),
                'mean': float(text_lengths.mean()),
                'median': float(text_lengths.median()),
            },
        }, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {summary_path}")

    print("\nConversion complete!")
    return train_records, val_records, test_records, label2id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build site-grouped train/val/test JSONL files')
    parser.add_argument('--input-csv', default='data/cleaned_data.csv')
    parser.add_argument('--real-points', default='data/real_points.csv')
    parser.add_argument('--output-dir', default='data')
    parser.add_argument('--val-sites', default=','.join(VAL_SITES),
                        help='Comma-separated site names held out for validation')
    parser.add_argument('--test-sites', default=','.join(TEST_SITES),
                        help='Comma-separated site names held out for test (frozen)')
    args = parser.parse_args()
    convert_to_jsonl(
        input_csv=args.input_csv,
        real_points_csv=args.real_points,
        output_dir=args.output_dir,
        val_sites=[s for s in args.val_sites.split(',') if s],
        test_sites=[s for s in args.test_sites.split(',') if s],
    )
