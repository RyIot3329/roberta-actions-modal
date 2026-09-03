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
import re
import sys
from collections import Counter, defaultdict

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clean_data import normalize_text
from extract_real_data import build_canonicalizer, build_display_name_index
import math
from augment import augment_pool

VAL_SITES = ['N4-Integ05']
TEST_SITES = ['N4-Integ06', 'Motorola_Points']

DROP_VALUES = {'DROP', 'DROPPED', 'DROPPED (TIE)'}
AUDIT_APPLY_STATUSES = {'merge_case', 'rename', 'merge_dup'}

PREPROCESSING_DEFAULTS = {
    'use_generated_synthetic': True,
    'augment_multiplier': 0.0,
    'augment_seed': 1337,
    'dropout_p': 0.1,
    # Conflict votes: per-site rows are capped then square-rooted so one
    # high-row site cannot outvote a multi-site consensus (0 = legacy raw rows)
    'row_cap': 100,
}


def load_preprocessing_config(path='config/training.yml'):
    """Preprocessing knobs from the training config's `preprocessing:` section.

    Lives there so knob changes retrain via the existing config push trigger
    with full git provenance. Degrades to defaults (augmentation off) when
    the file, section, or pyyaml is unavailable.
    """
    cfg = dict(PREPROCESSING_DEFAULTS)
    try:
        import yaml
    except ImportError:
        print("WARNING: pyyaml not installed -- using preprocessing defaults")
        return cfg
    if not os.path.exists(path):
        return cfg
    with open(path) as f:
        section = (yaml.safe_load(f) or {}).get('preprocessing') or {}
    for key in cfg:
        if key in section and section[key] is not None:
            cfg[key] = section[key]
    return cfg


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
        # Audit decisions apply on both sides of eo66 canonicalization:
        # index stripping can only surface a label's audited form
        # (Blr1_Sts -> BlrSts -> boilerStatus), never consume one.
        label = audit_map.get(label, label)
        label = eo66_canon(label)
        return audit_map.get(label, label)

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


def load_equivalences(path, canon):
    """
    Approved label equivalences for LENIENT scoring (never for training).

    data/label_equivalences.csv columns (drafted by scripts/draft_equivalences.py):
      label_a,label_b,relation,direction,...,status
    relation: same | site_variant (symmetric credit), parent (directional:
    predicting the parent when the child is gold is acceptable), confusable
    (report-only). Only rows with status == approved count.
    Returns {gold_label: set(acceptable predicted labels)}.
    """
    equiv = defaultdict(set)
    if not os.path.exists(path):
        return equiv
    df = pd.read_csv(path)
    needed = {'label_a', 'label_b', 'relation', 'status'}
    if not needed.issubset(df.columns):
        print(f"WARNING: {path} lacks columns {sorted(needed - set(df.columns))}; ignored")
        return equiv
    for row in df.itertuples():
        if str(row.status).strip().lower() != 'approved':
            continue
        a = canon(str(row.label_a).strip())
        b = canon(str(row.label_b).strip())
        relation = str(row.relation).strip().lower()
        direction = str(getattr(row, 'direction', 'both') or 'both').strip().lower()
        if relation in ('same', 'site_variant'):
            equiv[a].add(b)
            equiv[b].add(a)
        elif relation == 'parent':
            if direction in ('both', 'a_to_b'):
                equiv[b].add(a)  # gold b (child) accepts prediction a (parent)
            if direction in ('both', 'b_to_a'):
                equiv[a].add(b)
    return equiv


def build_evidence(synth, real_train, generated=None):
    """
    Per-text label evidence: {text: {label: {'synth': template rows,
    'gen': generated rows, 'sites': Counter{site: real rows}}}}.

    Train-site descriptions are labeled evidence in their own right
    ("Damper Command" is as real as "DmpCmd"); held-out site descriptions
    are never read here -- they feed the eval-time max-confidence ensemble.
    """
    def _empty():
        return {'synth': 0.0, 'gen': 0.0, 'sites': Counter()}
    evidence = defaultdict(lambda: defaultdict(_empty))
    for row in synth.itertuples():
        evidence[row.text][row.target]['synth'] += 1
    for row in real_train.itertuples():
        evidence[row.text][row.label]['sites'][row.site] += int(row.rows)
    stats = {'described_rows': 0}
    if 'description' in real_train.columns:
        described = real_train[real_train['description'].fillna('')
                               .astype(str).str.strip().str.len() > 0].copy()
        if len(described):
            described['desc_text'] = described['description'].astype(str).map(normalize_text)
            described = described[described['desc_text'].str.len() > 0]
            for row in described.itertuples():
                evidence[row.desc_text][row.label]['sites'][row.site] += int(row.rows)
            stats['described_rows'] = len(described)
    if generated is not None:
        for row in generated.itertuples():
            evidence[row.text][row.target]['gen'] += 1
    return evidence, stats


def label_score(ev, row_cap=None):
    """Vote strength of one label for one text.

    Legacy (row_cap falsy): templates 1.0 each + generated 0.5 each + raw real
    rows. Capped: real rows enter as sum over sites of sqrt(min(rows, cap)),
    so a single 1000-row site scores 10 (cap 100) while two sites with 50 rows
    each score 14.1 -- multi-site agreement beats one site's volume.
    """
    if row_cap:
        site_part = sum(math.sqrt(min(r, row_cap)) for r in ev['sites'].values())
    else:
        site_part = float(sum(ev['sites'].values()))
    return ev['synth'] + 0.5 * ev['gen'] + site_part


def make_overlap_fn(eo66_path):
    """Jaccard overlap between a text's tokens and a label's eo66 Display
    Name / Markers tokens (camelCase split for extension classes). Used only
    to break exact vote ties."""
    index = build_display_name_index(eo66_path)
    cache = {}

    def label_tokens(label):
        if label not in cache:
            toks = set(index.get(label, set()))
            toks |= {t.lower() for t in re.findall(r'[A-Z]?[a-z0-9]+|[A-Z]+(?![a-z])', label)}
            cache[label] = toks
        return cache[label]

    def overlap(text, label):
        t = set(text.split())
        l = label_tokens(label)
        return len(t & l) / len(t | l) if (t | l) else 0.0

    return overlap


def resolve_training_pool(evidence, overrides, row_cap=None, overlap_fn=None):
    """
    Collapse {text: {label: evidence}} to one label per text.

    Manual overrides win; otherwise the strongest label_score. In capped
    mode ties break on the number of sites backing a label, then on the
    Display-Name overlap prior (when an overlap_fn is given); a residual tie
    is dropped. Legacy mode (row_cap falsy, no overlap_fn) reproduces the
    original raw-row majority exactly.
    Returns (resolved {text: label}, conflicts list for the CSV report).
    """
    resolved = {}
    conflicts = []
    use_tiebreaks = bool(row_cap) or overlap_fn is not None

    def describe(text, label, ev, score):
        rows = int(sum(ev['sites'].values()))
        return f"{label} ({score:g}; rows {rows}, sites {len(ev['sites'])})"

    for text, label_ev in evidence.items():
        keyed = []
        for label, ev in label_ev.items():
            key = (label_score(ev, row_cap),
                   len(ev['sites']) if use_tiebreaks else 0,
                   overlap_fn(text, label) if overlap_fn is not None else 0.0)
            keyed.append((key, label, ev))
        keyed.sort(key=lambda x: x[1])                 # deterministic order among ties
        keyed.sort(key=lambda x: x[0], reverse=True)
        targets = ' | '.join(describe(text, label, ev, key[0]) for key, label, ev in keyed)

        if text in overrides:
            target = overrides[text]
            if len(label_ev) > 1:
                conflicts.append({
                    'text': text,
                    'targets': targets,
                    'resolution': f"{target if target is not None else 'DROPPED'} (override)",
                })
            if target is not None:
                resolved[text] = target
            continue

        if len(label_ev) == 1:
            resolved[text] = next(iter(label_ev))
            continue

        is_tie = keyed[0][0] == keyed[1][0]
        resolution = 'DROPPED (tie)' if is_tie else keyed[0][1]
        conflicts.append({'text': text, 'targets': targets, 'resolution': resolution})
        if not is_tie:
            resolved[text] = keyed[0][1]

    # Overrides for texts absent from the source data are honored but flagged
    known = set(evidence)
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


def load_sources(input_csv, real_points_csv, synthetic_csv, output_dir,
                 use_generated, canon, val_sites, test_sites):
    """Read + canonicalize the three training sources; returns a dict with
    synth, real, train_sites, generated (None when disabled/absent)."""
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

    # ----- Generated synthetic texts (eo66 Display Name variants) -----
    # Weight 0.5/row: any real occurrence (>=1) and any human-curated
    # template (1.0) strictly outranks a generated row in conflicts.
    generated = None
    if use_generated and os.path.exists(synthetic_csv):
        generated = pd.read_csv(synthetic_csv)
        generated['text'] = generated['text'].astype(str).str.strip()
        generated['target'] = generated['target'].astype(str).str.strip().map(canon)
        generated = generated[(generated['text'].str.len() > 0)
                              & (generated['target'].str.len() > 0)]
        print(f"Generated synthetic: {len(generated)} rows, "
              f"{generated['target'].nunique()} classes (weight 0.5/row)")
    elif use_generated:
        print(f"No {synthetic_csv} found -- continuing without generated texts")

    return {'synth': synth, 'real': real, 'train_sites': train_sites, 'generated': generated}


def convert_to_jsonl(
    input_csv='data/cleaned_data.csv',
    real_points_csv='data/real_points.csv',
    synthetic_csv='data/synthetic_points.csv',
    output_dir='data',
    val_sites=None,
    test_sites=None,
    use_generated=None,
    augment_multiplier=None,
):
    val_sites = val_sites or VAL_SITES
    test_sites = test_sites or TEST_SITES
    os.makedirs(output_dir, exist_ok=True)

    preprocessing = load_preprocessing_config()
    if use_generated is None:
        use_generated = bool(preprocessing['use_generated_synthetic'])
    if augment_multiplier is not None:  # CLI override wins
        preprocessing['augment_multiplier'] = augment_multiplier

    canon = build_label_canonicalizer(
        os.path.join(output_dir, 'eo66.xlsx'),
        os.path.join(output_dir, 'target_audit.csv'),
    )

    src = load_sources(input_csv, real_points_csv, synthetic_csv, output_dir,
                       use_generated, canon, val_sites, test_sites)
    synth, real, train_sites, generated = src['synth'], src['real'], src['train_sites'], src['generated']

    # ----- Training pool: per-site label evidence per text -----
    real_train = real[real['site'].isin(train_sites)].copy()
    evidence, ev_stats = build_evidence(synth, real_train, generated)
    if ev_stats['described_rows']:
        print(f"Train-site descriptions: +{ev_stats['described_rows']} aggregated rows of evidence")
    organic_labels = set(synth['target']) | set(real_train['label'])
    # Accept sets for LENIENT scoring: every label a TRAIN site used for the
    # identical text (site-convention credit) plus approved equivalences.
    # Held-out sites never contribute their own alternatives.
    train_site_labels = defaultdict(set)
    for row in real_train.itertuples():
        train_site_labels[row.text].add(row.label)
    equivalences = load_equivalences(os.path.join(output_dir, 'label_equivalences.csv'), canon)
    if equivalences:
        print(f"Label equivalences: {sum(len(v) for v in equivalences.values())} "
              f"approved directed credits")

    overrides = load_overrides(os.path.join(output_dir, 'label_overrides.csv'), canon)
    if overrides:
        n_dropped = sum(1 for t in overrides.values() if t is None)
        print(f"Loaded {len(overrides)} manual overrides "
              f"({len(overrides) - n_dropped} relabeled, {n_dropped} dropped)")

    row_cap = int(preprocessing.get('row_cap') or 0)
    overlap_fn = make_overlap_fn(os.path.join(output_dir, 'eo66.xlsx')) if row_cap else None
    resolved, conflicts = resolve_training_pool(evidence, overrides, row_cap=row_cap,
                                                overlap_fn=overlap_fn)
    print(f"Training pool: {len(evidence)} unique texts -> {len(resolved)} after conflict "
          f"resolution (row_cap={row_cap or 'legacy raw rows'})")

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

    # ----- Train-time augmentation (train-only by construction) -----
    heldout = real[real['site'].isin(val_sites + test_sites)]
    heldout_texts = set(heldout['text'])
    # Held-out descriptions also count as held-out strings: the eval-time
    # ensemble predicts them, so augmentation must not manufacture them
    if 'description' in heldout.columns:
        heldout_texts |= {normalize_text(str(d)) for d in heldout['description'].fillna('')
                          if str(d).strip()}
    aug_variants, aug_stats = augment_pool(
        resolved, heldout_texts, set(overrides), preprocessing)
    if aug_stats['augmented_texts']:
        drops = {k.replace('aug_dropped_', ''): v
                 for k, v in aug_stats.items() if k.startswith('aug_dropped_')}
        print(f"Augmentation: +{aug_stats['augmented_texts']} variants "
              f"(multiplier {preprocessing['augment_multiplier']}, "
              f"seed {preprocessing['augment_seed']}; dropped {drops})")

    # ----- Label space comes from the resolved pool (augmentation only
    # reuses existing labels) -----
    train_items = sorted(resolved.items())
    unique_labels = sorted({label for _, label in train_items})
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    # Classes whose entire training support is generated text: held-out
    # records they newly cover are attributable in the summary
    generated_only = {l for l in unique_labels if l not in organic_labels}
    print(f"Classes: {len(unique_labels)} "
          f"({len(generated_only)} supported only by generated texts)")

    train_records = [
        {'text': text, 'label': label, 'label_id': label2id[label]}
        for text, label in train_items
    ]
    train_records += [
        {'text': text, 'label': label, 'label_id': label2id[label], 'augmented': True}
        for text, label in aug_variants
    ]
    train_records.sort(key=lambda r: r['text'])
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
                    'accept': sorted({label} | train_site_labels.get(text, set())
                                     | equivalences.get(label, set())),
                }
                if label in label2id:
                    record['label_id'] = label2id[label]
                    covered.append(record)
                else:
                    uncovered.append(record)
        total = len(covered) + len(uncovered)
        seen = sum(1 for r in covered + uncovered if r['seen_in_train'])
        gen_only = sum(1 for r in covered if r['label'] in generated_only)
        multi = sum(1 for r in covered if len(r['accept']) > 1)
        print(f"{name}: {total} unique texts from {split_sites} "
              f"({len(covered)} covered = {len(covered) / total:.1%}, "
              f"{gen_only} covered only via generated classes, "
              f"{seen} seen in train, {ambiguous} ambiguous-gold dropped, "
              f"{multi} with lenient accept sets)")
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

    # Data-poverty signal stays pre-augmentation; the distribution the model
    # actually trains on includes augmented variants
    organic_counts = Counter(label for _, label in train_items)
    class_counts = Counter(r['label'] for r in train_records)
    text_lengths = pd.Series([len(r['text']) for r in train_records])

    def split_summary(records, uncovered, ambiguous, split_sites):
        total = len(records) + len(uncovered)
        return {
            'sites': split_sites,
            'unique_texts': total,
            'covered': len(records),
            'coverage_pct': round(100 * len(records) / total, 1) if total else None,
            'covered_generated_only': sum(1 for r in records if r['label'] in generated_only),
            'rows_total': sum(r['rows'] for r in records + uncovered),
            'seen_in_train': sum(1 for r in records + uncovered if r['seen_in_train']),
            'ambiguous_gold_dropped': ambiguous,
            'multi_accept': sum(1 for r in records if len(r.get('accept', [])) > 1),
        }

    summary_path = os.path.join(output_dir, 'dataset_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            'split_strategy': 'site-grouped (whole sites held out for val/test)',
            'num_classes': len(unique_labels),
            'train': {
                'unique_texts': len(train_records),
                'unique_texts_organic': len(train_items),
                'synthetic_rows': len(synth),
                'generated_rows': len(generated) if generated is not None else 0,
                'generated_only_classes': len(generated_only),
                'real_sites': train_sites,
                'real_rows_weight': int(real_train['rows'].sum()),
                'conflicting_texts': len(conflicts),
                'classes_with_lt3_texts': sum(1 for c in organic_counts.values() if c < 3),
                'augment_multiplier': preprocessing['augment_multiplier'],
                'augment_seed': preprocessing['augment_seed'],
                **aug_stats,
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
    parser.add_argument('--synthetic-points', default='data/synthetic_points.csv',
                        help='Generated texts from scripts/generate_synthetic.py')
    parser.add_argument('--no-generated', action='store_true',
                        help='Ignore the generated synthetic texts')
    parser.add_argument('--augment-multiplier', type=float, default=None,
                        help='Override preprocessing.augment_multiplier from config')
    parser.add_argument('--no-augment', action='store_true',
                        help='Disable train-time augmentation')
    parser.add_argument('--output-dir', default='data')
    parser.add_argument('--val-sites', default=','.join(VAL_SITES),
                        help='Comma-separated site names held out for validation')
    parser.add_argument('--test-sites', default=','.join(TEST_SITES),
                        help='Comma-separated site names held out for test (frozen)')
    args = parser.parse_args()
    convert_to_jsonl(
        input_csv=args.input_csv,
        real_points_csv=args.real_points,
        synthetic_csv=args.synthetic_points,
        output_dir=args.output_dir,
        val_sites=[s for s in args.val_sites.split(',') if s],
        test_sites=[s for s in args.test_sites.split(',') if s],
        use_generated=False if args.no_generated else None,
        augment_multiplier=0.0 if args.no_augment else args.augment_multiplier,
    )
