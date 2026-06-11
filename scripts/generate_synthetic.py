#!/usr/bin/env python3
"""
Generate synthetic training texts for under-supported classes
==============================================================

Builds abbreviation-variant texts from eo66 Display Names for classes that
the real sites use but the training pool barely covers: canonical labels
observed anywhere in data/real_points.csv whose training support (synthetic
+ real TRAINING-site unique texts) is below --min-texts.

Leakage stance: held-out sites contribute only their LABEL LIST (which
taxonomy classes the product must express -- a scope decision); generated
TEXTS derive solely from eo66 columns and data/abbreviations.csv. Held-out
site texts are never read.

Abbreviation dictionary: data/abbreviations.csv (word, abbrevs pipe-
separated, source=curated|mined) is a committed, human-reviewed artifact.
Generation reads ONLY this file. `--mine` refreshes the mined rows by
positionally aligning data/cleaned_data.csv texts against eo66 Display
Names (kept where token count matches), filtered to plausible pairs:
(abbreviation is a subsequence of the word AND shares its first letter)
OR relative frequency >= 8% (admits synonyms like discharge -> sa/supply).
Curated rows are never touched by mining.

Determinism: per-class RNG seeded by sha256(seed|label) -- adding a class
never reshuffles another class's variants, so the committed CSV diff stays
minimal. Words missing from the dictionary fall back to identity +
vowel-stripped form and are listed loudly for curation.

Output: data/synthetic_points.csv (text,target,source), sorted. Consumed
by convert_to_jsonl.py at weight 0.5/row so any real occurrence or human
template wins conflicts. Run offline and commit -- CI never executes this.

Usage:
    python scripts/generate_synthetic.py --mine   # refresh mined dictionary rows
    python scripts/generate_synthetic.py          # write data/synthetic_points.csv
"""

import argparse
import hashlib
import itertools
import os
import random
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clean_data import normalize_text
from extract_real_data import build_canonicalizer
from convert_to_jsonl import VAL_SITES, TEST_SITES, build_label_canonicalizer

SOURCE_TAG = 'eo66-dn-v1'
VARIANTS_CAP = 40
VOWELS = re.compile(r'[aeiou]')


def class_rng(seed: int, label: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}|{label}".encode()).hexdigest()
    return random.Random(int(digest, 16) % (2 ** 63))


def load_abbreviations(path):
    """{word: [abbrevs in preference order]} from the committed CSV."""
    if not os.path.exists(path):
        return {}, pd.DataFrame(columns=['word', 'abbrevs', 'source'])
    df = pd.read_csv(path)
    table = {}
    for row in df.itertuples():
        abbrevs = [a.strip() for a in str(row.abbrevs).split('|') if a.strip()]
        if row.word not in table:  # curated rows are listed first and win
            table[str(row.word)] = abbrevs
    return table, df


def mine_abbreviations(cleaned_csv, eo66_path, audit_path, out_path):
    """Positional alignment of normalized texts against Display Name words."""
    canon = build_label_canonicalizer(eo66_path, audit_path)
    eo66 = pd.read_excel(eo66_path)
    display = {}
    for d, n in zip(eo66['Definition'], eo66['Display Name']):
        d = str(d).strip()
        if d and canon(d) == d and isinstance(n, str):
            display[d] = normalize_text(n).split()

    df = pd.read_csv(cleaned_csv)
    df['target'] = df['target'].astype(str).map(canon)

    pair_counts = {}
    word_totals = {}
    for row in df.itertuples():
        words = display.get(row.target)
        if not words:
            continue
        tokens = str(row.text).split()
        if len(tokens) != len(words):
            continue
        for word, token in zip(words, tokens):
            pair_counts[(word, token)] = pair_counts.get((word, token), 0) + 1
            word_totals[word] = word_totals.get(word, 0) + 1

    mined = {}
    for (word, token), count in pair_counts.items():
        plausible = token[0:1] == word[0:1] and _is_subsequence(token, word)
        frequent = count / word_totals[word] >= 0.08
        if plausible or frequent:
            mined.setdefault(word, []).append((count, token))

    _, existing = load_abbreviations(out_path)
    curated = existing[existing['source'] == 'curated'] if len(existing) else existing
    curated_words = set(curated['word']) if len(curated) else set()

    rows = []
    for word in sorted(mined):
        if word in curated_words:
            continue  # human decisions outrank mining
        abbrevs = [t for _, t in sorted(mined[word], key=lambda x: (-x[0], x[1]))]
        rows.append({'word': word, 'abbrevs': '|'.join(abbrevs), 'source': 'mined'})

    out = pd.concat([curated, pd.DataFrame(rows)], ignore_index=True)
    out.to_csv(out_path, index=False)
    print(f"Mined {len(rows)} words ({len(curated_words)} curated rows preserved) -> {out_path}")


def _is_subsequence(small, big):
    it = iter(big)
    return all(c in it for c in small)


def training_support(cleaned_csv, real_points_csv, canon):
    """Unique-text count per canonical label across synthetic + train-site real data."""
    support = {}
    synth = pd.read_csv(cleaned_csv)
    synth['target'] = synth['target'].astype(str).map(canon)
    for label, group in synth.groupby('target'):
        support[label] = len(set(group['text']))
    real = pd.read_csv(real_points_csv)
    real = real[~real['site'].isin(VAL_SITES + TEST_SITES)]
    real['label'] = real['label'].astype(str).map(canon)
    real['text'] = real['name'].astype(str).map(normalize_text)
    for label, group in real.groupby('label'):
        support[label] = support.get(label, 0) + len(set(group['text']))
    return support


def generate_for_class(label, words, abbrev_table, seed, fallback_words):
    """Up to VARIANTS_CAP normalized abbreviation variants for one class."""
    options = []
    for word in words:
        abbrevs = abbrev_table.get(word)
        if not abbrevs:
            fallback_words.add(word)
            abbrevs = [word]
            stripped = word[0] + VOWELS.sub('', word[1:])
            if len(word) >= 5 and stripped != word and len(stripped) >= 2:
                abbrevs.append(stripped)
        options.append(abbrevs)

    rng = class_rng(seed, label)
    variants = set()

    def add(parts):
        text = normalize_text(' '.join(parts))
        if text:
            variants.add(text)

    add(words)                                  # full-words form
    add([opts[0] for opts in options])          # modal-abbreviation form

    total = 1
    for opts in options:
        total *= len(opts)
    if total <= 400:
        pool = [' '.join(parts) for parts in itertools.product(*options)]
        rng.shuffle(pool)
        for candidate in pool:
            if len(variants) >= VARIANTS_CAP:
                break
            add(candidate.split(' '))
    else:
        attempts = 0
        while len(variants) < VARIANTS_CAP and attempts < VARIANTS_CAP * 10:
            add([rng.choice(opts) for opts in options])
            attempts += 1
    return sorted(variants)[:VARIANTS_CAP]


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic texts for thin classes')
    parser.add_argument('--mine', action='store_true',
                        help='Refresh mined rows in the abbreviation dictionary and exit')
    parser.add_argument('--cleaned-csv', default='data/cleaned_data.csv')
    parser.add_argument('--real-points', default='data/real_points.csv')
    parser.add_argument('--eo66', default='data/eo66.xlsx')
    parser.add_argument('--audit', default='data/target_audit.csv')
    parser.add_argument('--abbreviations', default='data/abbreviations.csv')
    parser.add_argument('--output', default='data/synthetic_points.csv')
    parser.add_argument('--min-texts', type=int, default=3,
                        help='Generate for classes with fewer unique training texts than this')
    parser.add_argument('--labels', default='',
                        help='Comma-separated extra labels to generate regardless of support')
    parser.add_argument('--seed', type=int, default=20260611)
    args = parser.parse_args()

    if args.mine:
        mine_abbreviations(args.cleaned_csv, args.eo66, args.audit, args.abbreviations)
        return

    canon = build_label_canonicalizer(args.eo66, args.audit)
    abbrev_table, _ = load_abbreviations(args.abbreviations)
    if not abbrev_table:
        print(f"ERROR: {args.abbreviations} missing or empty -- run --mine and curate it first")
        sys.exit(1)

    eo66 = pd.read_excel(args.eo66)
    display = {}
    for d, n in zip(eo66['Definition'], eo66['Display Name']):
        d = str(d).strip()
        if d and canon(d) == d and isinstance(n, str):  # skip numbered variants
            display[d] = normalize_text(n).split()

    real_labels = set(pd.read_csv(args.real_points)['label'].astype(str).map(canon))
    support = training_support(args.cleaned_csv, args.real_points, canon)
    extra = {l.strip() for l in args.labels.split(',') if l.strip()}

    targets = sorted(l for l in (real_labels | extra)
                     if (support.get(l, 0) < args.min_texts or l in extra))
    generatable = [l for l in targets if l in display]
    ungeneratable = [l for l in targets if l not in display]

    print(f"Classes below {args.min_texts} training texts (observed in real data): {len(targets)}")
    print(f"  generatable from eo66 Display Names: {len(generatable)}")
    if ungeneratable:
        print(f"  NOT in eo66 ({len(ungeneratable)}) -- review as data/target_audit.csv candidates:")
        for label in ungeneratable:
            print(f"    {label}")

    fallback_words = set()
    rows = []
    for label in generatable:
        for text in generate_for_class(label, display[label], abbrev_table,
                                       args.seed, fallback_words):
            rows.append({'text': text, 'target': label, 'source': SOURCE_TAG})

    out = pd.DataFrame(rows).sort_values(['target', 'text']).reset_index(drop=True)
    out.to_csv(args.output, index=False)
    print(f"\nGenerated {len(out)} texts for {out['target'].nunique()} classes -> {args.output}")
    if fallback_words:
        print(f"\nWords missing from {args.abbreviations} (identity + vowel-strip fallback used) "
              f"-- curate these {len(fallback_words)}:")
        print('  ' + ', '.join(sorted(fallback_words)))


if __name__ == '__main__':
    main()
