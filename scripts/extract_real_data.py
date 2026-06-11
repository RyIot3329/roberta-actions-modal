#!/usr/bin/env python3
"""
Extract labeled (point name, EO66 label) pairs from real site exports
=====================================================================

Reads every .xlsx in data/real_data/, auto-detects the export format,
extracts the raw BAS point name and its EO66 ground-truth label, and
canonicalizes labels via the eo66 'Regular Expression' column (numbered
variants like heatingStage01 collapse to their base definition).

Supported formats (auto-detected by column names):
  - Integ01 style:   'pointPath from BAS' (slot path; name = last segment)
                     + 'EO66 Point'
  - Integ02+ style:  'proxyExt.pointId/BASpointName' (slot path; name =
                     last segment) + 'pointTag/EO66'
  - Motorola style:  'Bacnet Name' (already a bare point name) + 'eo66Def'

The 'Slot Path' / 'Niagara New Slot Path' columns are POST-mapping (they
end in the EO66 name) and must never be used as input text.

Output: data/real_points.csv with columns site,name,label_raw,label,rows --
aggregated to one row per unique (site, name, label_raw) with `rows` holding
the occurrence count, so the file stays small enough to commit and the
training pipeline can use real-world frequencies as evidence weights.
(name is the raw point name; normalization happens in clean_data.py.)

Usage:
    python scripts/extract_real_data.py
    python scripts/extract_real_data.py --input-dir data/real_data --output data/real_points.csv
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

# (text column, label column, is_slot_path) per format, tried in order
FORMATS = [
    ('pointPath from BAS', 'EO66 Point', True),
    ('proxyExt.pointId/BASpointName', 'pointTag/EO66', True),
    ('Bacnet Name', 'eo66Def', False),
]


def build_canonicalizer(eo66_path: str):
    """Map a label to its eo66 base definition via the shipped regexes.

    The eo66 regexes match canonical tag strings (heatingStage(\\d{0,4})),
    so they resolve numbered variants in LABELS -- they cannot parse raw
    point names.
    """
    eo66 = pd.read_excel(eo66_path)
    defs = set(eo66['Definition'].dropna().astype(str).str.strip())
    regexes = []
    for d, r in zip(eo66['Definition'], eo66['Regular Expression']):
        if pd.isna(r):
            continue
        try:
            regexes.append((str(d).strip(), re.compile('^' + str(r).strip() + '$')))
        except re.error:
            pass

    cache = {}

    def canon(label: str) -> str:
        if label in cache:
            return cache[label]
        out = label
        if label not in defs:
            for definition, rx in regexes:
                if rx.match(label):
                    out = definition
                    break
        cache[label] = out
        return out

    return canon, defs


def extract_file(path: Path) -> pd.DataFrame:
    """Extract (name, label_raw) pairs from one export, or None if no format matches."""
    df = pd.read_excel(path)
    for text_col, label_col, is_path in FORMATS:
        if text_col in df.columns and label_col in df.columns:
            out = pd.DataFrame({'name': df[text_col], 'label_raw': df[label_col]})
            out = out.dropna(subset=['name', 'label_raw'])
            out['name'] = out['name'].astype(str).str.strip()
            if is_path:
                out['name'] = out['name'].str.rstrip('/').str.split('/').str[-1]
            out['label_raw'] = out['label_raw'].astype(str).str.strip()
            out = out[(out['name'].str.len() > 0)
                      & (out['label_raw'].str.len() > 0)
                      & (~out['label_raw'].str.lower().isin(['nan', 'none']))]
            return out
    return None


def main():
    parser = argparse.ArgumentParser(description='Extract labeled points from real site exports')
    parser.add_argument('--input-dir', default='data/real_data')
    parser.add_argument('--eo66', default='data/eo66.xlsx')
    parser.add_argument('--output', default='data/real_points.csv')
    args = parser.parse_args()

    files = sorted(Path(args.input_dir).glob('*.xlsx'))
    if not files:
        print(f'Error: no .xlsx files in {args.input_dir}')
        sys.exit(1)

    canon, eo66_defs = build_canonicalizer(args.eo66)

    frames = []
    for path in files:
        site = path.stem.replace(' ', '_')
        df = extract_file(path)
        if df is None:
            print(f'WARNING: {path.name}: no known column format, skipped')
            continue
        df.insert(0, 'site', site)
        frames.append(df)
        print(f'{path.name}: {len(df)} labeled rows, {df["name"].nunique()} unique names')

    all_df = pd.concat(frames, ignore_index=True)
    all_df['label'] = all_df['label_raw'].map(canon)

    n_canon = (all_df['label'] != all_df['label_raw']).sum()
    in_eo66 = all_df['label'].isin(eo66_defs)
    print(f'\nTotal: {len(all_df)} rows across {all_df["site"].nunique()} sites')
    print(f'Labels canonicalized by eo66 regex: {n_canon} rows '
          f'({all_df.loc[all_df.label != all_df.label_raw, "label_raw"].nunique()} distinct labels)')
    print(f'Rows with canonical label in eo66: {in_eo66.mean():.1%}')
    outside = all_df.loc[~in_eo66, 'label'].value_counts()
    if len(outside) > 0:
        print(f'Labels outside eo66 ({len(outside)} distinct, {(~in_eo66).sum()} rows), top 10:')
        for label, count in outside.head(10).items():
            print(f'  {count:6d}  {label}')

    # Aggregate so the file is commit-friendly and keeps frequency as weight
    agg = (all_df.groupby(['site', 'name', 'label_raw', 'label'], sort=True)
           .size().reset_index(name='rows'))
    agg.to_csv(args.output, index=False)
    print(f'\nSaved: {args.output} ({len(agg)} aggregated rows from {len(all_df)})')


if __name__ == '__main__':
    main()
