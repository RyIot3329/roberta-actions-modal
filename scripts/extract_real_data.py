#!/usr/bin/env python3
"""
Extract labeled (point name, EO66 label) pairs from real site exports
=====================================================================

Reads every .xlsx in data/real_data/, auto-detects the export format,
extracts the raw BAS point name and its EO66 ground-truth label, and
canonicalizes labels via the eo66 'Regular Expression' column (numbered
variants like heatingStage01 collapse to their base definition), plus two
fallbacks for labels the regexes miss: unique case-insensitive definition
match (SecEnteringTemp -> secEnteringTemp) and species-preserving
equipment-index stripping (twrIsoValve01 -> twrIsoValve, zoneTempAvg03 ->
zoneTempAvg) -- see build_canonicalizer.

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

# (text column, label column, is_slot_path, description column or None)
# per format, tried in order. Descriptions are free-text point metadata
# (e.g. BACnet's description property) that carries the semantics when the
# name is a bare object reference like "AV 00".
FORMATS = [
    ('pointPath from BAS', 'EO66 Point', True, None),
    ('proxyExt.pointId/BASpointName', 'pointTag/EO66', True, None),
    ('Bacnet Name', 'eo66Def', False, 'BACnet Description'),
]


# Species / refrigerant tokens whose digits are semantic (zoneN2Alarm,
# zoneCO2Avg): masked before equipment-index digits are stripped from a
# label. Uppercase-only on purpose -- a lowercase 'n2' inside a label is
# equipment numbering (fan2), not a species. Longest tokens first so the
# alternation never truncates a match (PM25 before PM2, R134A before R11).
_LABEL_SPECIES = re.compile(
    r'(R134A|R410A|PM25|PM10|R114|R123|R125|R134|R410'
    r'|CO2|SO2|NO2|H2S|CH4|CL2|PM1|PM2|PM4|R11|R22|N2|O2|O3|H2)'
)
# A digit run plus an optional single instance letter glued to it (pump
# 01A / 01B, floor 10N / 20S). The lookahead keeps real word starts safe:
# in zoneTemp01Avg the A begins "Avg" (next char lowercase) and survives.
_LABEL_INDEX_RUN = re.compile(r'\d+(?:[A-Z](?=[A-Z]|$))?')
# Separators orphaned by index removal (secPump11-12Status -> secPump-Status,
# Blr1_Sts -> Blr_Sts, CLR_02 -> CLR_). eo66 definitions never contain -/_,
# so anything left holding one is a site mislabel being consolidated.
_LABEL_ORPHAN_SEP = re.compile(r'[-_]+(?=[A-Z]|$)')


def _strip_label_indices(label: str) -> str:
    """Drop equipment-index digits (and their unit letter) from a camelCase
    label: zoneTempAvg03 -> zoneTempAvg, secPump01APowerReal ->
    secPumpPowerReal, secPressureDelta10N -> secPressureDelta. Species
    digits survive: zoneN2Alarm and zoneCO2Avg are untouched."""
    parts = _LABEL_SPECIES.split(label)  # odd indices are species tokens
    stripped = ''.join(part if i % 2 else _LABEL_INDEX_RUN.sub('', part)
                       for i, part in enumerate(parts))
    return _LABEL_ORPHAN_SEP.sub('', stripped)


def build_display_name_index(eo66_path: str):
    """{definition: token set} from eo66 Display Names + Markers (normalized),
    used as a weak self-evidence prior when conflict votes tie: the label
    whose own words overlap the point name wins the tie."""
    from clean_data import normalize_text
    eo66 = pd.read_excel(eo66_path)
    index = {}
    for d, dn, mk in zip(eo66['Definition'], eo66['Display Name'], eo66['Markers']):
        if not isinstance(d, str):
            continue
        tokens = set()
        if isinstance(dn, str):
            tokens |= set(normalize_text(dn).split())
        if isinstance(mk, str):
            tokens |= {t.strip().lower() for t in mk.split(',') if t.strip()}
        index[d.strip()] = tokens
    return index


def build_canonicalizer(eo66_path: str):
    """Map a label to its eo66 base definition via the shipped regexes.

    The eo66 regexes match canonical tag strings (heatingStage(\\d{0,4})),
    so they resolve numbered variants in LABELS -- they cannot parse raw
    point names.

    Labels the regexes cannot resolve get two conservative fallbacks:
      1. unique case-insensitive match against the eo66 definitions
         (SecEnteringTemp -> secEnteringTemp, hotDeckdischargeFlow ->
         hotDeckDischargeFlow);
      2. equipment-index digits stripped (species-preserving), then
         re-resolved -- numbered site extensions whose BASE form is not an
         eo66 definition (twrIsoValve01, twrFan02Frequency) have no regex
         to catch them, so they escaped canonicalization entirely. The
         stripped form is kept even when it is not an eo66 definition:
         consolidating twrFan01Frequency and twrFan02Frequency into one
         twrFanFrequency extension class beats training two junk classes.
    Every fallback decision is recorded in canon.fallback_remaps for the
    caller to report.
    """
    eo66 = pd.read_excel(eo66_path)
    defs = set(eo66['Definition'].dropna().astype(str).str.strip())
    lower_defs = {}
    for d in sorted(defs):  # deterministic winner if a case-dup ever appears
        lower_defs.setdefault(d.lower(), d)
    regexes = []
    for d, r in zip(eo66['Definition'], eo66['Regular Expression']):
        if pd.isna(r):
            continue
        try:
            regexes.append((str(d).strip(), re.compile('^' + str(r).strip() + '$')))
        except re.error:
            pass

    def resolve(label: str):
        """Definition set, then eo66 regexes; None when neither matches."""
        if label in defs:
            return label
        for definition, rx in regexes:
            if rx.match(label):
                return definition
        return None

    cache = {}

    def canon(label: str) -> str:
        if label in cache:
            return cache[label]
        out = resolve(label)
        if out is None:
            out = lower_defs.get(label.lower())
            if out is None:
                stripped = _strip_label_indices(label)
                if stripped and stripped != label:
                    out = (resolve(stripped)
                           or lower_defs.get(stripped.lower())
                           or stripped)
            if out is None:
                out = label
            if out != label:
                canon.fallback_remaps[label] = out
        cache[label] = out
        return out

    canon.fallback_remaps = {}
    return canon, defs


def extract_file(path: Path) -> pd.DataFrame:
    """Extract (name, label_raw, description) from one export, or None if no format matches."""
    df = pd.read_excel(path)
    for text_col, label_col, is_path, desc_col in FORMATS:
        if text_col in df.columns and label_col in df.columns:
            out = pd.DataFrame({'name': df[text_col], 'label_raw': df[label_col]})
            if desc_col and desc_col in df.columns:
                out['description'] = df[desc_col].fillna('').astype(str).str.strip()
            else:
                out['description'] = ''
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
    if canon.fallback_remaps:
        print(f'Labels consolidated by case/index fallback '
              f'({len(canon.fallback_remaps)} distinct):')
        for src, dst in sorted(canon.fallback_remaps.items()):
            marker = '' if dst in eo66_defs else '  [extension]'
            print(f'  {src} -> {dst}{marker}')
    print(f'Rows with canonical label in eo66: {in_eo66.mean():.1%}')
    outside = all_df.loc[~in_eo66, 'label'].value_counts()
    if len(outside) > 0:
        print(f'Labels outside eo66 ({len(outside)} distinct, {(~in_eo66).sum()} rows), top 10:')
        for label, count in outside.head(10).items():
            print(f'  {count:6d}  {label}')

    # Aggregate so the file is commit-friendly and keeps frequency as weight
    all_df['description'] = all_df['description'].fillna('')
    agg = (all_df.groupby(['site', 'name', 'description', 'label_raw', 'label'], sort=True)
           .size().reset_index(name='rows'))
    n_desc = (agg['description'].str.len() > 0).sum()
    agg.to_csv(args.output, index=False)
    print(f'\nSaved: {args.output} ({len(agg)} aggregated rows from {len(all_df)}, '
          f'{n_desc} with descriptions)')


if __name__ == '__main__':
    main()
