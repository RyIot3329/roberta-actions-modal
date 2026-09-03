"""
Audit conflict resolution: legacy raw-row majority vs the capped per-site vote.

Builds the training evidence exactly like scripts/convert_to_jsonl.py, resolves
it both ways and writes data/conflict_review.csv (one row per conflicting text,
old vs new resolution, rows at stake, per-site votes, held-out golds) sorted by
rows at stake, so the flips can be bulk-reviewed. Approved corrections go into
data/label_overrides.csv, which always wins.

Usage:
    python scripts/audit_conflicts.py                 # row_cap from config (100)
    python scripts/audit_conflicts.py --row-cap 250
    python scripts/audit_conflicts.py --dry-run       # new==legacy settings: must report 0 changes
"""
import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from convert_to_jsonl import (VAL_SITES, TEST_SITES, build_evidence,  # noqa: E402
                              build_label_canonicalizer, label_score, load_overrides,
                              load_preprocessing_config, load_sources, make_overlap_fn,
                              resolve_training_pool, site_unique_texts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--row-cap', type=int, default=None, help='Override preprocessing.row_cap')
    ap.add_argument('--no-overlap', action='store_true', help='Disable the Display-Name tie-break')
    ap.add_argument('--dry-run', action='store_true',
                    help='Resolve with legacy settings on the new code path; must yield 0 changes')
    ap.add_argument('--out', default='data/conflict_review.csv')
    ap.add_argument('--data-dir', default='data')
    args = ap.parse_args()

    pre = load_preprocessing_config()
    row_cap = 0 if args.dry_run else (args.row_cap if args.row_cap is not None
                                      else int(pre.get('row_cap') or 0))
    canon = build_label_canonicalizer(os.path.join(args.data_dir, 'eo66.xlsx'),
                                      os.path.join(args.data_dir, 'target_audit.csv'))
    src = load_sources(os.path.join(args.data_dir, 'cleaned_data.csv'),
                       os.path.join(args.data_dir, 'real_points.csv'),
                       os.path.join(args.data_dir, 'synthetic_points.csv'),
                       args.data_dir, bool(pre['use_generated_synthetic']), canon,
                       VAL_SITES, TEST_SITES)
    real, train_sites = src['real'], src['train_sites']
    real_train = real[real['site'].isin(train_sites)].copy()
    evidence, _ = build_evidence(src['synth'], real_train, src['generated'])
    overrides = load_overrides(os.path.join(args.data_dir, 'label_overrides.csv'), canon)

    legacy, _ = resolve_training_pool(evidence, overrides)
    overlap_fn = None if (args.no_overlap or args.dry_run) else make_overlap_fn(
        os.path.join(args.data_dir, 'eo66.xlsx'))
    new, new_conflicts = resolve_training_pool(evidence, overrides, row_cap=row_cap,
                                               overlap_fn=overlap_fn)

    # Held-out golds for context (what the frozen sites call the same text)
    heldout_gold = {}
    for site in VAL_SITES + TEST_SITES:
        for text, label, rows in site_unique_texts(real[real['site'] == site]):
            heldout_gold.setdefault(text, {})[site] = label

    rows_out = []
    for text, label_ev in evidence.items():
        if len(label_ev) < 2:
            continue
        old_res = legacy.get(text, 'DROPPED (tie)')
        new_res = new.get(text, 'DROPPED (tie)')
        if text in overrides:
            old_res = new_res = f"{overrides[text] or 'DROPPED'} (override)"
        rows_total = int(sum(sum(ev['sites'].values()) for ev in label_ev.values()))
        per_site = {label: dict(ev['sites']) for label, ev in label_ev.items()}
        golds = heldout_gold.get(text, {})
        rows_out.append({
            'text': text,
            'old_resolution': old_res,
            'new_resolution': new_res,
            'changed': old_res != new_res,
            'n_labels': len(label_ev),
            'rows_at_stake': rows_total,
            'old_score': round(label_score(label_ev[old_res], 0), 3) if old_res in label_ev else '',
            'new_score': round(label_score(label_ev[new_res], row_cap), 3) if new_res in label_ev else '',
            'n_sites_old': len(label_ev[old_res]['sites']) if old_res in label_ev else '',
            'n_sites_new': len(label_ev[new_res]['sites']) if new_res in label_ev else '',
            'synth_gen_old': (f"{label_ev[old_res]['synth']:g}/{label_ev[old_res]['gen']:g}"
                              if old_res in label_ev else ''),
            'synth_gen_new': (f"{label_ev[new_res]['synth']:g}/{label_ev[new_res]['gen']:g}"
                              if new_res in label_ev else ''),
            'per_site_rows': str(per_site),
            'val_gold': golds.get(VAL_SITES[0], ''),
            'test_gold': ' | '.join(f"{s}:{l}" for s, l in golds.items() if s in TEST_SITES),
            'decision': '',
        })
    df = pd.DataFrame(rows_out).sort_values(['changed', 'rows_at_stake'], ascending=[False, False])
    changed = df[df['changed']]
    recovered = changed[(changed['old_resolution'] == 'DROPPED (tie)')
                        & (changed['new_resolution'] != 'DROPPED (tie)')]
    newly_dropped = changed[(changed['new_resolution'] == 'DROPPED (tie)')
                            & (changed['old_resolution'] != 'DROPPED (tie)')]
    flips = changed[(changed['old_resolution'] != 'DROPPED (tie)')
                    & (changed['new_resolution'] != 'DROPPED (tie)')]
    agree_val = sum(1 for r in flips.itertuples() if r.val_gold and r.val_gold == r.new_resolution)
    agree_val_old = sum(1 for r in flips.itertuples() if r.val_gold and r.val_gold == r.old_resolution)
    print(f"Conflicting texts: {len(df)}  (row_cap={row_cap or 'legacy'}, "
          f"overlap tie-break={'on' if overlap_fn else 'off'})")
    print(f"  changed: {len(changed)}  = flips {len(flips)} + recovered ties {len(recovered)} "
          f"+ newly dropped {len(newly_dropped)}")
    if len(flips):
        print(f"  flips where the validation site has a gold: new agrees {agree_val}, "
              f"old agreed {agree_val_old}")
    if args.dry_run:
        if len(changed):
            print("DRY-RUN FAILED: legacy settings changed the resolution")
            print(changed.head(20).to_string())
            sys.exit(1)
        print("DRY-RUN OK: new resolver reproduces the legacy pool exactly")
        return
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Saved: {args.out}")
    if len(flips):
        print("\nTop flips by rows at stake:")
        for r in flips.head(20).itertuples():
            print(f"  {r.text!r:<30} {r.old_resolution} -> {r.new_resolution}  "
                  f"(rows {r.rows_at_stake}, sites {r.n_sites_old}->{r.n_sites_new}; "
                  f"val gold {r.val_gold or '-'})")
    if len(recovered):
        print(f"\nRecovered ties (sample of {len(recovered)}):")
        for r in recovered.head(10).itertuples():
            print(f"  {r.text!r:<30} -> {r.new_resolution}  (rows {r.rows_at_stake})")


if __name__ == '__main__':
    main()
