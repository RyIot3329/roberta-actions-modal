"""
Markdown summary of a training run's <stem>_metrics.json (written by
scripts/finetune.py) for the results PR body / workflow summary.

Usage: python scripts/summarize_run.py [output/<stem>_metrics.json]
       (defaults to the newest *_metrics.json in output/)
"""
import glob
import json
import os
import sys


def pct(x):
    return "" if x is None else f"{100 * x:.2f}%"


def summarize(path):
    with open(path) as f:
        m = json.load(f)
    cand = m.get("candidate", {})
    gate = m.get("gate", {})
    lines = []
    passed = gate.get("passed")
    lines.append(f"**Gate:** {'PASSED' if passed else 'FAILED'} -- {gate.get('reason', '')}")
    boot = gate.get("bootstrap")
    if boot:
        lines.append(f"Paired bootstrap on {boot['n_paired']} texts: delta {boot['delta']:+.4f} "
                     f"[{boot['ci_lo']:+.4f}, {boot['ci_hi']:+.4f}], P(worse) {boot['p_worse']:.2f}")
    note = m.get("baseline_note")
    if note:
        lines.append(f"Baseline: {note}")
    fp = cand.get("fingerprints") or {}
    if fp:
        lines.append(f"Scoreboard: {fp.get('n_test')} test records, {fp.get('n_classes')} classes "
                     f"(test sha {str(fp.get('test_sha256'))[:12]})")
    lines.append("")
    lines.append("| seed | val strict | test strict | test lenient | test log1p-rows | tau |")
    lines.append("|---|---|---|---|---|---|")
    seeds = cand.get("seed_summary") or {}
    sel = cand.get("selected_seed")
    for s, v in seeds.items():
        mark = " (selected)" if str(sel) == str(s) else ""
        per = (m.get("per_seed") or {}).get(str(s), {})
        log1p = ((per.get("test") or {}).get("strict") or {}).get("log1p_rows_accuracy")
        lines.append(f"| {s}{mark} | {pct(v.get('val_strict'))} | {pct(v.get('test_strict'))} | "
                     f"{pct(v.get('test_lenient'))} | {pct(log1p)} | {v.get('tau', 0):.3f} |")
    ens = cand.get("ensemble")
    if ens:
        lines.append(f"| ensemble | {pct(ens.get('val_strict'))} | {pct(ens.get('test_strict'))} | "
                     f"{pct(ens.get('test_lenient'))} | {pct(ens.get('test_log1p_rows'))} | |")
    soup = cand.get("soup")
    if soup:
        mark = " (candidate)" if soup.get("is_candidate") else ""
        lines.append(f"| soup{soup.get('selected_seeds')}{mark} | {pct(soup.get('val_strict'))} | "
                     f"{pct(soup.get('test_strict'))} | {pct(soup.get('test_lenient'))} | "
                     f"{pct(soup.get('test_log1p_rows'))} | |")
    agree = cand.get("seed_agreement")
    if agree:
        lines.append(f"\nSeed agreement on test: pairwise {agree['pairwise_mean']:.3f}, "
                     f"unanimous {agree['unanimous']:.3f}")
    slices = ((cand.get("metrics") or {}).get("slices") or {})
    if slices:
        lines.append("\n| test slice (selected seed) | n | strict | lenient |")
        lines.append("|---|---|---|---|")
        for name, sl in slices.items():
            lines.append(f"| {name} | {sl['n']} | {pct(sl['strict'])} | {pct(sl['lenient'])} |")
    cov = ((cand.get("metrics") or {}).get("coverage") or {})
    if cov:
        lines.append(f"\nCoverage at tau={cov.get('tau', 0):.3f} (validation-fitted): "
                     f"{pct(cov.get('coverage_texts'))} of texts at precision {pct(cov.get('precision_at_tau'))}")
    ctx = (cand.get("context") or {}).get("test") if cand.get("context") else None
    if ctx:
        lines.append("\n| context view (test pairs, selected seed) | n | strict | lenient | log1p-rows |")
        lines.append("|---|---|---|---|---|")
        for view in ("name_on_pairs", "context", "ensemble"):
            v = ctx.get(view) or {}
            lines.append(f"| {view} | {v.get('n')} | {pct(v.get('strict'))} | {pct(v.get('lenient'))} | "
                         f"{pct(v.get('log1p_rows'))} |")
    floor = m.get("linear_floor_strict")
    if floor is not None:
        strict = ((cand.get("metrics") or {}).get("strict") or {}).get("accuracy")
        if strict is not None:
            lines.append(f"\nLinear floor strict {pct(floor)} -> transformer margin {100 * (strict - floor):+.2f}pp")
    axes = gate.get("axes") or []
    if axes:
        lines.append("\n| gate axis | baseline | candidate | delta | ok |")
        lines.append("|---|---|---|---|---|")
        for a in axes:
            def f(x):
                return "" if x is None else (f"{x:.4f}" if isinstance(x, float) else str(x))
            lines.append(f"| {a['axis']} | {f(a.get('baseline'))} | {f(a.get('candidate'))} | "
                         f"{f(a.get('delta'))} | {'yes' if a.get('ok') else 'NO'} |")
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        files = sorted(glob.glob("output/*_metrics.json"), key=os.path.getmtime)
        files = [f for f in files if not f.endswith("baseline_linear_metrics.json")]
        if not files:
            print("no run metrics found")
            sys.exit(0)
        path = files[-1]
    print(summarize(path))
