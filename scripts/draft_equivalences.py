"""
Draft data/label_equivalences.csv: candidate label pairs that evaluation
should treat as equivalent (lenient credit), for bulk human approval.

Sources of candidate pairs:
  (a) confusion pairs (gold, predicted) seen >= 2 times in archived
      validation/test predictions (output/*_predictions.jsonl,
      output/best_predictions.jsonl, JSON blobs in output/*.txt)
  (b) the live probe's acceptables / correct-twin / correct-upgrade rows
  (c) cross-site disagreements (data/conflict_review.csv: labels that
      different sites gave the same text)
  (d) eo66 Display-Name token-F1 >= 0.75 with a shared Equipment field

Relation drafting (all rows start as status=draft):
  same          identical eo66 marker set + units + kind  (the only rows a
                training-time merge may ever use)
  parent        one label's tokens are a strict subset of the other's
                (direction a_to_b: predicting a for gold b is acceptable)
  site_variant  everything else that looks like the same concept
                (qualifier twin Hi/Max, Lo/Min, Sp/Fb, ...; high name similarity)
  confusable    weak evidence (confusions only) -- report-only

--merge-existing keeps approved/rejected verdicts from the current file so
re-drafting is idempotent; approval is editing the status column.
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from clean_data import normalize_text  # noqa: E402
from convert_to_jsonl import build_label_canonicalizer  # noqa: E402

COLUMNS = ["label_a", "label_b", "relation", "direction", "evidence", "n_confusions", "sites",
           "marker_jaccard", "display_similarity", "units_match", "kind_match", "proposed_by",
           "status", "reviewer", "notes"]
QUALIFIER_TWINS = [{"hi", "max"}, {"lo", "min"}, {"sp", "fb"}, {"cmd", "fb"}, {"status", "fb"},
                   {"enable", "cmd"}, {"status", "cmd"}, {"cool", "max"}, {"heat", "min"},
                   {"cool", "hi"}, {"heat", "lo"}, {"filter", "unit"},
                   # chilled-water loop position and condenser/tower side are
                   # rarely decidable from the point name alone (probe finding 2)
                   {"pri", "sec"}, {"evap", "sec"}, {"evap", "pri"}, {"cond", "twr"}]


def camel_tokens(label):
    return [t.lower() for t in re.findall(r"[A-Z]?[a-z0-9]+|[A-Z]+(?![a-z])", label)]


def load_json_blob(path):
    txt = open(path, encoding="utf-8").read()
    i = txt.find("\n{\n")
    if i < 0:
        return None
    blob = txt[i + 1:]
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        j = blob.rfind("\n}")
        try:
            return json.loads(blob[: j + 2])
        except json.JSONDecodeError:
            return None


def confusion_pairs(canon, output_dir="output", max_txt=6):
    pairs = Counter()
    for path in glob.glob(os.path.join(output_dir, "*_predictions.jsonl")) + \
            [os.path.join(output_dir, "best_predictions.jsonl")]:
        if not os.path.exists(path):
            continue
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            g, p = canon(r["actual_label"]), canon(r["predicted_label"])
            if g != p:
                pairs[tuple(sorted((g, p)))] += 1
    txts = sorted(glob.glob(os.path.join(output_dir, "2026*_deberta*.txt")))[-max_txt:]
    for path in txts:
        blob = load_json_blob(path)
        if not blob:
            continue
        for key in ("validation_inference", "test_inference"):
            for r in (blob.get(key) or {}).get("predictions", []):
                g, p = canon(r["actual_label"]), canon(r["predicted_label"])
                if g != p:
                    pairs[tuple(sorted((g, p)))] += 1
    return pairs


def probe_pairs(canon, path="output/live_probe_20260713_rows.csv"):
    pairs = Counter()
    if not os.path.exists(path):
        return pairs
    df = pd.read_csv(path)
    for r in df.itertuples():
        gold = canon(str(r.gold).strip()) if isinstance(r.gold, str) else None
        if not gold:
            continue
        acc = set()
        if isinstance(getattr(r, "acceptables", None), str) and r.acceptables.strip():
            acc |= {canon(a.strip()) for a in r.acceptables.split("|") if a.strip()}
        verdict = str(r.verdict)
        pred = getattr(r, "predicted", None)
        if verdict.startswith("correct-") and isinstance(pred, str):
            acc.add(canon(pred.strip()))
        if verdict == "wrong-adjacent" and isinstance(pred, str):
            acc.add(canon(pred.strip()))
        for a in acc:
            if a != gold:
                pairs[tuple(sorted((gold, a)))] += 1
    return pairs


def cross_site_pairs(path="data/conflict_review.csv"):
    pairs = Counter()
    if not os.path.exists(path):
        return pairs
    df = pd.read_csv(path)
    for r in df.itertuples():
        try:
            per_site = eval(r.per_site_rows, {"__builtins__": {}})  # dict literal written by the audit
        except Exception:  # noqa: BLE001
            continue
        labels = [l for l, sites in per_site.items() if sites]
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                pairs[tuple(sorted((labels[i], labels[j])))] += 1
    return pairs


def eo66_index(path="data/eo66.xlsx"):
    eo = pd.read_excel(path)
    info = {}
    for r in eo.itertuples():
        d = getattr(r, "Definition")
        if not isinstance(d, str):
            continue
        markers = getattr(r, "Markers")
        info[d.strip()] = {
            "markers": frozenset(t.strip() for t in markers.split(",") if t.strip())
            if isinstance(markers, str) else frozenset(),
            "display": set(normalize_text(str(getattr(r, "_3"))).split())
            if isinstance(getattr(r, "_3", None), str) else set(),
            "units": str(getattr(r, "_13", "") or ""),
            "kind": str(getattr(r, "_10", "") or ""),
            "equipment": set(str(getattr(r, "Equipment") or "").split(","))
            if isinstance(getattr(r, "Equipment", None), str) else set(),
        }
    return info


def eo66_index_safe(path="data/eo66.xlsx"):
    """Column-name based (robust to pandas' positional attribute names)."""
    eo = pd.read_excel(path)
    cols = {c: c for c in eo.columns}
    info = {}
    for _, row in eo.iterrows():
        d = row.get("Definition")
        if not isinstance(d, str):
            continue
        markers = row.get("Markers")
        display = row.get("Display Name")
        units = row.get("Units/Facets (Imperial)")
        kind = row.get("Kind/Type Order")
        equip = row.get("Equipment")
        info[d.strip()] = {
            "markers": frozenset(t.strip() for t in markers.split(",") if t.strip())
            if isinstance(markers, str) else frozenset(),
            "display": set(normalize_text(display).split()) if isinstance(display, str) else set(),
            "units": str(units) if isinstance(units, str) else "",
            "kind": str(kind) if isinstance(kind, str) else "",
            "equipment": {e.strip() for e in equip.split(",")} if isinstance(equip, str) else set(),
        }
    return info


def token_f1(a, b):
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if not inter:
        return 0.0
    p, r = inter / len(b), inter / len(a)
    return 2 * p * r / (p + r)


def display_similar_pairs(info, labels, threshold=0.75):
    pairs = {}
    labels = [l for l in labels if l in info]
    for i in range(len(labels)):
        a = labels[i]
        for j in range(i + 1, len(labels)):
            b = labels[j]
            if not (info[a]["equipment"] & info[b]["equipment"]):
                continue
            f1 = token_f1(info[a]["display"], info[b]["display"])
            if f1 >= threshold:
                pairs[tuple(sorted((a, b)))] = f1
    return pairs


def draft_relation(a, b, info):
    ta, tb = set(camel_tokens(a)), set(camel_tokens(b))
    ia, ib = info.get(a), info.get(b)
    marker_j = None
    units_match = kind_match = None
    if ia and ib:
        ma, mb = ia["markers"], ib["markers"]
        marker_j = len(ma & mb) / len(ma | mb) if (ma | mb) else 0.0
        units_match = bool(ia["units"]) and ia["units"] == ib["units"]
        kind_match = bool(ia["kind"]) and ia["kind"] == ib["kind"]
        display_sim = token_f1(ia["display"], ib["display"])
    else:
        display_sim = token_f1(ta, tb)
    if marker_j == 1.0 and units_match and kind_match:
        return "same", "both", marker_j, display_sim, units_match, kind_match
    if ta < tb:
        return "parent", "a_to_b", marker_j, display_sim, units_match, kind_match
    if tb < ta:
        return "parent", "b_to_a", marker_j, display_sim, units_match, kind_match
    diff = ta ^ tb
    if diff in QUALIFIER_TWINS or (len(diff) == 2 and diff in QUALIFIER_TWINS):
        return "site_variant", "both", marker_j, display_sim, units_match, kind_match
    if display_sim >= 0.75 or (marker_j is not None and marker_j >= 0.8):
        return "site_variant", "both", marker_j, display_sim, units_match, kind_match
    return "confusable", "both", marker_j, display_sim, units_match, kind_match


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/label_equivalences.csv")
    ap.add_argument("--min-confusions", type=int, default=2)
    ap.add_argument("--merge-existing", action="store_true",
                    help="Keep approved/rejected verdicts from the existing file")
    args = ap.parse_args()

    canon = build_label_canonicalizer("data/eo66.xlsx", "data/target_audit.csv")
    with open("data/label_mapping.json") as f:
        labels = list(json.load(f)["label2id"])
    info = eo66_index_safe()

    conf = confusion_pairs(canon)
    probe = probe_pairs(canon)
    cross = cross_site_pairs()
    display = display_similar_pairs(info, labels)

    candidates = defaultdict(lambda: {"evidence": set(), "n_confusions": 0, "sites": 0,
                                      "display_from_eo66": None})
    for pair, n in conf.items():
        if n >= args.min_confusions:
            candidates[pair]["evidence"].add("confusion")
            candidates[pair]["n_confusions"] = n
    for pair, n in probe.items():
        candidates[pair]["evidence"].add("probe")
        candidates[pair]["n_confusions"] = max(candidates[pair]["n_confusions"], conf.get(pair, 0))
    for pair, n in cross.items():
        candidates[pair]["evidence"].add("cross_site")
        candidates[pair]["sites"] = n
        candidates[pair]["n_confusions"] = max(candidates[pair]["n_confusions"], conf.get(pair, 0))
    for pair, f1 in display.items():
        candidates[pair]["evidence"].add("display_sim")
        candidates[pair]["display_from_eo66"] = f1

    existing = {}
    if args.merge_existing and os.path.exists(args.out):
        for r in csv.DictReader(open(args.out, encoding="utf-8")):
            existing[(r["label_a"], r["label_b"])] = r

    rows = []
    for (a, b), c in candidates.items():
        relation, direction, mj, ds, um, km = draft_relation(a, b, info)
        # Keep the table reviewable: structural twins (same/parent/site_variant)
        # need any evidence beyond a lone weak signal; confusable pairs only
        # survive with strong, repeated evidence
        strong = (c["n_confusions"] >= args.min_confusions or "probe" in c["evidence"]
                  or c["sites"] >= 3)
        if relation == "confusable":
            if not (c["n_confusions"] >= 10 or ("probe" in c["evidence"] and c["n_confusions"] >= 3)):
                continue
        elif not strong and c["evidence"] == {"display_sim"}:
            continue
        elif not strong and c["evidence"] == {"cross_site"}:
            continue
        row = {
            "label_a": a, "label_b": b, "relation": relation, "direction": direction,
            "evidence": ",".join(sorted(c["evidence"])), "n_confusions": c["n_confusions"],
            "sites": c["sites"], "marker_jaccard": "" if mj is None else round(mj, 3),
            "display_similarity": round(ds, 3), "units_match": "" if um is None else um,
            "kind_match": "" if km is None else km, "proposed_by": "draft_equivalences.py",
            "status": "draft", "reviewer": "", "notes": "",
        }
        prev = existing.get((a, b))
        if prev and prev.get("status") in ("approved", "rejected"):
            row["status"] = prev["status"]
            row["reviewer"] = prev.get("reviewer", "")
            row["notes"] = prev.get("notes", "")
            if prev.get("relation"):
                row["relation"], row["direction"] = prev["relation"], prev.get("direction", direction)
        rows.append(row)
    rows.sort(key=lambda r: (-(r["n_confusions"] + 3 * r["sites"]), r["label_a"], r["label_b"]))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    rel = Counter(r["relation"] for r in rows)
    ev = Counter(e for r in rows for e in r["evidence"].split(","))
    print(f"Drafted {len(rows)} candidate pairs -> {args.out}")
    print(f"  relations: {dict(rel)}")
    print(f"  evidence:  {dict(ev)}")
    print(f"  status:    {dict(Counter(r['status'] for r in rows))}")
    print("\nTop 25 by evidence weight:")
    for r in rows[:25]:
        print(f"  {r['label_a']:<26} ~ {r['label_b']:<26} {r['relation']:<12} {r['direction']:<6} "
              f"conf={r['n_confusions']} sites={r['sites']} mj={r['marker_jaccard']} "
              f"ds={r['display_similarity']} [{r['evidence']}]")


if __name__ == "__main__":
    main()
