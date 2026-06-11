# Autotagging Model — Accuracy & Efficiency Audit (2026-06-11)

Multi-agent audit of the DeBERTa-v3-base point-name autotagger. All quantitative
claims below were independently recomputed from repo files by adversarial
verification agents (18/18 verified).

## TL;DR

The model is not failing for model reasons. The 97.3% test accuracy is an
artifact of evaluating a synthetic template grammar against itself; on the real
Niagara N4 site export it scores 47.0% row-weighted / 15.4% on unique point
names. The error decomposition on unique names:

| Cause | Share of errors | Fixable by |
|---|---|---|
| True label outside the 343-class taxonomy | 64.0% | Taxonomy expansion (mostly already in eo66.xlsx) |
| OOV tokens: equipment prefixes/indices, site codes, `$xx` escapes | 32.5% | Normalization + augmentation |
| True in-vocab generalization failure | 3.6% | Model improvements |

A perfect 343-class model caps at 75.7% row / 45.9% unique on this site.
Covering eo66 fully raises the ceiling to 99.5% row / 93.3% unique.

## Verified key facts

- Training data is a synonym cross-product grammar: 15,459 unique texts from
  ~25 distinct tokens/class; vocab 2,209 words. Only 326/1,226 external words
  appear in it; only 20.1% of external uniques are fully in-vocab.
- Test-set leakage: 86.4% of test texts are within 1 token-edit (94.6% within a
  2-token set-difference) of a same-label train text → test acc measures
  template memorization.
- 142 missing external labels are verbatim eo66 Definitions (18,765 rows,
  21.1% of the site). Top: heatingValve (5,869 rows), dischargeFlowMaxSp/MinSp
  (3,105), heatingDemand/coolingDemand (621).
- `data/target_audit.csv` is fully verified (29/29 counts). Applying its 3
  renames (dischargeFanEnable→dischargeFan, returnFanEnable→returnFan,
  condWaterLeavingFlow→condLeavingFlow) recovers +6.15pp row-weighted at zero
  model risk. Case-dup classes confirmed: zoneTempMaxSP/Sp, zoneTempMinSP/Sp,
  emergencyShutdown/emergencyStop all exist as separate class IDs.
- The prior stripping experiment (external_eval_n4_points_stripped.csv) proves
  prefix causality: unique acc 15.4%→26.3% (437 fixed vs ~50 regressed);
  in-taxonomy changed uniques 17.0%→51.8%.
- 124 unique names carry Niagara `$xx` escapes ($20/$2d/$2f) through
  normalize_text (no `$` handling); they score 8.9%.
- Confidence: in-taxonomy row precision is 98.7% at conf≥0.999 (47% coverage);
  the apparent miscalibration at ≥0.99 is mostly out-of-taxonomy labels. Top-20
  error names cover 65.8% of all error rows — a deduplicated review queue is
  small.
- Process gap: finetune.py pushes EVERY run to the production HF repo — a
  56.5%-val-accuracy run (output/20260609_204338) overwrote it on 2026-06-09;
  any push to scripts/** retriggers this (train.yml paths).
- Errors are wrong-family, not wrong-modifier: only ~5% of in-taxonomy unique
  errors are same-family. A hierarchical family→modifier classifier buys ~3pp;
  don't build it.
- Edge efficiency: a TF-IDF(char 2-4 + word 1-2)+LinearSVC trained in 2.2s
  beats the deployed 184M transformer externally (54.6% vs 47.0% row; 25.9% vs
  15.4% unique) at 17µs/point and <50MB. DeBERTa-v3 is a poor edge family:
  53-70% of params are the 128k-vocab embedding and disentangled attention is
  ~3.1× slower than same-shape BERT at batch=1 on CPU.

## Recommendations (ranked)

### Phase 0 — this week, no retraining risk
1. **Fix `normalize_text` (scripts/clean_data.py:24-34)**: decode `$hh` hex
   escapes BEFORE separator splitting; add letter↔digit boundary splits;
   map equipment-index tokens (`ch1`, `ahu13`, `ut0032`, `ai1003`) to their
   alpha stem (keep `ch`, drop the digits) rather than deleting them (total
   removal caused the 50 regressions in the stripping experiment). Share the
   exact same function between training and inference.
2. **Apply data/target_audit.csv merges/renames** (343→339 classes) and retrain.
   +6.15pp row-weighted external accuracy from renames alone; removes 10/130
   training label conflicts.
3. **Add a CI quality gate**: only push to HF if external frozen-slice accuracy
   and test F1 beat the previous best; drop `scripts/**` from push triggers
   (or require a `[train]` commit token).
4. **Edge short-circuits**: dedupe names per site before inference (24.7×
   fewer forwards); exact-match LUT of the 15,459 training texts (~1.5MB,
   22.2% row coverage at 98.3%, identical to the model on those rows).
5. **Operating point**: auto-accept at conf≥0.999, deduplicated human review
   queue ranked by row count. Today that is 47% of rows at 85% raw precision
   (98.7% in-taxonomy); clearing the top-20 queue names resolves 65.8% of
   error rows.
6. **Label canonicalization in eval**: use eo66's Regular Expression column to
   canonicalize gold labels (heatingStage01→heatingStage; 2,484 rows). Note
   these regexes are useless as a raw-name matcher (0.1% correct) — only as a
   label canonicalizer.

### Phase 1 — data (the big lever)
7. **Add the 142 missing eo66 classes** (top-20 cover ~19K of 21.7K unreachable
   rows), generating templated texts the same way as existing classes
   (~9.1K new rows). Raises the row ceiling 75.7%→96%+ on this site.
8. **Prefix/index/site-code augmentation**: emit training variants with
   prepended equipment+ordinal tokens, site codes, BACnet refs, `$xx`-escaped
   separators, token dropout, separator swaps. Targets the 32.5% OOV error
   share (~+15-19pp unique expected).
9. **LLM-label the 3,606 unique N4 names** (retrieval over eo66 Display
   Names/Markers, output restricted to the taxonomy; human spot-check ~200) and
   fold in as real training + a true external benchmark. Strongest-evidence
   path to 85%+ row-weighted (BMS-RAG near-100% on real datasets; GPT-4 96-98.5%
   on the analogous TG-263 task; LLM-label-trained classifiers match
   human-label-trained, arXiv 2406.17633).
10. **Fix the evaluation protocol**: grouped split by template family
    (token-set/Jaccard groups) instead of exact-text dedup; expect headline
    test acc to drop toward the honest number; make external unique-text
    accuracy the primary model-selection metric.
11. **Training hygiene** (scripts/finetune.py): label_smoothing_factor=0.1;
    temperature-scale on a real-site calibration split; remove
    ignore_mismatched_sizes (line 158) + assert num_labels; persist the
    per-class classification_report (lines 301-308, currently lost every run);
    seed control + 3-seed variance for headline numbers; metric_for_best_model
    f1_macro; epochs ~24-30 so cosine actually anneals (currently stops at
    41/60 leaving LR at 23% of peak); unify the 3 divergent default sets
    (finetune.py / train.yml / training.yml).

### Phase 2 — model & architecture
12. **DAPT**: continued MLM pretraining on real point-name corpora — the 89k N4
    dump + gtfierro/point-label-sharing (103,064 names, 92 buildings, BSD-3) +
    Mortar. Best published cross-site name-only result (>70% on 30 held-out
    buildings, Waterworth 2021) came from exactly this.
13. **Marker multi-label head (larger bet)**: predict the eo66 Markers set
    (396-dim sigmoid) and map set→definition; 96.5% of non-blacklist eo66
    definitions have a unique marker set (collisions are electrical phase
    variants, fixable with ~10 discriminator tags). Full 3,794-definition
    coverage with compositional generalization, no per-class synthetic data
    (flat softmax would need ~243K rows). Existing data converts free via the
    target→Markers join.
14. **Per-site adaptation loop**: cluster unmapped names at commissioning,
    label ~100-160 representative ones (Scrabble's number for 99%), SetFit-style
    quick retrain or k-NN fallback for low-confidence points.
15. **Don't build**: hierarchical family→modifier classifier (+3pp ceiling);
    zero-shot TF-IDF retrieval over eo66 (9.1% top-1, far below the classifier);
    eo66 regexes as a raw-name first pass.

### Edge efficiency track (orthogonal to accuracy — pick per need)
- **Stopgap**: ONNX export (optimum, deberta-v2 arch supported) + ORT dynamic
  INT8: 739MB→~190MB, expected 2-4× CPU speedup. Benchmark the artifact —
  DeBERTa's gather-heavy attention sometimes regresses under INT8.
- **Baseline to beat**: TF-IDF+LinearSVC (<15MB pruned, 17µs/point) currently
  beats the transformer externally; keep it as a permanent benchmark gate.
- **End state**: distill to a 4L×256 WordPiece student (bert-mini class, 11.3M,
  45MB fp32 / ~11MB INT8, 0.6-1.9ms/point ≈ >50× faster), transfer set =
  training texts + the 3,606 real N4 names with teacher soft logits. Expected
  ~94-96.5% in-distribution retention. Do NOT distill into deberta-v3-xsmall
  (3.1× CPU architecture penalty, 70% embedding params, tokenizer packaging
  issues). Distill AFTER the data fixes, or you clone today's external weakness.
- If staying on DeBERTa: prune the 128k embedding to ~16k observed pieces
  (TextPruner-style): 739MB→~390MB fp32 with bit-identical logits on covered
  inputs.

## Measured CPU ladder (4 threads, seq32, i9-13900HX; industrial edge ~2-4× slower)

| Model | Params | fp32 disk | b1 latency | INT8 b1 |
|---|---|---|---|---|
| deberta-v3-base (current) | 184.7M | 739MB | 97.1ms | 55.5ms |
| deberta-v3-small | 142.2M | 569MB | 45.2ms | 28.2ms |
| deberta-v3-xsmall | 71.0M | 284MB | 28.1ms | 21.4ms |
| bert-small 6L×768 | 67.2M | 269MB | 14.5ms | 8.3ms |
| bert-mini 4L×256 (student target) | 11.3M | 45MB | 1.8ms | 1.9ms |
| TF-IDF + LinearSVC | — | <50MB (<15MB pruned) | 0.017ms | — |

## Expected trajectory on the N4 site (row-weighted)

47.0% today → ~53% (audit renames) → ~75%+ ceiling unlocked by taxonomy
expansion + normalization/augmentation → 85%+ with LLM-labeled real data and
DAPT. Caveat: one labeled site is doing triple duty (eval, calibration,
augmentation source) — acquire additional site exports and keep one untouched
as a final holdout.

---

## Addendum (2026-06-11, later): real labeled data added — measured impact

`data/real_data/` now holds 7 labeled site exports (~476k rows, 15.3k unique
point names): N4-Integ01 (= the prior eval site, 3,606/3,607 name overlap),
five genuinely new N4 sites, and a Motorola BACnet export. 98.3% of rows
canonicalize into eo66 (`scripts/extract_real_data.py` → `data/real_points.csv`;
eo66 regexes resolve 1,448 numbered-variant labels). Within-site label
conflicts: 3.8% of unique texts. Cross-site name overlap is only 6–18%.

Leave-one-site-out experiment (TF-IDF char/word + LinearSVC, audit renames
applied to synthetic labels; train = synthetic and/or the 6 other sites,
test = held-out site):

| Metric (mean over 7 sites) | synth only | synth + real | real only |
|---|---|---|---|
| Unique-name accuracy | 33.1% | **59.1%** | 58.7% |
| Row-weighted accuracy | 46.1% | **69.5%** | 68.4% |
| Label coverage of held-out site | 35–55% | 95–97% | 95–97% |

On Integ01 (the old benchmark): 82.9% row-weighted vs the deployed
transformer's 47.0%. Real data roughly doubles cross-site accuracy even with
a 17µs linear model; the remaining gap is in-taxonomy name transfer
(~52–66% unique), which is what the transformer + augmentation + DAPT should
close further.

Motorola insight: 51.7% of its names are bare BACnet object IDs (`AV 00`)
with no semantic content, but `BACnet Description` is clean English
("Heating Signal"). Inference rule "use description when the name is a bare
object ref" lifts Motorola row accuracy 49.7% → 84.3%. The production tagger
should accept name + optional description.

Updated next steps: (1) site-grouped train/val/test split in
convert_to_jsonl.py (hold out one N4 site + Motorola as frozen benchmarks);
(2) retrain on synthetic + real; (3) CI quality gate BEFORE pushing anything —
note any push touching `data/**` or `scripts/**` currently triggers a retrain
that overwrites the production HF model.

---

## Addendum (2026-06-11, later still): Phase 0 + site-grouped split IMPLEMENTED

All Phase-0 items and the site-grouped split are now wired in:

- `scripts/clean_data.py` — normalize_text decodes Niagara `$xx` escapes,
  treats every non-word run as a separator, strips equipment-index digits
  (`AHU13_SaTmp`→`ahu sa tmp`, `CWP01Flt`→`cwp flt`) with a species/refrigerant
  preserve list (`co2`, `pm25`, `r134a`, glued `returnco2level`→`returnco2
  level`), keeps single-letter+digit phase tokens (`l1`), falls back for
  all-index names (`AV 00`→`av`). Idempotent; shared by train + inference.
- `scripts/extract_real_data.py` — site exports → `data/real_points.csv`
  (aggregated site,name,label_raw,label,rows; 1.1MB; eo66-regex label
  canonicalization).
- `scripts/convert_to_jsonl.py` — site-grouped split (train = synthetic +
  Integ01-04; val = Integ05; test = Integ06 + Motorola, frozen). Applies
  target_audit merges/renames + eo66 canonicalization to ALL labels,
  weighted-majority conflict resolution (real row counts as evidence),
  re-normalizing override loader, coverage + seen_in_train reporting,
  `*_uncovered.jsonl` for out-of-label-space gold.
- `scripts/finetune.py` — quality gate (skips Hub push when test f1 <
  `output/best_metrics.json` baseline; baseline updates only on gated push),
  label_smoothing plumbed (config: 0.1).
- `.github/workflows/train.yml` — `scripts/**` no longer triggers training;
  `data/real_points.csv` does; label_smoothing passed through.
- `scripts/evaluate_external.py` — gold + prediction canonicalization (alias
  credit), default threshold 0.999, deduplicated review-queue CSV, native
  real_points.csv `--site` mode, local-cache fallback.

New honest baselines (TF-IDF+LinearSVC floor; 901 classes, 21,513 train texts):
validation (Integ05) 52.1% unique / 71.5% row-weighted; test (Integ06+Motorola)
60.9% unique / 79.0% row-weighted. Old deployed DeBERTa on frozen Integ06:
21.2% unique / 64.5% row-weighted. The retrained transformer must beat the
linear floor; `output/best_metrics.json` starts unset so the first new-protocol
run establishes the gate baseline.
