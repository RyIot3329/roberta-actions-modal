# Live-service probe of the deployed autotagging model — 2026-07-13

Five test suites (110 points) run against the **deployed** model
(IOTechSystems/haystack-autotagging @ 02d91dc, June-12 DAPT fine-tune, 985
classes, T=1.141) through the live haystack-autotag MCP / ai-inference service
(normalization-parity + calibration build). Service identity verified by a
4-point sanity batch (OAT/CLG_VLV/SPACE_RH/SAT_SP → all correct, calibrated
confidences, `normalizedPoint` = repo `normalize_text` output).

Every point was checked against the repo's data: **0 of 110 normalized forms
appear in validation or test JSONL** (24 originally-planned canonical names
collided and were respelled to realistic alternates). 35/110 appear in train
(flagged per-row; results split below). Public-data suite uses the plastering
benchmark's **GHC building** — real point labels with published gold tagsets,
from a building never used anywhere in this repo (the DAPT corpus pulled only
SDH/SODA/IBM/UVA). The Kaggle dataset suggested (claytonmiller/…point-label-
examples) requires authenticated download; GHC is the freely-fetchable
equivalent in the same domain. Per-row data: `live_probe_20260713_rows.csv`
(gold, acceptables, verdict, rationale, seen-in-train, train-label lookup).

## Scores

| Suite | What it probes | Judged acc | Notes |
|---|---|---|---|
| T1 canonical abbreviations (25, imagined) | cross-vendor shorthand (RTN_AIR_TEMP, OAD_POS, HHW_VLV_CMD…) | **24/25 = 96%** | one qualifier hallucination |
| T2 GHC public building (29, real) | never-seen building, human-style labels | **22/29 = 75.9%** | errors cluster on request/meter/cryptic forms |
| T3 vendor-mangled names (20, imagined) | Niagara `$xx`, JCI/Siemens paths, glued `AHU12MAT`, indices | **18/20 = 90%** | normalization fix has largely neutralized this axis |
| T4 adjacency minimal pairs (22, imagined) | entering/leaving, pri/sec, twr/cond, Sp/Fb/Cmd | **19/22 = 86.4%** | valve cmd/fb axis is now solid; CDW polarity is not |
| T5 out-of-taxonomy (14, imagined) | lighting/elevator/weather/noise → should abstain | 0/14 abstained; **14/14 rejected at conf<0.35** | service never returns predicted_unknown |

Aggregate T1–T4: **86.5% judged** (83/96); 72.9% strict against pre-registered
golds only — the 13.5pp gap is almost entirely predictions landing on
*taxonomy twins* of the intended class (counted correct after review, see
finding 2). Seen-in-train rows: 97.1%; unseen rows: **80.3%**.

Confidence separates outcomes cleanly end-to-end (calibration is working):

| Outcome | mean | median | range |
|---|---|---|---|
| correct (n=83) | 0.802 | 0.845 | 0.23–0.94 |
| wrong (n=13) | 0.447 | 0.412 | 0.16–0.80 |
| out-of-taxonomy (n=14) | 0.152 | 0.170 | 0.04–0.31 |

Threshold sweep (this probe): τ=0.35 → 91.7% coverage @ 90.9% precision, 0/14
OOD admitted; τ=0.70 → 79.2% @ 97.4%. (Internal frozen-test triage previously
suggested ~0.8 for 90% precision on the harder site distribution — pick τ per
release from validation, but *some* τ should exist; see finding 5.)

## Findings (ranked)

1. **A majority-vote data conflict produces the worst error type: high-confidence
   wrong.** `return temp` → `zoneTemp` @ **0.80** (returnTemp exists and is the
   obvious gold). `data/label_conflicts.csv` records why: `zoneTemp (1008) |
   returnTemp (103) | secEnteringTemp (12) | evapEnteringTemp (2)` — one
   high-row site that labels return-air sensors as zone temp outvotes the
   semantically correct label 10:1, and weighted-majority resolution bakes the
   site idiosyncrasy into a universal surface form. Every site that names a
   point "Return Temp" now gets zoneTemp, confidently.

2. **Taxonomy twins fragment the label space.** 10/96 scoreable rows (10.4%)
   predicted a near-duplicate of the intended class. Confirmed coexisting in
   the deployed 985: {zoneTempHiSp, zoneTempMaxSp, zoneCoolSp} (a *triplet*
   for zone cooling setpoint), {zoneTempLoSp, zoneTempMinSp, zoneHeatSp},
   {filterAlarm, unitFilterAlarm}, {evapEnteringTemp, priEnteringTemp,
   secEnteringTemp} (loop position rarely decidable from the name),
   minOutsideDamper vs outsideDamperMinSp. Consequences: downstream consumers
   get different marker sets for identical concepts depending on which twin
   wins, and training probability mass splits 2–3 ways on classes that occur
   constantly in real sites. This is a *different* cleanup than the
   gate-failed junk-form consolidation (numbered variants, train-only tiny
   classes — mostly inert mass): twin merging consolidates mass on classes
   that actually appear in val/test, so it should move the gate metric.

3. **Condenser-water supply/return polarity is internally contradictory in the
   training data.** Train contains `ahu cdws sp` → condEnteringTempSp
   (supply = entering, classic chiller-centric) *and* `cdw ret flow` →
   condEnteringFlow (return = entering, the opposite polarity);
   label_conflicts additionally shows `cond water return temp` resolved to
   twrEnteringTemp. The model learned {sup→leaving, ret→entering} and applied
   it consistently (CDWS_T → condLeavingTemp 0.67, CDWR_T → condEnteringTemp
   0.66) — inverted vs the pre-registered ASHRAE-convention gold. Scored
   wrong, but the root cause is data: no convention was chosen, so no mapping
   the model picks can be right for both sites. Compact acronyms
   (`cdws t`/`cdwr t`) have zero train coverage.

4. **Under uncertainty the model hallucinates qualifiers instead of backing off
   to the base class.** 10 of 13 errors add a modifier absent from the name:
   KW_DEMAND→**zone**PowerReal, `oa dmpr fb`→**min**OutsideDamperFb,
   `airflow request`→dischargeFlow**CoolMax**Sp, CLG_STG_2→coolingStage**Status**,
   `flow control _ flow input`→dischargeFlow**Status**,
   EVAP_ISO_VLV_STS→evap**Entering**IsoValve (and dropped Status),
   MIN_OA_DMPR→outsideDamper**MinSp**. The correct base/generic class existed
   in every case. Only 3/13 errors are wrong-family (`run request`→
   chillerDemand, `prv command`→priEnteringValve, `return temp`→zoneTemp).
   The old "wrong-family" failure mode has become a "wrong-modifier" one —
   progress, and a much easier target.

5. **The service never abstains.** Pure noise (`XK7_QQZ_9911`, normalized
   `xk qqz`) returns secEnteringTemp @ 0.18; all 14 OOD probes got tags
   (lighting → zoneTempHeatDb, wind direction → outsideFan, elevator →
   condStatus…). The calibrated confidence *does* separate them (max OOD
   0.314) but nothing applies a threshold, so MCP/API consumers see junk tags
   for out-of-domain points unless they filter themselves. predicted_unknown
   effectively never fires.

6. **Extension classes ship no metadata.** Live-confirmed: zonePowerReal and
   twrIsoValve predictions return empty `tagMetadata` (bundle's haystack.csv
   covers eo66 definitions only). Consumers can't get markers for ~any
   non-eo66 class the model was extended with.

7. **What's working (keep it):** normalization parity end-to-end — `$xx`
   escapes, glued `AHU12MAT`, JCI `NAE01/FEC12.HTG-VLV-O`, Siemens `B3.AHU02:
   SF-SS`, lowercase, index stripping all resolved correctly (T3 90%); the
   valve cmd/fb/position axis and zone hi/lo setpoints are solid; transfer to
   a truly unseen building (GHC, no DAPT contamination) at 76% on
   human-readable labels; `stpt`/`setpt` variants handled.

## Recommendations

Ranked by expected lever size; 1–3 are data/taxonomy work in this repo.

1. **Fix universal-form conflicts (data, cheap, high yield).** Add
   label_overrides for surface forms whose conflict winner contradicts the
   form's own tokens (`return temp` → returnTemp first). Mechanical audit:
   scan label_conflicts.csv for rows where the winning label's eo66
   display-name/marker tokens share *no* token with the text while a losing
   label's do — those are site idiosyncrasies overriding semantics. Then
   change conflict weighting so one site can't win on raw row count: weight by
   per-site *unique names* (or √rows) and add a self-evidence prior (label
   whose tokens overlap the text wins ties).

2. **Merge taxonomy twins via eo66 Markers (taxonomy, the real cleanup).**
   Marker sets are near-identical inside each twin family — generate
   target_audit merge rows from marker-set equality/subset instead of only
   regex/typo matching, pick one canonical class per family, rerun the gated
   retrain. Expect a real gate move (unlike the junk-form cleanup, which was
   metric-neutral because those classes were train-only). Also teach
   `evaluate_external.py` alias credit for twin families so the gate stops
   mis-scoring twin hits either way.

3. **Pick a condenser-water convention and enforce it (data).** Canonicalize
   chiller-centric (sup/supply → condEntering\*, ret/return → condLeaving\*)
   across real_points via overrides; add synthetic/augmentation coverage for
   the compact acronyms (CDWS, CDWR, CWS, CWR, CWST, CWRT ± T/TEMP/FLOW). Flag
   name-undecidable cases (bare CHWS/CHWR pri-vs-sec, CDW polarity) as
   accept-either in eval.

4. **Attack qualifier hallucination (training/augmentation).** Augment with
   qualifier-stripped variants mapped to base classes (bare "kw" → powerReal,
   "oa dmpr fb" → outsideDamperFb) so bare forms have support competitive with
   their decorated siblings; optionally add an inference-time Occam re-rank
   (among top-k within a margin, prefer the class whose marker set is implied
   by the name tokens — could live in evaluate_external first as an
   experiment, no retrain needed).

5. **Ship an acceptance threshold (serving).** Store e.g.
   `config.acceptance_threshold` chosen on validation at the 90–95% precision
   operating point each release; service maps below-τ to predicted_unknown
   (which is what the MCP contract already documents). This probe: τ=0.35
   rejects 14/14 OOD at 91.7%/90.9% coverage/precision; the frozen-site
   distribution wants a higher τ — hence per-release selection, not a
   hardcode. Also regenerate bundle metadata for the full class space so
   extension classes stop returning empty tagMetadata.

6. **Freeze this probe as a regression suite (eval).** 110 rows with judged
   golds is cheap to run post-gate and catches exactly what aggregate site
   metrics hide: conflict-induced high-confidence regressions ('return temp'),
   twin drift, OOD overreach. Manifest + results CSV are in output/; wiring it
   into evaluate_external as `--probe` is ~an hour of work.

Caveats: n=110; gold labels are my judgment (documented per-row, twins and
undecidables credited/annotated explicitly); the probe skews easier than the
frozen Integ06+Motorola benchmark (64.7% test acc) because every name here has
readable semantics — its value is the per-axis decomposition, not the headline.
