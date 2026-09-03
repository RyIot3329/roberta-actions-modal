# Experiment log

One line per gated run. "Strict" = exact-label accuracy on the frozen per-name test
(N4-Integ06 626 + Motorola 204 texts); "lenient" credits the accept set (labels any
train site used for the identical name + approved equivalences); numbers come from
`output/<stem>_metrics.json` (see `scripts/summarize_run.py`). Seeds are trained in
parallel; the candidate is the median-validation seed unless the greedy weight soup
beats it on validation.

| date (UTC) | run | change vs previous | seeds val strict | seeds test strict | ensemble / soup | candidate | gate | notes |
|---|---|---|---|---|---|---|---|---|
| 2026-06-12 | deployed (June) | DAPT encoder, 985 classes | 63.97 | 64.73 (f1w 0.6345) | – | seed 42 | pushed | re-scored on the 2026-09-03 scoreboard: 64.46 strict / 69.52 lenient / 72.34 log1p-rows |
| 2026-07-13 | 29276489603 / 29283249956 | label-space cleanup (948 classes) | 63.70 / 64.36 | 63.98 / 63.49 | – | – | failed (stale v1 baseline) | the ~1pp "drop" was a scoreboard change, not a regression |
| 2026-09-03 16:09 | 33774113288 | de-glue normalization (= production v2), capped per-site conflict votes (row_cap 100, 28 flips + 83 recovered ties), lenient accept sets, composite gate v2, 3 seeds | 64.90 / 65.28 / 64.33 | 66.63 / 64.22 / 63.01 | ensemble 66.27 (lenient 71.33) | seed 42 (median val) | PASSED: +2.17pp [-0.12, +4.46], P(worse) 0.03 | pushed to RyIoT33/haystack-autotagging; seed spread on test 3.6pp; linear floor 56.5 |
| 2026-09-03 16:43 | 33780299432 | context-aware training (27.5k `name | context` records, context dropout 0.5 / field dropout 0.2, max_seq_length 64), greedy soup | pending | pending | pending | pending | pending | gate primary stays the name-only per-name view; context views reported on the 4,593 test pairs |

## Queued A/B sequence (one knob per push, 3 seeds each, adopt on >= +0.5pp validation mean)

1. **A – finish the anneal**: `early_stopping_patience: 0`, `epochs: 20` (today the best checkpoint is taken at ~70% of peak LR).
2. **B – layer-wise LR decay**: `llrd_decay: 0.9`, `head_lr_multiplier: 10`.
3. **E – EMA**: `ema_decay: 0.999`.
4. **C – R-Drop**: `rdrop_alpha: 1.0` with `label_smoothing: 0.05` (2x step cost).
5. **D – logit adjustment** (`logit_adjustment_tau: 0.5`): judged on macro-F1 only, never on the primary.
6. **H – deberta-v3-large probe**, then distillation into the DAPT base if it clears +1.5pp.

Evidence gathered offline (2026-09-03, not runs): the Occam qualifier re-rank gives
+0.2-0.3pp (not adopted); per-site kNN / prior adaptation with 100-150 labels gives
-3..+2pp (not a lever); condenser-water polarity is inconsistent inside sites
(`data/cdw_polarity_review.csv`, decision row `cdwPolarityConvention`).
