# RoBERTa Fine-tuning POC

A proof-of-concept for fine-tuning `FacebookAI/roberta-base` using Modal and GitHub Actions with automated data preprocessing.

## What This Does

1. **Data Preprocessing** - Normalizes point names and builds site-grouped JSONL splits
2. **GitHub Actions** triggers training (manually, or on push to the raw inputs:
   `train_all.csv`, `real_points.csv`, `label_overrides.csv`, `config/training.yml`)
3. **Modal** runs the fine-tuning job on a GPU
4. **Held-out-site Evaluation** - Scores the model on buildings it never saw
5. **Quality-gated Hub push** - The model ships only if it beats the previous best
6. **Results** are committed back via an automatically created Pull Request

## Pipeline Steps

```
┌──────────────────┐   ┌─────────────┐   ┌─────────────────────────┐
│ Extract Real Data│──▶│ Clean Data  │──▶│ Site-grouped Split,     │
│ (site exports)   │   │ (normalize) │   │ Augment & Convert to    │
└──────────────────┘   └─────────────┘   │ JSONL                   │
┌──────────────────┐                     └─────────────────────────┘
│ Generate Synthetic│ (offline, committed)          │
│ (eo66 Display     │──────────────────▲            ▼
│  Names, thin      │   ┌─────────────┐   ┌──────────────────────────┐
│  classes)         │   │ Create PR ◀─│───│ Fine-tune on Modal       │
└──────────────────┘   │  + Commit    │   │ + Temperature scaling    │
                        └─────────────┘   │ + Quality-gated Hub push │
                                          └──────────────────────────┘
```

Key data-quality guarantees:

- **Normalization is deployment-faithful**: Niagara `$xx` slot-path escapes are
  decoded (`NGT$20CLG$20STPT` → `ngt clg stpt`), word boundaries are preserved
  (`zoneCO2Sp` → `zone co2 sp`), and equipment-index digits are stripped
  (`AHU13_SaTmp` → `ahu sa tmp`) while chemical species (`co2`, `pm25`) and
  phase tokens (`l1`) survive. The same `normalize_text` runs at training and
  inference time.
- **Site-grouped split**: whole real sites are held out for validation and test
  (`convert_to_jsonl.py --val-sites/--test-sites`), so the metrics measure
  cross-site generalization — tagging a building the model has never seen —
  instead of template memorization. Coverage (gold labels outside the training
  label space) and seen-in-train fractions are reported in
  `data/dataset_summary.json`.
- **Labels are canonicalized**: eo66 numbered variants collapse to their base
  definition (`heatingStage01` → `heatingStage`) via eo66's own regex column,
  and `data/target_audit.csv` merge/rename decisions are applied everywhere.
- **Conflicting labels are resolved by weighted majority**: each real-site
  occurrence counts as evidence, ties are dropped, and every conflict is logged
  to `data/label_conflicts.csv`.
- **Manual overrides**: to fix conflicts (or any mislabel) by hand, edit the
  `resolution` column of `data/label_conflicts.csv` (set the correct label, or
  `DROP` to exclude) and save it as `data/label_overrides.csv`. Overrides always
  win over majority-vote resolution; texts are re-normalized on load so the
  file survives normalization changes.
- **Thin classes get generated support**: classes the real sites use but the
  training pool barely covers receive abbreviation variants generated from
  eo66 Display Names (`scripts/generate_synthetic.py` →
  `data/synthetic_points.csv`, weight 0.5 so real evidence always outranks
  them). The abbreviation dictionary (`data/abbreviations.csv`) is a
  human-reviewed artifact with a `--mine` refresh mode.
- **Train-time augmentation**: perturbed variants (equipment-stem prefixes,
  token dropout, adjacent swaps) teach prefix invariance; variants colliding
  with held-out site texts are dropped so the seen-in-train reporting stays
  honest. Knobs live under `preprocessing:` in `config/training.yml`.
- **Calibrated confidences**: a temperature fitted on validation is stored in
  the model config (`calibration_temperature`) and applied everywhere
  confidences are reported, so triage thresholds mean what they say.
- **Quality-gated deployment**: the Hub push is skipped when the new model's
  test F1 is below the previous best (`output/best_metrics.json`), so a bad
  run can never overwrite the production model. The baseline records the test
  size and class count; a PR that changes the test composition (new holdout
  sites, big coverage shifts) should deliberately delete the baseline file so
  the next push re-seeds it against the new scoreboard.

## Project Structure

```
roberta-poc/
├── .github/workflows/
│   └── train.yml              # CI/CD pipeline
├── data/
│   ├── train_all.csv          # Input: synthetic training data (text, target)
│   ├── real_data/             # Input: labeled site exports (.xlsx, see below)
│   ├── eo66.xlsx              # Input: EO66 tag taxonomy (source of truth)
│   ├── target_audit.csv       # Input: taxonomy merge/rename decisions
│   ├── real_points.csv        # Step 0 output (aggregated real points)
│   # Generated files:
│   ├── cleaned_data.csv       # Step 1 output
│   ├── train.jsonl            # Step 3 output (synthetic + training sites)
│   ├── validation.jsonl       # Step 3 output (held-out site, covered labels)
│   ├── test.jsonl             # Step 3 output (frozen sites, covered labels)
│   ├── *_uncovered.jsonl      # Step 3 output (gold labels outside label space)
│   ├── label_mapping.json     # Step 3 output
│   ├── dataset_summary.json   # Step 3 output (incl. coverage / seen-in-train)
│   └── label_conflicts.csv    # Step 3 output (conflicting labels for review)
├── scripts/
│   ├── extract_real_data.py   # Step 0: Extract labeled points from site exports
│   ├── clean_data.py          # Step 1: Normalize point names
│   ├── print_tags.py          # Step 2: Show label distribution
│   ├── convert_to_jsonl.py    # Step 3: Canonicalize, site-grouped split, JSONL
│   ├── finetune.py            # Modal fine-tuning script (quality-gated push)
│   └── evaluate_external.py   # Score a model against any labeled site export
├── output/                     # Training results (auto-generated)
│   └── best_metrics.json      # Quality-gate baseline (best pushed model)
├── requirements.txt           # Python dependencies
└── README.md
```

## Setup

### 1. Create Modal Account

Sign up at [modal.com](https://modal.com) (free tier available).

### 2. Get Modal Tokens

```bash
pip install modal
modal token new
cat ~/.modal.toml  # Copy token-id and token-secret
```

### 3. Add GitHub Secrets

Go to your repo → Settings → Secrets → Actions, and add:

| Secret               | Description          |
| -------------------- | -------------------- |
| `MODAL_TOKEN_ID`     | From `~/.modal.toml` |
| `MODAL_TOKEN_SECRET` | From `~/.modal.toml` |

### 4. Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/roberta-poc.git
git push -u origin main
```

## Usage

### Manual Trigger

1. Go to **Actions** tab in GitHub
2. Select **Fine-tune RoBERTa** workflow
3. Click **Run workflow**
4. Optionally set epochs and batch size
5. Click **Run workflow**

### Automatic Trigger

A push to `main` that touches any raw training input starts a run:
`data/train_all.csv`, `data/real_points.csv`, `data/label_overrides.csv`,
or `config/training.yml`. Generated files merged back via results PRs do
not re-trigger, and neither do script edits (retrain after code changes
via the Actions tab).

```bash
# Update your training data
git add data/train_all.csv
git commit -m "Update training data"
git push
```

## Adding a New Site's Labeled Data

CI never reads the `.xlsx` exports directly — it consumes the committed
`data/real_points.csv`. One manual extraction step sits between dropping an
export in the folder and a retrain firing. That's deliberate: it gives you a
review checkpoint (and a diff to eyeball in the PR) before any GPU run or
model push happens.

```bash
# 1. Drop the export into the folder
cp ~/Downloads/NewSite.xlsx data/real_data/

# 2. Regenerate the aggregated extraction
python scripts/extract_real_data.py

# 3. (Recommended) Dry-run the split locally and review the effects
python scripts/clean_data.py
python scripts/convert_to_jsonl.py

# 4. Commit the extraction -- this push starts the training run by itself
git add data/real_points.csv "data/real_data/NewSite.xlsx"   # xlsx optional
git commit -m "Add NewSite labeled points"
git push
```

What each step does, and what to check:

1. **Drop the export.** Three column layouts are auto-detected (see the table
   under *Input Data Format*). A file with unrecognized columns is **skipped
   with a warning**, not an error — read the extraction output. Supporting a
   new vendor layout is a one-line addition to `FORMATS` in
   `scripts/extract_real_data.py`.
2. **Extraction** (`extract_real_data.py`) rebuilds `data/real_points.csv`
   from the whole folder — a deterministic full rebuild, so there is no
   incremental state to manage. It collapses duplicate names into per-name
   row counts (used later as evidence weights) and canonicalizes labels via
   eo66's regex column (`heatingStage01` → `heatingStage`). Check the printed
   diagnostics: rows and unique names per site, and the "labels outside eo66"
   list — a spike there usually means the site uses extension labels that
   deserve a `data/target_audit.csv` decision.
3. **Local dry run** is the review checkpoint. The new site's row counts join
   the weighted-majority label vote, so skim the new entries in
   `data/label_conflicts.csv` (disagreements are resolved by evidence weight
   and logged; `data/label_overrides.csv` wins if you disagree with a
   resolution). In `data/dataset_summary.json`, check the class count and
   the val/test coverage figures — a new training site can introduce classes
   that move previously-uncovered held-out points into the scored test set.
4. **Commit and push.** `data/real_points.csv` is a workflow trigger path, so
   the push starts a training run on its own. Sites are routed by exception:
   anything not named in `--val-sites` / `--test-sites` lands in the
   **training pool automatically** — a new site needs no configuration to be
   trained on. The quality gate still applies: if the new data somehow makes
   the model worse, the Hub push is skipped and the results PR shows the gate
   failure. Committing the `.xlsx` itself is optional (the CSV is what CI
   uses); commit it if you want the raw export versioned alongside.

To make a new site a **holdout instead of training data**, pass it explicitly:
`python scripts/convert_to_jsonl.py --test-sites N4-Integ06,Motorola_Points,NewSite`
(and remember: changing the test sites changes the scoreboard — deliberately
delete `output/best_metrics.json` in the same commit so the quality-gate
baseline re-seeds against the new benchmark).

### Local Testing

```bash
pip install -r requirements.txt
pip install modal

# Run preprocessing locally
python scripts/extract_real_data.py   # when site exports in data/real_data/ change
python scripts/clean_data.py
python scripts/print_tags.py
python scripts/convert_to_jsonl.py    # --val-sites / --test-sites to change holdouts

# Run training on Modal
modal run scripts/finetune.py --epochs 2

# Score any model against a held-out site export
python scripts/evaluate_external.py --csv data/real_points.csv --site N4-Integ06
```

## Output Format

Results are saved to `output/YYYYMMDD_HHMMSS_roberta-base.txt`:

```
============================================================
RoBERTa Fine-tuning Results
============================================================

Timestamp: 2024-01-15T10:30:00
Model: FacebookAI/roberta-base
GPU: Tesla T4

Configuration:
  epochs: 2
  batch_size: 8
  learning_rate: 2e-05
  num_labels: 5
  train_samples: 20
  val_samples: 5

Training Metrics:
  loss: 0.1234
  runtime_seconds: 45.67

Evaluation Metrics:
  accuracy: 0.8000
  f1_weighted: 0.7856
  loss: 0.2345
```

## Input Data Format

Synthetic training data lives in `data/train_all.csv` with two columns:

```csv
text,target
ZONE_TEMP_SP,zoneTempSp
discharge-temp,dischargeTemp
RAT,returnTemp
```

- **text**: The raw point name to classify
- **target**: The standardized label

Labeled real site exports go in `data/real_data/*.xlsx`. Three formats are
auto-detected by `scripts/extract_real_data.py` (point name = last slot-path
segment for the N4 styles):

| Format | Point name column | Label column |
| ------ | ----------------- | ------------ |
| N4 style A | `pointPath from BAS` | `EO66 Point` |
| N4 style B | `proxyExt.pointId/BASpointName` | `pointTag/EO66` |
| BACnet export | `Bacnet Name` | `eo66Def` |

The extraction writes `data/real_points.csv` (aggregated with per-name row
counts), which is what the pipeline and CI consume — commit it whenever the
exports change. New sites land in the training pool automatically; promote
them to validation/test with `convert_to_jsonl.py --val-sites/--test-sites`.
Full walkthrough with review checkpoints: *Adding a New Site's Labeled Data*
above.

## Preprocessing Scripts

| Script                 | Input                                  | Output                 | Description                                              |
| ---------------------- | -------------------------------------- | ---------------------- | -------------------------------------------------------- |
| `extract_real_data.py` | `real_data/*.xlsx`                     | `real_points.csv`      | Extracts + aggregates labeled site points (manual step)  |
| `generate_synthetic.py`| `eo66.xlsx` + `abbreviations.csv`      | `synthetic_points.csv` | Generates texts for thin classes (manual step; `--mine` refreshes the dictionary) |
| `clean_data.py`        | `train_all.csv`                        | `cleaned_data.csv`     | Normalizes point names ($xx decode, index stripping)     |
| `print_tags.py`        | `cleaned_data.csv`                     | Console                | Shows label distribution                                 |
| `convert_to_jsonl.py`  | `cleaned_data.csv` + `real_points.csv` + `synthetic_points.csv` | `*.jsonl` + mappings | Canonicalizes labels, resolves conflicts, site-grouped split, augmentation |

## Notes

- Synthetic training data goes in `data/train_all.csv`; labeled site exports
  go in `data/real_data/` (see *Adding a New Site's Labeled Data* — one manual
  extraction step is required before they enter the corpus)
- Steps 1–3 of preprocessing run automatically in the workflow; extraction
  (step 0) is local-only by design
- Generated files are committed back to the repo via PR
- Modal's free tier includes GPU credits for testing
