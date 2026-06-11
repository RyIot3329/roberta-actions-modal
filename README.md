# RoBERTa Fine-tuning POC

A proof-of-concept for fine-tuning `FacebookAI/roberta-base` using Modal and GitHub Actions with automated data preprocessing.

## What This Does

1. **Data Preprocessing** - Cleans and converts CSV data to JSONL format
2. **GitHub Actions** triggers training (manually or on push to `data/` or `scripts/`)
3. **Modal** runs the fine-tuning job on a GPU (T4)
4. **Validation Testing** - Tests the model on validation data
5. **Results** are committed back to the repo with a timestamp
6. **Pull Request** is automatically created for review

## Pipeline Steps

```
┌──────────────────┐   ┌─────────────┐   ┌──────────────────────┐
│ Extract Real Data│──▶│ Clean Data  │──▶│ Site-grouped Split & │
│ (site exports)   │   │ (normalize) │   │  Convert to JSONL    │
└──────────────────┘   └─────────────┘   └──────────────────────┘
                                                    │
                                                    ▼
┌─────────────┐   ┌─────────────┐   ┌──────────────────────────┐
│  Create PR  │◀──│   Commit    │◀──│ Fine-tune on Modal       │
│             │   │   Results   │   │ + Quality-gated Hub push │
└─────────────┘   └─────────────┘   └──────────────────────────┘
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
- **Quality-gated deployment**: the Hub push is skipped when the new model's
  test F1 is below the previous best (`output/best_metrics.json`), so a bad
  run can never overwrite the production model.

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

Push changes to the `data/` or `scripts/` directory:

```bash
# Update your training data
git add data/train_all.csv
git commit -m "Update training data"
git push
```

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

## Preprocessing Scripts

| Script                | Input              | Output                   | Description                                        |
| --------------------- | ------------------ | ------------------------ | -------------------------------------------------- |
| `clean_data.py`       | `train_all.csv`    | `cleaned_data.csv`       | Normalizes point names (word-boundary preserving)  |
| `print_tags.py`       | `cleaned_data.csv` | Console                  | Shows label distribution                           |
| `convert_to_jsonl.py` | `cleaned_data.csv` | `*.jsonl` + mappings     | Dedupes, resolves conflicts, 80/10/10 split        |

## Notes

- Place your raw CSV data in `data/train_all.csv`
- All preprocessing happens automatically in the workflow
- Generated files are committed back to the repo via PR
- Modal's free tier includes GPU credits for testing
