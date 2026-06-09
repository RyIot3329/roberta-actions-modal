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
┌─────────────┐   ┌─────────────┐   ┌──────────────────┐
│  Clean Data │──▶│ Print Tags  │──▶│ Dedupe, Split &  │
│ (normalize) │   │             │   │ Convert to JSONL │
└─────────────┘   └─────────────┘   └──────────────────┘
                                             │
                                             ▼
┌─────────────┐   ┌─────────────┐   ┌──────────────────┐
│  Create PR  │◀──│   Commit    │◀──│   Fine-tune on   │
│             │   │   Results   │   │ Modal + Test Eval│
└─────────────┘   └─────────────┘   └──────────────────┘
```

Key data-quality guarantees:

- **Normalization preserves word boundaries**: `CDK_DMPR_STATUS` → `cdk dmpr status`,
  `zoneCO2Sp` → `zone co2 sp` (instead of stripping spaces).
- **No train/val/test leakage**: data is deduplicated to one row per unique text
  before splitting, and the split is assert-checked for overlap.
- **Conflicting labels are resolved**: texts mapped to multiple targets keep the
  majority label (ties dropped) and are logged to `data/label_conflicts.csv`.
- **Manual overrides**: to fix conflicts (or any mislabel) by hand, edit the
  `resolution` column of `data/label_conflicts.csv` (set the correct label, or
  `DROP` to exclude) and save it as `data/label_overrides.csv`. Overrides always
  win over majority-vote resolution and pushing the file triggers retraining.
- **Held-out test set**: validation drives early stopping / model selection;
  the test set is only touched once at the end and provides the headline metrics.

## Project Structure

```
roberta-poc/
├── .github/workflows/
│   └── train.yml              # CI/CD pipeline
├── data/
│   └── train_all.csv          # Input: Raw training data (text, target)
│   # Generated files:
│   ├── cleaned_data.csv       # Step 1 output
│   ├── train.jsonl            # Step 3 output
│   ├── validation.jsonl       # Step 3 output
│   ├── test.jsonl             # Step 3 output
│   ├── label_mapping.json     # Step 3 output
│   ├── dataset_summary.json   # Step 3 output
│   └── label_conflicts.csv    # Step 3 output (conflicting labels for review)
├── scripts/
│   ├── clean_data.py          # Step 1: Normalize point names
│   ├── print_tags.py          # Step 2: Show label distribution
│   ├── convert_to_jsonl.py    # Step 3: Dedupe, split, convert to JSONL
│   └── finetune.py            # Modal fine-tuning script
├── output/                     # Training results (auto-generated)
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
python scripts/clean_data.py
python scripts/print_tags.py
python scripts/convert_to_jsonl.py

# Run training on Modal
modal run scripts/finetune.py --epochs 2
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

Place your training data in `data/train_all.csv` with two columns:

```csv
text,target
ZONE_TEMP_SP,zoneTempSp
discharge-temp,dischargeTemp
RAT,returnTemp
```

- **text**: The raw point name to classify
- **target**: The standardized label

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
