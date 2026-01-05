# RoBERTa Fine-tuning POC

A minimal proof-of-concept for fine-tuning `FacebookAI/roberta-base` using Modal and GitHub Actions.

## What This Does

1. **GitHub Actions** triggers training (manually or on push to `data/` or `scripts`)
2. **Modal** runs the fine-tuning job on a GPU (T4)
3. **Results** are committed back to the repo with a timestamp
4. **Pull Request** is automatically created for review

## Project Structure

```
roberta-poc/
├── .github/workflows/
│   └── train.yml          # CI/CD pipeline
├── data/
│   ├── train.jsonl        # Training data (20 samples)
│   └── validation.jsonl   # Validation data (5 samples)
├── output/                 # Training results (auto-generated)
├── finetune.py            # Modal fine-tuning script
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

Push changes to the `data/` directory:

```bash
# Edit data/train.jsonl
git add data/
git commit -m "Update training data"
git push
```

### Local Testing

```bash
pip install modal
modal run finetune.py --epochs 2
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

## Data Format

JSONL format with three fields:

```json
{"text": "ZONE_TEMP_SP", "label": "zoneTempSp", "label_id": 0}
{"text": "DISCHARGE_TEMP", "label": "dischargeTemp", "label_id": 1}
```

## Workflow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Trigger   │────▶│    Modal    │────▶│   Commit    │────▶│  Create PR  │
│  (Manual/   │     │  Training   │     │   Results   │     │             │
│   Push)     │     │   (GPU)     │     │             │     │             │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

## Notes

- This is a **proof-of-concept** with minimal training data
- The model won't be production-ready with only 20 samples
- Increase `data/train.jsonl` for real use cases
- Modal's free tier includes GPU credits for testing
