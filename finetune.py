"""
Fine-tune RoBERTa for HVAC Point Classification on Modal
=========================================================

A minimal proof-of-concept for fine-tuning FacebookAI/roberta-base
on Modal's serverless GPU infrastructure.

Usage:
    modal run finetune.py
    modal run finetune.py --epochs 3
"""

from datetime import datetime
from pathlib import Path

import modal

# Define the container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch",
        "transformers",
        "datasets",
        "scikit-learn",
        "accelerate",
    )
)

# Create a persistent volume for storing outputs
volume = modal.Volume.from_name("roberta-finetune-vol", create_if_missing=True)
volume_path = Path("/root") / "data"

# Create the Modal app
app = modal.App("roberta-finetune", image=image, volumes={volume_path: volume})

# Training configuration
TRAIN_GPU = "T4"  # Options: "T4", "A10G", "A100"
TRAIN_TIMEOUT = 30 * 60  # 30 minutes


@app.function(
    gpu=TRAIN_GPU,
    timeout=TRAIN_TIMEOUT,
)
def train(
    train_data: list[dict],
    val_data: list[dict],
    num_labels: int,
    epochs: int = 2,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
) -> dict:
    """
    Fine-tune RoBERTa on the provided data.
    
    Args:
        train_data: List of {"text": str, "label_id": int} dicts
        val_data: Validation data in same format
        num_labels: Number of classification labels
        epochs: Training epochs
        batch_size: Batch size
        learning_rate: Learning rate
    
    Returns:
        Dictionary with training results and metrics
    """
    import json
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
    )
    from datasets import Dataset
    from sklearn.metrics import accuracy_score, f1_score
    import numpy as np

    print("=" * 60)
    print("RoBERTa Fine-tuning on Modal")
    print("=" * 60)
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Train samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")
    print(f"Num labels: {num_labels}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print("=" * 60)

    # Convert to HuggingFace datasets
    train_dataset = Dataset.from_dict({
        "text": [d["text"] for d in train_data],
        "label": [d["label_id"] for d in train_data],
    })
    val_dataset = Dataset.from_dict({
        "text": [d["text"] for d in val_data],
        "label": [d["label_id"] for d in val_data],
    })

    # Load tokenizer and model
    model_name = "FacebookAI/roberta-base"
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )

    # Tokenize datasets
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=64,
        )

    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)

    # Define metrics
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return {
            "accuracy": accuracy_score(labels, predictions),
            "f1_weighted": f1_score(labels, predictions, average="weighted"),
        }

    # Set up training
    output_dir = volume_path / "training_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        logging_steps=10,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Train
    print("\nStarting training...")
    train_result = trainer.train()

    # Evaluate
    print("\nEvaluating...")
    eval_result = trainer.evaluate()

    # Prepare results
    results = {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_labels": num_labels,
            "train_samples": len(train_data),
            "val_samples": len(val_data),
        },
        "train_metrics": {
            "loss": train_result.metrics.get("train_loss"),
            "runtime_seconds": train_result.metrics.get("train_runtime"),
            "samples_per_second": train_result.metrics.get("train_samples_per_second"),
        },
        "eval_metrics": {
            "accuracy": eval_result.get("eval_accuracy"),
            "f1_weighted": eval_result.get("eval_f1_weighted"),
            "loss": eval_result.get("eval_loss"),
        },
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }

    # Save results to volume
    results_path = volume_path / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    volume.commit()  # Persist to volume

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Accuracy: {results['eval_metrics']['accuracy']:.4f}")
    print(f"F1 (weighted): {results['eval_metrics']['f1_weighted']:.4f}")
    print(f"Training time: {results['train_metrics']['runtime_seconds']:.2f}s")
    print("=" * 60)

    return results


@app.local_entrypoint()
def main(epochs: int = 2, batch_size: int = 8):
    """
    Run fine-tuning from the command line.
    
    Args:
        epochs: Number of training epochs
        batch_size: Training batch size
    """
    import json

    print("Loading training data...")
    
    # Load data from local files
    def load_jsonl(filepath: str) -> list:
        data = []
        with open(filepath, "r") as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data

    train_data = load_jsonl("data/train.jsonl")
    val_data = load_jsonl("data/validation.jsonl")

    # Get number of unique labels
    all_labels = set(d["label_id"] for d in train_data + val_data)
    num_labels = len(all_labels)

    print(f"Loaded {len(train_data)} train, {len(val_data)} val samples")
    print(f"Number of labels: {num_labels}")
    print(f"Starting Modal training with {epochs} epochs...")

    # Run training on Modal
    results = train.remote(
        train_data=train_data,
        val_data=val_data,
        num_labels=num_labels,
        epochs=epochs,
        batch_size=batch_size,
    )

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS FROM MODAL")
    print("=" * 60)
    print(json.dumps(results, indent=2))

    # Save results locally
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"output/{timestamp}_roberta-base.txt"
    
    with open(output_file, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("RoBERTa Fine-tuning Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Timestamp: {results['timestamp']}\n")
        f.write(f"Model: {results['model']}\n")
        f.write(f"GPU: {results['gpu']}\n\n")
        f.write("Configuration:\n")
        for k, v in results['config'].items():
            f.write(f"  {k}: {v}\n")
        f.write("\nTraining Metrics:\n")
        for k, v in results['train_metrics'].items():
            f.write(f"  {k}: {v}\n")
        f.write("\nEvaluation Metrics:\n")
        for k, v in results['eval_metrics'].items():
            if v is not None:
                f.write(f"  {k}: {v:.4f}\n")
        f.write("\n" + "=" * 60 + "\n")
        f.write("Raw JSON:\n")
        f.write(json.dumps(results, indent=2))

    print(f"\nResults saved to: {output_file}")
    return results
