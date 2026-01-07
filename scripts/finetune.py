"""
Fine-tune for HVAC Point Classification on Modal
=========================================================

A minimal proof-of-concept for fine-tuning a BERT model
on Modal's serverless GPU infrastructure.

Usage:
    modal run finetune.py
    modal run finetune.py --epochs 9 --batch-size 2 --learning-rate 1e-5
    modal run finetune.py --epochs 9 --push-to-hub --hf-repo username/model-name
"""

from datetime import datetime
from pathlib import Path
import os

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
        "huggingface_hub",
    )
)

# Create a persistent volume for storing outputs
volume = modal.Volume.from_name("autotagging-finetune-vol", create_if_missing=True)
volume_path = Path("/root") / "data"

# Create the Modal app
app = modal.App("autotagging-finetune", image=image, volumes={volume_path: volume})

# Training configuration
TRAIN_GPU = "T4"  # Options: "T4", "A10G", "A100"
TRAIN_TIMEOUT = 60 * 60  # 60 minutes (increased for more epochs)


@app.function(
    gpu=TRAIN_GPU,
    timeout=TRAIN_TIMEOUT,
)
def train(
    train_data: list[dict],
    val_data: list[dict],
    num_labels: int,
    label2id: dict,
    id2label: dict,
    epochs: int = 9,
    batch_size: int = 2,
    learning_rate: float = 1e-5,
    max_seq_length: int = 25,
    optimizer: str = "adamw_torch",
    scheduler: str = "linear",
    gradient_accumulation: int = 8,
    mixed_precision: str = "fp16",
    weight_decay: float = 0.075,
    push_to_hub: bool = False,
    hf_repo: str = None,
    hf_token: str = None,
) -> dict:
    """
    Fine-tune RoBERTa on the provided data.
    
    Args:
        train_data: List of {"text": str, "label_id": int} dicts
        val_data: Validation data in same format
        num_labels: Number of classification labels
        label2id: Dict mapping label names to IDs
        id2label: Dict mapping IDs to label names
        epochs: Training epochs
        batch_size: Batch size per device
        learning_rate: Learning rate
        max_seq_length: Maximum sequence length for tokenization
        optimizer: Optimizer type (adamw_torch, adamw_hf, sgd, adafactor)
        scheduler: LR scheduler type (linear, cosine, etc.)
        gradient_accumulation: Gradient accumulation steps
        mixed_precision: Mixed precision mode (fp16, bf16, no)
        weight_decay: Weight decay for regularization
        push_to_hub: Whether to push model to Hugging Face Hub
        hf_repo: Hugging Face repo ID (username/model-name)
        hf_token: Hugging Face API token
    
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
    print("-" * 60)
    print("Training Parameters:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Max seq length: {max_seq_length}")
    print(f"  Optimizer: {optimizer}")
    print(f"  Scheduler: {scheduler}")
    print(f"  Gradient accumulation: {gradient_accumulation}")
    print(f"  Mixed precision: {mixed_precision}")
    print(f"  Weight decay: {weight_decay}")
    print("-" * 60)
    print(f"Push to Hub: {push_to_hub}")
    if push_to_hub:
        print(f"HF Repo: {hf_repo}")
        print(f"HF Token: {'provided' if hf_token else 'NOT PROVIDED'}")
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
    model_name = "microsoft/deberta-v3-base"
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )

    # Convert id2label keys to integers for model config
    # (JSON keys are always strings, but model expects int keys)
    id2label_int = {int(k): v for k, v in id2label.items()}
    
    # Set label mappings on model config
    model.config.id2label = id2label_int
    model.config.label2id = label2id

    # Tokenize datasets
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
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

    # Determine mixed precision settings
    fp16 = mixed_precision == "fp16" and torch.cuda.is_available()
    bf16 = mixed_precision == "bf16" and torch.cuda.is_available()

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        gradient_accumulation_steps=gradient_accumulation,
        optim=optimizer,
        lr_scheduler_type=scheduler,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        logging_steps=10,
        fp16=fp16,
        bf16=bf16,
        report_to="none",
        warmup_ratio=0.1,  # 10% warmup
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

    # Run validation inference - test each sample individually
    print("\nRunning validation inference...")
    
    model.eval()
    predictions_list = []
    
    with torch.no_grad():
        for sample in val_data:
            # Tokenize single sample
            inputs = tokenizer(
                sample["text"],
                padding="max_length",
                truncation=True,
                max_length=max_seq_length,
                return_tensors="pt"
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Get prediction
            outputs = model(**inputs)
            pred_id = torch.argmax(outputs.logits, dim=1).item()
            confidence = torch.softmax(outputs.logits, dim=1).max().item()
            
            # Use id2label for predicted label name
            pred_label = id2label_int.get(pred_id, f"unknown_{pred_id}")
            actual_label = sample["label"]
            is_correct = pred_id == sample["label_id"]
            
            predictions_list.append({
                "text": sample["text"],
                "actual_label": actual_label,
                "actual_id": sample["label_id"],
                "predicted_label": pred_label,
                "predicted_id": pred_id,
                "confidence": confidence,
                "correct": is_correct,
            })
    
    # Calculate validation accuracy
    val_correct = sum(1 for p in predictions_list if p["correct"])
    val_total = len(predictions_list)
    val_accuracy = val_correct / val_total if val_total > 0 else 0

    # Push to Hugging Face Hub if requested
    hf_url = None
    if push_to_hub and hf_repo:
        print("\n" + "=" * 60)
        print("Pushing model to Hugging Face Hub...")
        print("=" * 60)
        
        if not hf_token:
            print("WARNING: HF_TOKEN not provided. Skipping push to Hub.")
        else:
            try:
                # Save model and tokenizer locally first
                final_model_path = volume_path / "final_model"
                final_model_path.mkdir(parents=True, exist_ok=True)
                
                trainer.save_model(str(final_model_path))
                tokenizer.save_pretrained(str(final_model_path))
                
                print(f"Model saved locally to: {final_model_path}")
                
                # Push to Hub using login for authentication
                from huggingface_hub import HfApi, login
                
                # Login first - this sets up authentication properly
                login(token=hf_token)
                print("Logged in to Hugging Face Hub")
                
                api = HfApi()
                
                # Create repo - be explicit about all parameters
                print(f"Creating/verifying repo: {hf_repo}")
                try:
                    repo_url = api.create_repo(
                        repo_id=hf_repo,
                        private=True,
                        exist_ok=True,
                        repo_type="model",
                    )
                    print(f"Repo ready: {repo_url}")
                except Exception as e:
                    print(f"Repo creation note: {e}")
                    # Continue anyway - repo might already exist
                
                # Upload folder
                print(f"Uploading model files...")
                api.upload_folder(
                    folder_path=str(final_model_path),
                    repo_id=hf_repo,
                    repo_type="model",
                    commit_message=f"Training: {epochs}ep, bs{batch_size}, lr{learning_rate}, F1:{eval_result.get('eval_f1_weighted', 0):.4f}",
                )
                
                hf_url = f"https://huggingface.co/{hf_repo}"
                print(f"Model pushed successfully to: {hf_url}")
                
            except Exception as e:
                print(f"ERROR pushing to Hub: {e}")
                import traceback
                traceback.print_exc()
    
    # Prepare results
    results = {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_seq_length": max_seq_length,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "gradient_accumulation": gradient_accumulation,
            "mixed_precision": mixed_precision,
            "weight_decay": weight_decay,
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
        "validation_inference": {
            "accuracy": val_accuracy,
            "correct": val_correct,
            "total": val_total,
            "predictions": predictions_list,
        },
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "huggingface_url": hf_url,
    }

    # Save results to volume
    results_path = volume_path / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    volume.commit()  # Persist to volume

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Training time: {results['train_metrics']['runtime_seconds']:.2f}s")
    print(f"Final F1 (weighted): {eval_result.get('eval_f1_weighted', 0):.4f}")
    print(f"Final Accuracy: {eval_result.get('eval_accuracy', 0):.4f}")
    print(f"Results saved to Modal volume")
    if hf_url:
        print(f"Model available at: {hf_url}")
    print("=" * 60)

    return results


@app.local_entrypoint()
def main(
    epochs: int = 9,
    batch_size: int = 2,
    learning_rate: float = 1e-5,
    max_seq_length: int = 25,
    optimizer: str = "adamw_torch",
    scheduler: str = "linear",
    gradient_accumulation: int = 8,
    mixed_precision: str = "fp16",
    weight_decay: float = 0.075,
    push_to_hub: bool = False,
    hf_repo: str = None,
):
    """
    Run fine-tuning from the command line.
    
    Args:
        epochs: Number of training epochs
        batch_size: Training batch size per device
        learning_rate: Learning rate
        max_seq_length: Maximum sequence length for tokenization
        optimizer: Optimizer type
        scheduler: Learning rate scheduler type
        gradient_accumulation: Gradient accumulation steps
        mixed_precision: Mixed precision mode (fp16, bf16, no)
        weight_decay: Weight decay for regularization
        push_to_hub: Whether to push model to Hugging Face Hub
        hf_repo: Hugging Face repo ID (username/model-name)
    """
    import json

    print("Loading training data...")
    
    # Get HF token from environment variable
    hf_token = os.environ.get("HF_TOKEN")
    if push_to_hub and not hf_token:
        print("WARNING: --push-to-hub specified but HF_TOKEN environment variable not set!")
    
    # Load data from local files (paths relative to repo root)
    def load_jsonl(filepath: str) -> list:
        data = []
        with open(filepath, "r") as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data

    train_data = load_jsonl("data/train.jsonl")
    val_data = load_jsonl("data/validation.jsonl")

    # Load label mapping (contains label2id, id2label, num_labels)
    with open("data/label_mapping.json", "r") as f:
        label_mapping = json.load(f)
    
    # Extract the nested structures
    label2id = label_mapping["label2id"]
    id2label = label_mapping["id2label"]
    num_labels = label_mapping["num_labels"]

    print(f"Loaded {len(train_data)} train, {len(val_data)} val samples")
    print(f"Number of labels: {num_labels}")
    print(f"Starting Modal training with {epochs} epochs...")
    print(f"Config: bs={batch_size}, lr={learning_rate}, seq_len={max_seq_length}")
    if push_to_hub:
        print(f"Will push to Hugging Face: {hf_repo}")

    # Run training on Modal
    results = train.remote(
        train_data=train_data,
        val_data=val_data,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_seq_length=max_seq_length,
        optimizer=optimizer,
        scheduler=scheduler,
        gradient_accumulation=gradient_accumulation,
        mixed_precision=mixed_precision,
        weight_decay=weight_decay,
        push_to_hub=push_to_hub,
        hf_repo=hf_repo,
        hf_token=hf_token,
    )

    # Save results locally
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"output/{timestamp}_debertav3-base.txt"
    
    with open(output_file, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("Fine-tuning Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Timestamp: {results['timestamp']}\n")
        f.write(f"Model: {results['model']}\n")
        f.write(f"GPU: {results['gpu']}\n")
        if results.get('huggingface_url'):
            f.write(f"Hugging Face: {results['huggingface_url']}\n")
        f.write("\nConfiguration:\n")
        for k, v in results['config'].items():
            f.write(f"  {k}: {v}\n")
        f.write("\nTraining Metrics:\n")
        for k, v in results['train_metrics'].items():
            f.write(f"  {k}: {v}\n")
        f.write("\nEvaluation Metrics:\n")
        for k, v in results['eval_metrics'].items():
            if v is not None:
                f.write(f"  {k}: {v:.4f}\n")
        
        # Add validation inference results
        f.write("\n" + "=" * 60 + "\n")
        f.write("Validation Inference Results\n")
        f.write("=" * 60 + "\n")
        val_inf = results.get('validation_inference', {})
        f.write(f"Accuracy: {val_inf.get('correct', 0)}/{val_inf.get('total', 0)} ({val_inf.get('accuracy', 0):.2%})\n\n")
        f.write("Predictions:\n")
        f.write("-" * 60 + "\n")
        for pred in val_inf.get('predictions', []):
            status = "✓" if pred['correct'] else "✗"
            f.write(f"{status} Input: '{pred['text']}'\n")
            f.write(f"   Predicted: {pred['predicted_label']} (confidence: {pred['confidence']:.2%})\n")
            f.write(f"   Actual:    {pred['actual_label']}\n\n")
        
        f.write("=" * 60 + "\n")
        f.write("Raw JSON:\n")
        f.write(json.dumps(results, indent=2, default=str))

    print(f"\nResults saved to: {output_file}")
    return results