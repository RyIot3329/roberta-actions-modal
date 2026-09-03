"""
Fine-tune DeBERTa-v3 / RoBERTa for HVAC Point Classification on Modal
=====================================================================

Fine-tuning transformer models on Modal's serverless GPU infrastructure.

Usage:
    modal run finetune.py
    modal run finetune.py --model microsoft/deberta-v3-base
    modal run finetune.py --model FacebookAI/roberta-base
    modal run finetune.py --gpu A10G --epochs 9 --push-to-hub --hf-repo username/model-name
"""

import json
from datetime import datetime
from pathlib import Path
import os

import numpy as np

import modal

# Define the container image with all dependencies
# Versions are pinned so runs are reproducible and comparable
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "datasets==3.1.0",
        "scikit-learn==1.5.2",
        "accelerate==1.1.1",
        "huggingface_hub==0.26.2",
        "sentencepiece==0.2.0",  # Required for DeBERTa tokenizer
    )
)

# Create a persistent volume for storing outputs
volume = modal.Volume.from_name("deberta-finetune-vol", create_if_missing=True)
volume_path = Path("/root") / "data"

# Create the Modal app
app = modal.App("deberta-finetune", image=image, volumes={volume_path: volume})

# Training configuration
TRAIN_TIMEOUT = 180 * 60  # 180 minutes

# Available models
AVAILABLE_MODELS = {
    "deberta-v3-base": "microsoft/deberta-v3-base",
    "deberta-v3-small": "microsoft/deberta-v3-small",
    "deberta-v3-large": "microsoft/deberta-v3-large",
    "roberta-base": "FacebookAI/roberta-base",
    "roberta-large": "FacebookAI/roberta-large",
    # DAPT checkpoint produced by scripts/dapt.py (private repo)
    "deberta-v3-bms": "RyIoT33/deberta-v3-bms-base",
}


def _train_impl(
    train_data: list[dict],
    val_data: list[dict],
    test_data: list[dict],
    num_labels: int,
    label2id: dict,
    id2label: dict,
    model_name: str = "microsoft/deberta-v3-base",
    epochs: int = 30,
    batch_size: int = 128,
    learning_rate: float = 4e-5,
    max_seq_length: int = 32,
    optimizer: str = "adamw_torch",
    scheduler: str = "cosine",
    gradient_accumulation: int = 1,
    mixed_precision: str = "bf16",
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    label_smoothing: float = 0.1,
    metric_for_best_model: str = "f1_weighted",
    seed: int = 42,
    push_to_hub: bool = False,
    hf_repo: str = None,
    hf_token: str = None,
    baseline_f1: float = None,
    save_name: str = "final_model",
    target_precision: float = 0.85,
) -> dict:
    """
    Fine-tune a transformer model on the provided data.

    This is the shared implementation called by GPU-specific wrapper functions.
    Defaults mirror config/training.yml, which is the single source of truth.
    """
    import json
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
        EarlyStoppingCallback,
        DataCollatorWithPadding,
        set_seed,
    )
    from datasets import Dataset
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    import numpy as np

    set_seed(seed)

    # TF32 is free speedup on Ampere+ (A100/L4/H100) with no accuracy cost
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # bf16 is preferred on Ampere+ but unsupported on T4: fall back to fp16
    if mixed_precision == "bf16" and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        print("WARNING: bf16 not supported on this GPU, falling back to fp16")
        mixed_precision = "fp16"

    print("=" * 60)
    print("Transformer Fine-tuning on Modal")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Train samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")
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
    print(f"  Effective batch size: {batch_size * gradient_accumulation}")
    print(f"  Mixed precision: {mixed_precision}")
    print(f"  Weight decay: {weight_decay}")
    print(f"  Warmup ratio: {warmup_ratio}")
    print(f"  Label smoothing: {label_smoothing}")
    print(f"  Model selection metric: {metric_for_best_model}")
    print(f"  Seed: {seed}")
    print("-" * 60)
    print(f"Push to Hub: {push_to_hub}")
    if push_to_hub and baseline_f1 is not None:
        print(f"Quality gate: push only if test f1_weighted >= {baseline_f1:.4f}")
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
    test_dataset = Dataset.from_dict({
        "text": [d["text"] for d in test_data],
        "label": [d["label_id"] for d in test_data],
    })

    # Load tokenizer and model
    print(f"\nLoading model: {model_name}")
    
    assert len(label2id) == num_labels == len(id2label), (
        f"Label space mismatch: label2id has {len(label2id)}, "
        f"num_labels is {num_labels}, id2label has {len(id2label)}")

    # token enables private base checkpoints (e.g. the DAPT-pretrained
    # encoder); None is fine for public models
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        token=hf_token,
    )

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Convert id2label keys to integers for model config
    # (JSON keys are always strings, but model expects int keys)
    id2label_int = {int(k): v for k, v in id2label.items()}
    
    # Set label mappings on model config
    model.config.id2label = id2label_int
    model.config.label2id = label2id

    # Tokenize datasets (no padding here; the collator pads per batch)
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
        )

    print("\nTokenizing datasets...")
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    # Warn if any sample loses tokens to truncation
    n_truncated = sum(
        1 for ds in (train_dataset, val_dataset, test_dataset)
        for ids in ds["input_ids"] if len(ids) >= max_seq_length
    )
    if n_truncated > 0:
        print(f"WARNING: {n_truncated} samples hit max_seq_length={max_seq_length} and may be truncated")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Define metrics
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        f1_weighted = f1_score(labels, predictions, average="weighted", zero_division=0)
        f1_macro = f1_score(labels, predictions, average="macro", zero_division=0)
        f1_micro = f1_score(labels, predictions, average="micro", zero_division=0)
        
        return {
            "accuracy": accuracy,
            "f1_weighted": f1_weighted,
            "f1_macro": f1_macro,
            "f1_micro": f1_micro,
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
        per_device_eval_batch_size=max(batch_size, 128),
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        gradient_accumulation_steps=gradient_accumulation,
        optim=optimizer,
        lr_scheduler_type=scheduler,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=True,
        seed=seed,
        logging_steps=10,
        fp16=fp16,
        bf16=bf16,
        label_smoothing_factor=label_smoothing,
        report_to="none",
        warmup_ratio=warmup_ratio,
        save_total_limit=2,  # Keep only best 2 checkpoints
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Train
    print("\nStarting training...")
    train_result = trainer.train()

    # Evaluate
    print("\nEvaluating...")
    eval_result = trainer.evaluate()

    # Post-hoc temperature scaling: a single scalar T fitted on validation
    # (the only split we may tune on) so reported confidences are calibrated.
    # Stored in the model config below so downstream consumers apply it too.
    def fit_temperature(logits, labels):
        log_t = torch.zeros(1, requires_grad=True)
        lbfgs = torch.optim.LBFGS([log_t], lr=0.1, max_iter=50)

        def closure():
            lbfgs.zero_grad()
            loss = torch.nn.functional.cross_entropy(logits / torch.exp(log_t), labels)
            loss.backward()
            return loss

        lbfgs.step(closure)
        # Clamp away the degenerate optimum (T->0 when val is near-perfectly
        # classified) so stored confidences never saturate to exactly 1.0
        return float(min(max(torch.exp(log_t).item(), 0.25), 10.0))

    def build_predictions(output, samples, temperature):
        logits = torch.from_numpy(output.predictions).float()
        probs = torch.softmax(logits / temperature, dim=1)
        confidences, pred_ids = probs.max(dim=1)
        topk_ids = probs.topk(min(10, probs.shape[1]), dim=1).indices.tolist()

        predictions = []
        for sample, pred_id, confidence, topk in zip(
                samples, pred_ids.tolist(), confidences.tolist(), topk_ids):
            record = {
                "text": sample["text"],
                "actual_label": sample["label"],
                "actual_id": sample["label_id"],
                "predicted_label": id2label_int.get(pred_id, f"unknown_{pred_id}"),
                "predicted_id": pred_id,
                "confidence": confidence,
                "correct": pred_id == sample["label_id"],
                "topk_labels": [id2label_int.get(i, f"unknown_{i}") for i in topk],
            }
            # Slice fields for the composite gate (site / rows / seen_in_train /
            # lenient accept set) travel with each prediction
            for key in ("site", "rows", "seen_in_train", "accept"):
                if key in sample:
                    record[key] = sample[key]
            predictions.append(record)
        return predictions

    print("\nRunning validation inference...")
    val_output = trainer.predict(val_dataset)
    temperature = fit_temperature(
        torch.from_numpy(val_output.predictions).float(),
        torch.tensor([d["label_id"] for d in val_data]),
    )
    print(f"Calibration temperature (fitted on validation): {temperature:.3f}")
    model.config.calibration_temperature = temperature

    predictions_list = build_predictions(val_output, val_data, temperature)

    # Acceptance threshold: smallest calibrated confidence at which validation
    # precision reaches `target_precision` (same rule as metrics_core.
    # fit_acceptance_threshold; kept inline because the container does not
    # ship the local scripts). Stored in the config so the serving layer can
    # map lower-confidence predictions to predicted_unknown.
    def fit_acceptance_threshold(preds, target):
        order = sorted(preds, key=lambda p: -p["confidence"])
        hits = 0
        best = 1.0
        for i, p in enumerate(order, start=1):
            hits += 1 if p["correct"] else 0
            if hits / i >= target:
                best = float(p["confidence"])
        return best

    acceptance_threshold = fit_acceptance_threshold(predictions_list, target_precision)
    model.config.acceptance_threshold = acceptance_threshold
    model.config.acceptance_target_precision = target_precision
    print(f"Acceptance threshold for {target_precision:.0%} validation precision: "
          f"{acceptance_threshold:.3f}")
    val_correct = sum(1 for p in predictions_list if p["correct"])
    val_total = len(predictions_list)
    val_accuracy = val_correct / val_total if val_total > 0 else 0

    # Test set was never seen during training or model selection, so these
    # are the headline metrics
    print("\nRunning test inference...")
    test_output = trainer.predict(test_dataset)
    test_metrics = test_output.metrics
    test_predictions = build_predictions(test_output, test_data, temperature)
    test_correct = sum(1 for p in test_predictions if p["correct"])
    test_total = len(test_predictions)
    test_accuracy = test_correct / test_total if test_total > 0 else 0

    all_labels = [p["actual_id"] for p in test_predictions]
    all_preds = [p["predicted_id"] for p in test_predictions]
    report_label_ids = sorted(set(all_labels + all_preds))
    report_args = dict(
        labels=report_label_ids,
        target_names=[id2label_int[i] for i in report_label_ids],
        zero_division=0,
    )
    print("\nTest Set Classification Report:")
    print(classification_report(all_labels, all_preds, **report_args))
    # Persisted per-class metrics: per-run regressions stay visible in the
    # results PR instead of vanishing with the Modal logs
    test_per_class = classification_report(all_labels, all_preds, output_dict=True, **report_args)

    # Quality gate: never overwrite the production model with a worse one.
    # The baseline comes from output/best_metrics.json in the repo (written
    # only after a successful gated push), so regressions cannot ship.
    gate = None
    if push_to_hub and hf_repo and baseline_f1 is not None:
        new_f1 = test_metrics.get("test_f1_weighted") or 0.0
        if new_f1 < baseline_f1:
            print("\n" + "=" * 60)
            print(f"QUALITY GATE FAILED: test f1_weighted {new_f1:.4f} < "
                  f"previous best {baseline_f1:.4f} -- skipping Hub push")
            print("=" * 60)
            push_to_hub = False
            gate = {"passed": False, "baseline_f1": baseline_f1, "new_f1": new_f1}
        else:
            gate = {"passed": True, "baseline_f1": baseline_f1, "new_f1": new_f1}

    # Always persist the final model (best checkpoint + calibrated config) to
    # the volume: the local composite gate in main() decides whether
    # push_saved_model() ships it, so nothing is lost when the gate runs later
    final_model_path = volume_path / save_name
    final_model_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_model_path))
    tokenizer.save_pretrained(str(final_model_path))
    volume.commit()
    print(f"Model saved to volume: {final_model_path}")

    # Legacy in-container push (main() now gates locally and pushes via
    # push_saved_model; this path only runs when explicitly requested)
    hf_url = None
    if push_to_hub and hf_repo:
        print("\n" + "=" * 60)
        print("Pushing model to Hugging Face Hub...")
        print("=" * 60)
        
        if not hf_token:
            print("WARNING: HF_TOKEN not provided. Skipping push to Hub.")
        else:
            try:
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
                model_short_name = model_name.split("/")[-1]
                api.upload_folder(
                    folder_path=str(final_model_path),
                    repo_id=hf_repo,
                    repo_type="model",
                    commit_message=f"{model_short_name}: {epochs}ep, bs{batch_size}x{gradient_accumulation}, lr{learning_rate}, testF1:{test_metrics.get('test_f1_weighted', 0):.4f}",
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
        "model_params": {
            "total": total_params,
            "trainable": trainable_params,
        },
        "config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "effective_batch_size": batch_size * gradient_accumulation,
            "learning_rate": learning_rate,
            "max_seq_length": max_seq_length,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "gradient_accumulation": gradient_accumulation,
            "mixed_precision": mixed_precision,
            "weight_decay": weight_decay,
            "warmup_ratio": warmup_ratio,
            "label_smoothing": label_smoothing,
            "metric_for_best_model": metric_for_best_model,
            "seed": seed,
            "num_labels": num_labels,
            "train_samples": len(train_data),
            "val_samples": len(val_data),
            "test_samples": len(test_data),
        },
        "train_metrics": {
            "loss": train_result.metrics.get("train_loss"),
            "runtime_seconds": train_result.metrics.get("train_runtime"),
            "samples_per_second": train_result.metrics.get("train_samples_per_second"),
            "epochs_completed": train_result.metrics.get("epoch"),
        },
        "eval_metrics": {
            "accuracy": eval_result.get("eval_accuracy"),
            "f1_weighted": eval_result.get("eval_f1_weighted"),
            "f1_macro": eval_result.get("eval_f1_macro"),
            "f1_micro": eval_result.get("eval_f1_micro"),
            "loss": eval_result.get("eval_loss"),
        },
        "test_metrics": {
            "accuracy": test_metrics.get("test_accuracy"),
            "f1_weighted": test_metrics.get("test_f1_weighted"),
            "f1_macro": test_metrics.get("test_f1_macro"),
            "f1_micro": test_metrics.get("test_f1_micro"),
            "loss": test_metrics.get("test_loss"),
        },
        "validation_inference": {
            "accuracy": val_accuracy,
            "correct": val_correct,
            "total": val_total,
            "predictions": predictions_list,
        },
        "test_inference": {
            "accuracy": test_accuracy,
            "correct": test_correct,
            "total": test_total,
            "predictions": test_predictions,
        },
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "huggingface_url": hf_url,
        "quality_gate": gate,
        "calibration_temperature": temperature,
        "acceptance_threshold": acceptance_threshold,
        "target_precision": target_precision,
        "test_per_class": test_per_class,
        "save_name": save_name,
    }
    # Raw logits (not written to results.json) let the local driver apply the
    # stored temperature, build multi-seed ensembles, and fit the acceptance
    # threshold without another GPU pass
    logits_payload = {
        "val_logits": val_output.predictions.astype(np.float16),
        "test_logits": test_output.predictions.astype(np.float16),
    }

    # Save results to volume
    results_path = volume_path / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    volume.commit()  # Persist to volume

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Training time: {results['train_metrics']['runtime_seconds']:.2f}s")
    print(f"Epochs completed: {results['train_metrics']['epochs_completed']}")
    print(f"Validation F1 (weighted): {eval_result.get('eval_f1_weighted', 0):.4f}")
    print(f"Validation Accuracy: {eval_result.get('eval_accuracy', 0):.4f}")
    print(f"Test F1 (weighted): {test_metrics.get('test_f1_weighted', 0):.4f}")
    print(f"Test F1 (macro): {test_metrics.get('test_f1_macro', 0):.4f}")
    print(f"Test Accuracy: {test_metrics.get('test_accuracy', 0):.4f}")
    print(f"Results saved to Modal volume")
    if hf_url:
        print(f"Model available at: {hf_url}")
    print("=" * 60)

    results.update(logits_payload)
    return results


# =============================================================================
# GPU-specific wrapper functions
# Each function runs on a different GPU type but calls the same implementation
# =============================================================================

@app.function(gpu="T4", timeout=TRAIN_TIMEOUT)
def train_t4(**kwargs) -> dict:
    """Training function for T4 GPU (budget option)."""
    return _train_impl(**kwargs)


@app.function(gpu="L4", timeout=TRAIN_TIMEOUT)
def train_l4(**kwargs) -> dict:
    """Training function for L4 GPU (balanced price/performance)."""
    return _train_impl(**kwargs)


@app.function(gpu="A10G", timeout=TRAIN_TIMEOUT)
def train_a10g(**kwargs) -> dict:
    """Training function for A10G GPU (good for medium models)."""
    return _train_impl(**kwargs)


@app.function(gpu="A100", timeout=TRAIN_TIMEOUT)
def train_a100(**kwargs) -> dict:
    """Training function for A100-40GB GPU (large models, fast training)."""
    return _train_impl(**kwargs)


@app.function(gpu="H100", timeout=TRAIN_TIMEOUT)
def train_h100(**kwargs) -> dict:
    """Training function for H100 GPU (maximum performance)."""
    return _train_impl(**kwargs)


@app.function(timeout=45 * 60)
def push_saved_model(save_name: str, hf_repo: str, hf_token: str, commit_message: str) -> str:
    """Upload a model dir saved on the volume by _train_impl to the Hub.

    Called by main() only after the local composite gate passed, so the
    production repo is never overwritten by a regression."""
    from huggingface_hub import HfApi, login

    volume.reload()
    model_dir = volume_path / save_name
    if not model_dir.exists():
        raise FileNotFoundError(f"{model_dir} not found on the volume")
    login(token=hf_token)
    api = HfApi()
    try:
        api.create_repo(repo_id=hf_repo, private=True, exist_ok=True, repo_type="model")
    except Exception as e:  # noqa: BLE001
        print(f"Repo creation note: {e}")
    api.upload_folder(folder_path=str(model_dir), repo_id=hf_repo, repo_type="model",
                      commit_message=commit_message)
    url = f"https://huggingface.co/{hf_repo}"
    print(f"Model pushed to {url}")
    return url


# Mapping from GPU name to training function
GPU_FUNCTIONS = {
    "T4": train_t4,
    "L4": train_l4,
    "A10G": train_a10g,
    "A100": train_a100,
    "A100-40GB": train_a100,
    "A100-80GB": train_a100,  # Modal will handle the specific variant
    "H100": train_h100,
}




# =============================================================================
# Local driver: multi-seed training, composite gate, gated push
# =============================================================================

def _load_jsonl(filepath: str) -> list:
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _git_sha():
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def _ensemble_predictions(seed_results, samples, split_key, id2label_int):
    """Average the temperature-scaled probabilities of every seed for one
    split and turn them into prediction records (same schema as
    build_predictions)."""
    probs = None
    for res in seed_results.values():
        logits = np.asarray(res[f"{split_key}_logits"], dtype=np.float32)
        t = float(res["calibration_temperature"] or 1.0)
        z = logits / t
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        probs = p if probs is None else probs + p
    probs /= len(seed_results)
    order = np.argsort(-probs, axis=1)[:, :10]
    preds = []
    for sample, ranked in zip(samples, order):
        pred_id = int(ranked[0])
        record = {
            "text": sample["text"],
            "actual_label": sample["label"],
            "actual_id": sample["label_id"],
            "predicted_label": id2label_int.get(pred_id, f"unknown_{pred_id}"),
            "predicted_id": pred_id,
            "confidence": float(probs[len(preds), pred_id]),
            "correct": pred_id == sample["label_id"],
            "topk_labels": [id2label_int.get(int(i), f"unknown_{i}") for i in ranked],
        }
        for key in ("site", "rows", "seen_in_train", "accept"):
            if key in sample:
                record[key] = sample[key]
        preds.append(record)
    return preds


def _seed_agreement(seed_results):
    seeds = sorted(seed_results)
    if len(seeds) < 2:
        return None
    preds = {s: [p["predicted_id"] for p in seed_results[s]["test_inference"]["predictions"]]
             for s in seeds}
    n = len(preds[seeds[0]])
    pair = []
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            pair.append(sum(a == b for a, b in zip(preds[seeds[i]], preds[seeds[j]])) / n)
    unanimous = sum(len({preds[s][k] for s in seeds}) == 1 for k in range(n)) / n
    return {"pairwise_mean": float(np.mean(pair)), "unanimous": float(unanimous)}


@app.local_entrypoint()
def main(
    model: str = "microsoft/deberta-v3-base",
    gpu: str = "T4",
    epochs: int = 30,
    batch_size: int = 128,
    learning_rate: float = 4e-5,
    max_seq_length: int = 32,
    optimizer: str = "adamw_torch",
    scheduler: str = "cosine",
    gradient_accumulation: int = 1,
    mixed_precision: str = "bf16",
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    label_smoothing: float = 0.1,
    metric_for_best_model: str = "f1_weighted",
    seed: int = 42,
    seeds: str = "",
    push_to_hub: bool = False,
    hf_repo: str = None,
    output_dir: str = "output",
    baseline_path: str = "output/best_metrics.json",
    target_precision: float = 0.85,
):
    """
    Run fine-tuning from the command line.

    Defaults mirror config/training.yml (the single source of truth); the
    GitHub workflow passes every value explicitly from that file.

    Multi-seed: `--seeds 42,43,44` trains every seed in parallel on Modal,
    scores each with the shared metrics_core scorer (strict/lenient accuracy,
    log1p(rows) weighting, per-site and seen/unseen slices, coverage at the
    validation-fitted acceptance threshold), reports the logit ensemble, and
    selects the seed with the median VALIDATION strict accuracy as the
    candidate. The composite non-inferiority gate compares the candidate
    against output/best_metrics.json (fingerprinted; run
    scripts/rescore_baseline.py when it is stale). Only a passing candidate is
    pushed (push_saved_model) and only then is the baseline rewritten.

    Args:
        model: Model to fine-tune (e.g., microsoft/deberta-v3-base, FacebookAI/roberta-base)
        gpu: GPU type for Modal (T4, L4, A10G, A100, H100)
        epochs: Number of training epochs
        batch_size: Training batch size per device
        learning_rate: Learning rate
        max_seq_length: Maximum sequence length for tokenization
        optimizer: Optimizer type
        scheduler: Learning rate scheduler type
        gradient_accumulation: Gradient accumulation steps
        mixed_precision: Mixed precision mode (fp16, bf16, no)
        weight_decay: Weight decay for regularization
        warmup_ratio: Ratio of total steps for warmup
        label_smoothing: Label smoothing factor (also softens confidences)
        metric_for_best_model: Validation metric for checkpoint selection
        seed: Random seed (used when --seeds is empty)
        seeds: Comma-separated seeds for a multi-seed run (overrides --seed)
        push_to_hub: Push the gated candidate to the Hub
        hf_repo: Hugging Face repo ID (username/model-name)
        output_dir: Where results files are written
        baseline_path: Quality-gate baseline (schema v2, see metrics_core)
        target_precision: Validation precision the acceptance threshold targets
    """
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import metrics_core as mc

    print("=" * 60)
    print("HVAC Point Classification - Fine-tuning Pipeline")
    print("=" * 60)

    model_name = AVAILABLE_MODELS.get(model, model)
    if model in AVAILABLE_MODELS:
        print(f"Using model shorthand: {model} -> {model_name}")
    gpu_upper = gpu.upper()
    if gpu_upper not in GPU_FUNCTIONS:
        print(f"WARNING: Unknown GPU '{gpu}', falling back to T4")
        gpu_upper = "T4"
    seed_list = [int(x) for x in seeds.split(",") if x.strip()] or [seed]
    print(f"Model: {model_name}\nGPU: {gpu_upper}\nSeeds: {seed_list}")
    print("=" * 60)

    hf_token = os.environ.get("HF_TOKEN")
    if push_to_hub and not hf_token:
        print("WARNING: --push-to-hub specified but HF_TOKEN environment variable not set!")

    train_data = _load_jsonl("data/train.jsonl")
    val_data = _load_jsonl("data/validation.jsonl")
    test_data = _load_jsonl("data/test.jsonl")
    with open("data/label_mapping.json", "r") as f:
        label_mapping = json.load(f)
    label2id = label_mapping["label2id"]
    id2label = label_mapping["id2label"]
    id2label_int = {int(k): v for k, v in id2label.items()}
    num_labels = label_mapping["num_labels"]
    print(f"Loaded {len(train_data)} train, {len(val_data)} val, {len(test_data)} test samples")
    print(f"Number of labels: {num_labels}")

    # ----- Scoreboard + baseline -----
    fingerprints = mc.fingerprints(test_data, list(label2id))
    baseline = None
    baseline_preds = None
    baseline_note = "no baseline file"
    if os.path.exists(baseline_path):
        baseline = mc.legacy_baseline(baseline_path)
        if baseline.get("schema_version", 1) < 2 or not baseline.get("fingerprints"):
            baseline_note = "baseline is schema v1 (no fingerprints): STALE -- run scripts/rescore_baseline.py"
            baseline = None
        elif not mc.fingerprints_match(fingerprints, baseline["fingerprints"]):
            baseline_note = ("baseline fingerprints do not match the current test set / label space: "
                             "STALE -- run scripts/rescore_baseline.py")
        else:
            baseline_note = (f"baseline {baseline.get('model')} strict "
                             f"{baseline['metrics']['strict']['accuracy']:.4f} on the same scoreboard")
            pred_path = baseline.get("predictions_path")
            if pred_path and os.path.exists(pred_path):
                baseline_preds = mc.load_predictions_jsonl(pred_path)
    print(f"Quality gate: {baseline_note}")

    # ----- Train every seed in parallel -----
    train_fn = GPU_FUNCTIONS[gpu_upper]
    common = dict(
        train_data=train_data, val_data=val_data, test_data=test_data,
        num_labels=num_labels, label2id=label2id, id2label=id2label,
        model_name=model_name, epochs=epochs, batch_size=batch_size,
        learning_rate=learning_rate, max_seq_length=max_seq_length,
        optimizer=optimizer, scheduler=scheduler,
        gradient_accumulation=gradient_accumulation, mixed_precision=mixed_precision,
        weight_decay=weight_decay, warmup_ratio=warmup_ratio,
        label_smoothing=label_smoothing, metric_for_best_model=metric_for_best_model,
        push_to_hub=False, hf_repo=hf_repo, hf_token=hf_token, baseline_f1=None,
        target_precision=target_precision,
    )
    print(f"\nStarting Modal training on {gpu_upper} for seeds {seed_list}...")
    calls = {s: train_fn.spawn(seed=s, save_name=f"final_model_seed{s}", **common)
             for s in seed_list}
    seed_results = {s: call.get() for s, call in calls.items()}

    # ----- Score every seed locally -----
    per_seed = {}
    for s, res in seed_results.items():
        val_preds = res["validation_inference"]["predictions"]
        test_preds = res["test_inference"]["predictions"]
        tau = mc.fit_acceptance_threshold(val_preds, target_precision)
        per_seed[s] = {
            "tau": tau,
            "val": mc.score_predictions(val_preds, tau=tau),
            "test": mc.score_predictions(test_preds, tau=tau),
            "test_preds": test_preds,
        }
    ensemble = None
    if len(seed_results) > 1:
        ens_val = _ensemble_predictions(seed_results, val_data, "val", id2label_int)
        ens_test = _ensemble_predictions(seed_results, test_data, "test", id2label_int)
        tau_e = mc.fit_acceptance_threshold(ens_val, target_precision)
        ensemble = {"tau": tau_e, "val": mc.score_predictions(ens_val, tau=tau_e),
                    "test": mc.score_predictions(ens_test, tau=tau_e), "test_preds": ens_test}
    agreement = _seed_agreement(seed_results)

    # Candidate = seed with the median validation strict accuracy (never
    # selected on the test set)
    ranked = sorted(seed_list, key=lambda s: per_seed[s]["val"]["strict"]["accuracy"])
    selected = ranked[len(ranked) // 2]
    results = seed_results[selected]
    cand_test = per_seed[selected]["test"]
    cand_preds = per_seed[selected]["test_preds"]
    print(f"\nSelected seed {selected} (median validation strict accuracy "
          f"{per_seed[selected]['val']['strict']['accuracy']:.4f})")

    candidate_record = mc.build_metrics_record(
        cand_test, fingerprints, model=model_name, hf_repo=hf_repo, git_sha=_git_sha(),
        timestamp=results["timestamp"], seeds=seed_list, selected_seed=selected,
        extra={
            "calibration_temperature": results.get("calibration_temperature"),
            "acceptance_threshold": per_seed[selected]["tau"],
            "target_precision": target_precision,
            "validation_metrics": {k: v for k, v in per_seed[selected]["val"].items()
                                   if k != "coverage_curve"},
            "seed_summary": {str(s): {"val_strict": per_seed[s]["val"]["strict"]["accuracy"],
                                      "test_strict": per_seed[s]["test"]["strict"]["accuracy"],
                                      "test_f1_weighted": per_seed[s]["test"]["strict"]["f1_weighted"],
                                      "test_lenient": per_seed[s]["test"]["lenient"]["accuracy"],
                                      "tau": per_seed[s]["tau"]} for s in seed_list},
            "seed_agreement": agreement,
            "ensemble": None if ensemble is None else {
                "test_strict": ensemble["test"]["strict"]["accuracy"],
                "test_lenient": ensemble["test"]["lenient"]["accuracy"],
                "val_strict": ensemble["val"]["strict"]["accuracy"],
                "test_log1p_rows": ensemble["test"]["strict"]["log1p_rows_accuracy"]},
        },
    )
    if baseline is not None:
        decision = mc.promote_decision(candidate_record, baseline, cand_preds, baseline_preds)
    else:
        decision = {"passed": False, "reason": baseline_note, "axes": []}
    print("\n" + mc.format_axes_table(decision))

    # Linear floor (written by scripts/baseline_linear.py in CI)
    floor = None
    floor_path = os.path.join(output_dir, "baseline_linear_metrics.json")
    if os.path.exists(floor_path):
        with open(floor_path) as f:
            floor_rec = json.load(f)
        if mc.fingerprints_match(floor_rec.get("fingerprints"), fingerprints):
            floor = floor_rec["metrics"]["strict"]["accuracy"]

    # ----- Gated push + baseline rewrite -----
    hf_url = None
    if decision["passed"] and push_to_hub and hf_repo and hf_token:
        msg = (f"{model_name.split('/')[-1]}: seed {selected} of {seed_list}, {epochs}ep, "
               f"bs{batch_size}x{gradient_accumulation}, lr{learning_rate}, "
               f"test strict {cand_test['strict']['accuracy']:.4f}")
        hf_url = push_saved_model.remote(results["save_name"], hf_repo, hf_token, msg)
        os.makedirs(os.path.dirname(baseline_path) or ".", exist_ok=True)
        pred_path = os.path.join(os.path.dirname(baseline_path) or ".", "best_predictions.jsonl")
        candidate_record["predictions_path"] = pred_path
        candidate_record["huggingface_url"] = hf_url
        with open(baseline_path, "w") as f:
            json.dump(candidate_record, f, indent=2)
        mc.write_predictions_jsonl(cand_preds, pred_path)
        print(f"Updated quality-gate baseline: {baseline_path}")
    elif decision["passed"] and push_to_hub:
        print("Gate passed but no HF token/repo: nothing pushed, baseline unchanged")
    results["huggingface_url"] = hf_url
    results["quality_gate"] = {"passed": decision["passed"], "reason": decision["reason"],
                               "primary": mc.PRIMARY_METRIC}

    # ----- Results files -----
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short_name = model_name.split("/")[-1]
    stem = f"{timestamp}_{model_short_name}"
    metrics_file = os.path.join(output_dir, f"{stem}_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump({"candidate": candidate_record, "gate": decision,
                   "baseline_note": baseline_note, "linear_floor_strict": floor,
                   "per_seed": {str(s): {"tau": per_seed[s]["tau"],
                                         "validation": {k: v for k, v in per_seed[s]["val"].items()
                                                        if k != "coverage_curve"},
                                         "test": {k: v for k, v in per_seed[s]["test"].items()
                                                  if k != "coverage_curve"}} for s in seed_list},
                   "ensemble": None if ensemble is None else {
                       "tau": ensemble["tau"],
                       "validation": {k: v for k, v in ensemble["val"].items() if k != "coverage_curve"},
                       "test": {k: v for k, v in ensemble["test"].items() if k != "coverage_curve"}},
                   "coverage_curve_test": cand_test.get("coverage_curve")},
                  f, indent=2, default=str)
    mc.write_predictions_jsonl(cand_preds, os.path.join(output_dir, f"{stem}_predictions.jsonl"))

    def fmt_block(name, m):
        s_, l_ = m["strict"], m["lenient"]
        cov = m.get("coverage", {})
        line = (f"  {name}: strict {s_['accuracy']:.4f} (f1w {s_['f1_weighted']:.4f}, "
                f"f1m {s_['f1_macro']:.4f}) lenient {l_['accuracy']:.4f} "
                f"log1p-rows {s_['log1p_rows_accuracy']:.4f} rows {s_['rows_accuracy']:.4f}")
        if cov:
            line += (f" | coverage@tau={cov['tau']:.3f}: {cov['coverage_texts']:.1%} texts "
                     f"at precision {cov['precision_at_tau']:.1%}")
        if m.get("topk"):
            line += f" | top3 {m['topk']['top3']:.4f} top5 {m['topk']['top5']:.4f}"
        return line + "\n"

    output_file = os.path.join(output_dir, f"{stem}.txt")
    with open(output_file, "w") as f:
        f.write("=" * 60 + "\n")
        f.write(f"{model_short_name} Fine-tuning Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Timestamp: {results['timestamp']}\n")
        f.write(f"Model: {results['model']}\n")
        f.write(f"GPU: {results['gpu']}\n")
        f.write(f"Parameters: {results['model_params']['total']:,} total, "
                f"{results['model_params']['trainable']:,} trainable\n")
        if hf_url:
            f.write(f"Hugging Face: {hf_url}\n")
        f.write(f"Seeds: {seed_list} (selected {selected} by median validation strict accuracy)\n")
        f.write(f"Scoreboard: {fingerprints['n_test']} test records, {fingerprints['n_classes']} classes, "
                f"test sha {fingerprints['test_sha256'][:12]}, labels sha "
                f"{fingerprints['label_space_sha256'][:12]}\n")
        f.write("\nConfiguration:\n")
        for k, v in results['config'].items():
            f.write(f"  {k}: {v}\n")
        f.write("\nTraining Metrics (selected seed):\n")
        for k, v in results['train_metrics'].items():
            if v is not None:
                f.write(f"  {k}: {v:.4f}\n" if isinstance(v, float) else f"  {k}: {v}\n")
        f.write("\nEvaluation Metrics (validation set, used for model selection):\n")
        for k, v in results['eval_metrics'].items():
            if v is not None:
                f.write(f"  {k}: {v:.4f}\n")
        f.write("\nTest Metrics (held-out, headline numbers):\n")
        for k, v in results['test_metrics'].items():
            if v is not None:
                f.write(f"  {k}: {v:.4f}\n")

        f.write("\nComposite metrics (metrics_core; strict = exact label, lenient = accept set):\n")
        for s in seed_list:
            f.write(fmt_block(f"seed {s} validation", per_seed[s]["val"]))
            f.write(fmt_block(f"seed {s} test      ", per_seed[s]["test"]))
        if ensemble is not None:
            f.write(fmt_block("ensemble validation", ensemble["val"]))
            f.write(fmt_block("ensemble test      ", ensemble["test"]))
        if agreement:
            f.write(f"  seed agreement on test: pairwise {agreement['pairwise_mean']:.3f}, "
                    f"unanimous {agreement['unanimous']:.3f}\n")
        f.write("\nTest slices (selected seed; strict / lenient / n):\n")
        for name, sl in cand_test["slices"].items():
            f.write(f"  {name:<26} {sl['strict']:.4f} / {sl['lenient']:.4f} / {sl['n']}\n")
        if floor is not None:
            f.write(f"\nLinear floor (TF-IDF+SGD) strict: {floor:.4f} -> transformer margin "
                    f"{cand_test['strict']['accuracy'] - floor:+.4f}"
                    f"{'  (WARNING: < 0.05)' if cand_test['strict']['accuracy'] - floor < 0.05 else ''}\n")
        f.write("\n" + mc.format_axes_table(decision) + "\n")
        f.write(f"\nQuality Gate: {'PASSED' if decision['passed'] else 'FAILED -- Hub push skipped'} "
                f"({decision['reason']})\n")
        f.write(f"\nCalibration temperature (fitted on validation, stored in model config): "
                f"{results['calibration_temperature']:.3f}\n")
        f.write(f"Acceptance threshold ({target_precision:.0%} validation precision, stored in model "
                f"config): {per_seed[selected]['tau']:.3f}\n")

        per_class = results.get('test_per_class') or {}
        scored = [(name, m) for name, m in per_class.items()
                  if isinstance(m, dict) and 'f1-score' in m and m.get('support', 0) > 0]
        if scored:
            worst = sorted(scored, key=lambda x: (x[1]['f1-score'], -x[1]['support']))[:20]
            f.write("\nWorst 20 test classes by F1 (precision / recall / f1 / support):\n")
            for name, m in worst:
                f.write(f"  {name}: {m['precision']:.2f} / {m['recall']:.2f} / "
                        f"{m['f1-score']:.2f} / {int(m['support'])}\n")

        for title, key in [("Validation", 'validation_inference'), ("Test", 'test_inference')]:
            f.write("\n" + "=" * 60 + "\n")
            f.write(f"{title} Inference Results (selected seed)\n")
            f.write("=" * 60 + "\n")
            inf = results.get(key, {})
            f.write(f"Accuracy: {inf.get('correct', 0)}/{inf.get('total', 0)} "
                    f"({inf.get('accuracy', 0):.2%})\n\n")
            f.write("Predictions:\n")
            f.write("-" * 60 + "\n")
            for pred in inf.get('predictions', []):
                status = "✓" if pred['correct'] else "✗"
                f.write(f"{status} Input: '{pred['text']}'\n")
                f.write(f"   Predicted: {pred['predicted_label']} (confidence: {pred['confidence']:.2%})\n")
                f.write(f"   Actual:    {pred['actual_label']}\n\n")

        f.write("=" * 60 + "\n")
        f.write("Raw JSON:\n")
        serializable = {k: v for k, v in results.items() if k not in ("val_logits", "test_logits")}
        f.write(json.dumps(serializable, indent=2, default=str))

    print(f"\nResults saved to: {output_file}")
    print(f"Metrics saved to: {metrics_file}")

    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"GPU: {gpu_upper}")
    for s in seed_list:
        print(f"Seed {s}: val strict {per_seed[s]['val']['strict']['accuracy']:.4f} | "
              f"test strict {per_seed[s]['test']['strict']['accuracy']:.4f} "
              f"lenient {per_seed[s]['test']['lenient']['accuracy']:.4f}")
    if ensemble is not None:
        print(f"Ensemble: test strict {ensemble['test']['strict']['accuracy']:.4f} "
              f"lenient {ensemble['test']['lenient']['accuracy']:.4f}")
    print(f"Gate: {'PASSED' if decision['passed'] else 'FAILED'} ({decision['reason']})")
    print("=" * 60)
