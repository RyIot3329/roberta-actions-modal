"""
Fine-tune DeBERTa-v3 / RoBERTa for HVAC Point Classification on Modal
=====================================================================

Fine-tuning transformer models on Modal's serverless GPU infrastructure.

Usage:
    modal run finetune.py::main
    modal run finetune.py::main --model microsoft/deberta-v3-base
    modal run finetune.py::main --model FacebookAI/roberta-base
    modal run finetune.py::main --gpu A10G --epochs 9 --push-to-hub --hf-repo username/model-name
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
    head_init_seed: int = 1234,
    train_ctx_data: list = None,
    val_ctx_data: list = None,
    test_ctx_data: list = None,
    context_dropout: float = 0.5,
    field_dropout: float = 0.2,
    context_version: str = None,
    early_stopping_patience: int = 3,
    llrd_decay: float = 1.0,
    head_lr_multiplier: float = 1.0,
    rdrop_alpha: float = 0.0,
    logit_adjustment_tau: float = 0.0,
    ema_decay: float = 0.0,
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
        TrainerCallback,
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

    # Fresh head (pooler + classifier) initialized from a FIXED seed, independent
    # of the training seed: seeds then differ only in data order and dropout,
    # which keeps them in one loss basin so their weights can be averaged
    # (weight soup). Without this every seed gets its own random head and
    # averaging is meaningless.
    if head_init_seed is not None and head_init_seed >= 0:
        gen_state = torch.random.get_rng_state()
        torch.manual_seed(head_init_seed)
        n_reinit = 0
        for name, module in model.named_modules():
            if name.split(".")[0] in ("pooler", "classifier") and isinstance(module, torch.nn.Linear):
                model._init_weights(module)
                n_reinit += 1
        torch.random.set_rng_state(gen_state)
        set_seed(seed)
        print(f"Head re-initialized from seed {head_init_seed} ({n_reinit} linear layers)")

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

    # ----- Datasets: name-only records + optional (name, context) records -----
    # The classifier input is "<name> | <context>" (scripts/clean_data.py
    # build_model_input); an empty context is exactly the legacy name-only
    # input. Training applies CONTEXT DROPOUT per example per epoch so the
    # name-only view (the deployed API today) keeps learning, plus per-field
    # dropout so partial context at inference (no units, no description) is
    # in-distribution. Evaluation datasets never drop anything.
    import random

    CONTEXT_SEP = " | "

    class ContextDataset(torch.utils.data.Dataset):
        def __init__(self, records, p_name_only=0.0, p_field=0.0):
            self.records = records
            self.p_name_only = p_name_only
            self.p_field = p_field

        def __len__(self):
            return len(self.records)

        def __getitem__(self, i):
            r = self.records[i]
            ctx = r.get("context") or ""
            if ctx and self.p_name_only > 0:
                # A flip (label differs from the name-only majority) is only
                # right given its context: never blank it, and keep at least
                # one field
                if not r.get("flip") and random.random() < self.p_name_only:
                    ctx = ""
                elif self.p_field > 0:
                    all_fields = ctx.split(CONTEXT_SEP)
                    fields = [f for f in all_fields if random.random() >= self.p_field]
                    if not fields and r.get("flip"):
                        fields = [random.choice(all_fields)]
                    ctx = CONTEXT_SEP.join(fields)
            text = r["text"] if not ctx else f"{r['text']}{CONTEXT_SEP}{ctx}"
            return {"text": text, "label": r["label_id"]}

    def data_collator(batch):
        enc = tokenizer([b["text"] for b in batch], padding=True, truncation=True,
                        max_length=max_seq_length, return_tensors="pt")
        enc["labels"] = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        return enc

    use_context = bool(train_ctx_data)
    train_records = list(train_data) + (list(train_ctx_data) if use_context else [])
    train_dataset = ContextDataset(train_records, p_name_only=context_dropout if use_context else 0.0,
                                   p_field=field_dropout if use_context else 0.0)
    val_dataset = ContextDataset(val_data)
    test_dataset = ContextDataset(test_data)
    val_ctx_dataset = ContextDataset(val_ctx_data) if use_context and val_ctx_data else None
    test_ctx_dataset = ContextDataset(test_ctx_data) if use_context and test_ctx_data else None
    print(f"Training records: {len(train_data)} name-only"
          + (f" + {len(train_ctx_data)} with context (context dropout {context_dropout}, "
             f"field dropout {field_dropout})" if use_context else ""))

    # Warn if the longest inputs would be truncated
    probe_texts = sorted({d["text"] for d in train_records}, key=len)[-50:]
    if use_context:
        probe_texts += sorted((f"{d['text']}{CONTEXT_SEP}{d['context']}" for d in train_ctx_data
                               if d.get("context")), key=len)[-50:]
    longest = max((len(tokenizer(t)["input_ids"]) for t in probe_texts), default=0)
    if longest >= max_seq_length:
        print(f"WARNING: longest input has {longest} tokens >= max_seq_length={max_seq_length}; "
              f"raise max_seq_length")

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
        remove_unused_columns=False,
    )

    # ----- Recipe knobs (all default-off; each is A/B-tested with 3 seeds) -----
    # Layer-wise LR decay: deeper (earlier) layers get lr * decay^(distance
    # from the top); the fresh classifier head gets lr * head_lr_multiplier.
    optimizers = (None, None)
    if llrd_decay != 1.0 or head_lr_multiplier != 1.0:
        base = getattr(model, model.base_model_prefix)
        layers = list(base.encoder.layer)
        n_layers = len(layers)
        no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight", "LayerNorm.bias")
        groups = []
        seen = set()

        def add_group(named, lr):
            decay_params = [p for n, p in named if not any(k in n for k in no_decay)]
            nodecay_params = [p for n, p in named if any(k in n for k in no_decay)]
            for p in decay_params + nodecay_params:
                seen.add(id(p))
            if decay_params:
                groups.append({"params": decay_params, "lr": lr, "weight_decay": weight_decay})
            if nodecay_params:
                groups.append({"params": nodecay_params, "lr": lr, "weight_decay": 0.0})

        add_group(list(base.embeddings.named_parameters()), learning_rate * llrd_decay ** n_layers)
        for i, layer in enumerate(layers):
            add_group(list(layer.named_parameters()), learning_rate * llrd_decay ** (n_layers - 1 - i))
        head = [(n, p) for n, p in model.named_parameters()
                if id(p) not in seen and not n.startswith(model.base_model_prefix + ".encoder.layer")]
        encoder_rest = [(n, p) for n, p in head if n.startswith(model.base_model_prefix + ".")]
        head_only = [(n, p) for n, p in head if not n.startswith(model.base_model_prefix + ".")]
        add_group(encoder_rest, learning_rate)
        add_group(head_only, learning_rate * head_lr_multiplier)
        optimizers = (torch.optim.AdamW(groups, lr=learning_rate, weight_decay=weight_decay), None)
        print(f"LLRD: decay {llrd_decay} over {n_layers} layers, head x{head_lr_multiplier} "
              f"({len(groups)} param groups)")

    # Logit adjustment: subtract tau*log(prior) at train time so rare classes
    # are not drowned by the head classes (Menon et al. 2021)
    log_prior = None
    if logit_adjustment_tau > 0:
        counts = np.bincount([d["label_id"] for d in train_records], minlength=num_labels).astype(np.float64)
        prior = (counts + 1.0) / (counts + 1.0).sum()
        log_prior = torch.tensor(np.log(prior), dtype=torch.float32)

    class RecipeTrainer(Trainer):
        """Trainer with optional R-Drop consistency and logit adjustment."""

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            if rdrop_alpha <= 0 and log_prior is None:
                return super().compute_loss(model, inputs, return_outputs=return_outputs,
                                            num_items_in_batch=num_items_in_batch)
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            adj = logits + logit_adjustment_tau * log_prior.to(logits.device) \
                if (log_prior is not None and model.training) else logits
            loss = torch.nn.functional.cross_entropy(adj, labels, label_smoothing=label_smoothing)
            if rdrop_alpha > 0 and model.training:
                logits2 = model(**inputs).logits
                adj2 = logits2 + logit_adjustment_tau * log_prior.to(logits.device) \
                    if log_prior is not None else logits2
                loss2 = torch.nn.functional.cross_entropy(adj2, labels, label_smoothing=label_smoothing)
                p, q = torch.log_softmax(adj, -1), torch.log_softmax(adj2, -1)
                kl = (torch.nn.functional.kl_div(p, q, log_target=True, reduction="batchmean")
                      + torch.nn.functional.kl_div(q, p, log_target=True, reduction="batchmean")) / 2
                loss = (loss + loss2) / 2 + rdrop_alpha * kl
            inputs["labels"] = labels
            return (loss, outputs) if return_outputs else loss

    class EMACallback(TrainerCallback):
        """Exponential moving average of weights; the EMA weights are swapped in
        before each epoch-level evaluation/checkpoint (on_epoch_end fires before
        the epoch evaluation in the Trainer loop) and swapped back for training."""

        def __init__(self, decay):
            self.decay = decay
            self.shadow = None
            self.backup = None

        def on_train_begin(self, args, state, control, model=None, **kwargs):
            self.shadow = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

        def on_step_end(self, args, state, control, model=None, **kwargs):
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if n in self.shadow:
                        self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

        def on_epoch_end(self, args, state, control, model=None, **kwargs):
            self.backup = {n: p.detach().clone() for n, p in model.named_parameters() if n in self.shadow}
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if n in self.shadow:
                        p.copy_(self.shadow[n])

        def on_epoch_begin(self, args, state, control, model=None, **kwargs):
            if self.backup is not None:
                with torch.no_grad():
                    for n, p in model.named_parameters():
                        if n in self.backup:
                            p.copy_(self.backup[n])
                self.backup = None

    callbacks = []
    if early_stopping_patience and early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))
    else:
        print("Early stopping disabled: the cosine schedule anneals fully; best epoch by validation")
    if ema_decay > 0:
        callbacks.append(EMACallback(ema_decay))
        print(f"EMA of weights enabled (decay {ema_decay})")
    if rdrop_alpha > 0:
        print(f"R-Drop enabled (alpha {rdrop_alpha}): two forward passes per step")
    if log_prior is not None:
        print(f"Logit adjustment enabled (tau {logit_adjustment_tau})")

    trainer = RecipeTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        optimizers=optimizers,
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
            for key in ("site", "rows", "seen_in_train", "accept", "context", "pair_seen_in_train"):
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

    # Context views (evaluated once, no dropout): the same model scored on
    # "<name> | <context>" pairs of the held-out sites
    val_ctx_predictions = test_ctx_predictions = None
    val_ctx_name_predictions = test_ctx_name_predictions = None
    val_ctx_output = test_ctx_output = None

    def blank_context(records):
        return [dict(r, context="") for r in records]

    if val_ctx_dataset is not None:
        print("\nRunning validation (context) inference...")
        val_ctx_output = trainer.predict(val_ctx_dataset)
        val_ctx_predictions = build_predictions(val_ctx_output, val_ctx_data, temperature)
        # Name-only view on the same pairs: identical texts with the context
        # blanked, so the two views (and their ensemble) align record by record
        val_ctx_name_predictions = build_predictions(
            trainer.predict(ContextDataset(blank_context(val_ctx_data))), val_ctx_data, temperature)
    if test_ctx_dataset is not None:
        print("Running test (context) inference...")
        test_ctx_output = trainer.predict(test_ctx_dataset)
        test_ctx_predictions = build_predictions(test_ctx_output, test_ctx_data, temperature)
        test_ctx_name_predictions = build_predictions(
            trainer.predict(ContextDataset(blank_context(test_ctx_data))), test_ctx_data, temperature)
    if use_context:
        model.config.context_version = context_version or "1"
        model.config.context_trained = True
        model.config.context_dropout = context_dropout

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
            "early_stopping_patience": early_stopping_patience,
            "llrd_decay": llrd_decay,
            "head_lr_multiplier": head_lr_multiplier,
            "rdrop_alpha": rdrop_alpha,
            "logit_adjustment_tau": logit_adjustment_tau,
            "ema_decay": ema_decay,
            "head_init_seed": head_init_seed,
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
        "validation_ctx_inference": None if val_ctx_predictions is None else {
            "accuracy": sum(1 for p in val_ctx_predictions if p["correct"]) / max(1, len(val_ctx_predictions)),
            "total": len(val_ctx_predictions),
            "predictions": val_ctx_predictions,
        },
        "test_ctx_inference": None if test_ctx_predictions is None else {
            "accuracy": sum(1 for p in test_ctx_predictions if p["correct"]) / max(1, len(test_ctx_predictions)),
            "total": len(test_ctx_predictions),
            "predictions": test_ctx_predictions,
        },
        "validation_ctx_name_inference": None if val_ctx_name_predictions is None else {
            "predictions": val_ctx_name_predictions},
        "test_ctx_name_inference": None if test_ctx_name_predictions is None else {
            "predictions": test_ctx_name_predictions},
        "context": {"enabled": use_context, "version": context_version,
                    "context_dropout": context_dropout, "field_dropout": field_dropout,
                    "train_ctx_records": len(train_ctx_data) if use_context else 0},
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
        "val_ctx_logits": None if val_ctx_output is None else val_ctx_output.predictions.astype(np.float16),
        "test_ctx_logits": None if test_ctx_output is None else test_ctx_output.predictions.astype(np.float16),
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


def _score_context_views(val_ctx_preds, test_ctx_preds, val_name_preds, test_name_preds,
                         target_precision, mc):
    """Score the context view, the name-only view on the same (name, context)
    pairs (aligned record by record), and their max-confidence ensemble."""
    def ensemble_view(ctx_preds, name_preds):
        out = []
        for p, n in zip(ctx_preds, name_preds):
            q = dict(p)
            if n["confidence"] > p["confidence"]:
                for key in ("predicted_label", "predicted_id", "confidence", "correct", "topk_labels"):
                    q[key] = n.get(key)
            q["source"] = "name" if n["confidence"] > p["confidence"] else "context"
            out.append(q)
        return out

    views = {}
    for split, ctx_preds, name_preds in (("val", val_ctx_preds, val_name_preds),
                                          ("test", test_ctx_preds, test_name_preds)):
        assert len(ctx_preds) == len(name_preds), "context and name views must align"
        views[split] = {"context": ctx_preds, "name_on_pairs": name_preds,
                        "ensemble": ensemble_view(ctx_preds, name_preds)}
    out = {"val": {}, "test": {}, "tau": {}, "test_preds": views["test"]}
    for view in ("context", "name_on_pairs", "ensemble"):
        tau = mc.fit_acceptance_threshold(views["val"][view], target_precision)
        out["tau"][view] = tau
        out["val"][view] = mc.score_predictions(views["val"][view], tau=tau)
        out["test"][view] = mc.score_predictions(views["test"][view], tau=tau)
    out["ensemble_source_context_share"] = float(np.mean(
        [q["source"] == "context" for q in views["test"]["ensemble"]])) if views["test"]["ensemble"] else None
    return out


def _ctx_summary(ctx):
    if not ctx:
        return None
    return {view: {"strict": ctx["test"][view]["strict"]["accuracy"],
                   "lenient": ctx["test"][view]["lenient"]["accuracy"],
                   "log1p_rows": ctx["test"][view]["strict"]["log1p_rows_accuracy"],
                   "n": ctx["test"][view]["strict"]["n"],
                   "tau": ctx["tau"][view]}
            for view in ("name_on_pairs", "context", "ensemble")}


def _ctx_full(ctx):
    if not ctx:
        return None
    return {split: {view: {k: v for k, v in ctx[split][view].items() if k != "coverage_curve"}
                    for view in ctx[split]} for split in ("val", "test")} | {"tau": ctx["tau"]}


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
    context: bool = True,
    context_dropout: float = 0.5,
    field_dropout: float = 0.2,
    early_stopping_patience: int = 3,
    llrd_decay: float = 1.0,
    head_lr_multiplier: float = 1.0,
    rdrop_alpha: float = 0.0,
    logit_adjustment_tau: float = 0.0,
    ema_decay: float = 0.0,
    soup: bool = True,
    head_init_seed: int = 1234,
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
    # Optional (name, context) views written by convert_to_jsonl.py
    ctx_paths = {k: f"data/{k}_ctx.jsonl" for k in ("train", "validation", "test")}
    use_context = context and all(os.path.exists(v) for v in ctx_paths.values())
    if context and not use_context:
        print("WARNING: --context requested but data/*_ctx.jsonl missing; training name-only")
    train_ctx = _load_jsonl(ctx_paths["train"]) if use_context else None
    val_ctx = _load_jsonl(ctx_paths["validation"]) if use_context else None
    test_ctx = _load_jsonl(ctx_paths["test"]) if use_context else None
    context_version = None
    if use_context:
        from clean_data import CONTEXT_VERSION as context_version  # noqa: N812
        print(f"Context views: {len(train_ctx)} train / {len(val_ctx)} val / {len(test_ctx)} test "
              f"(name, context) records; context version {context_version}")
    label2id = label_mapping["label2id"]
    id2label = label_mapping["id2label"]
    id2label_int = {int(k): v for k, v in id2label.items()}
    num_labels = label_mapping["num_labels"]
    print(f"Loaded {len(train_data)} train, {len(val_data)} val, {len(test_data)} test samples")
    print(f"Number of labels: {num_labels}")

    # ----- Scoreboard + baseline -----
    fingerprints = mc.fingerprints(test_data, list(label2id), pair_records=test_ctx)
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
        train_ctx_data=train_ctx, val_ctx_data=val_ctx, test_ctx_data=test_ctx,
        context_dropout=context_dropout, field_dropout=field_dropout,
        context_version=context_version,
        early_stopping_patience=early_stopping_patience, llrd_decay=llrd_decay,
        head_lr_multiplier=head_lr_multiplier, rdrop_alpha=rdrop_alpha,
        logit_adjustment_tau=logit_adjustment_tau, ema_decay=ema_decay,
        head_init_seed=head_init_seed,
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
        # Context views: the same model on (name, context) pairs; the name-only
        # prediction of each pair's text; and their max-confidence ensemble
        if use_context and res.get("test_ctx_inference"):
            per_seed[s]["ctx"] = _score_context_views(
                res["validation_ctx_inference"]["predictions"],
                res["test_ctx_inference"]["predictions"],
                res["validation_ctx_name_inference"]["predictions"],
                res["test_ctx_name_inference"]["predictions"],
                target_precision, mc)
            # Operational view on the pair test: what the service returns when
            # callers send context (max-confidence of name and context views)
            per_seed[s]["pairs"] = per_seed[s]["ctx"]["test_preds"]["ensemble"]
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
    cand_name = f"seed{selected}"
    cand_save_name = results["save_name"]
    cand_temperature = results.get("calibration_temperature")
    cand_tau = per_seed[selected]["tau"]
    cand_ctx = per_seed[selected].get("ctx")
    cand_pairs = per_seed[selected].get("pairs")
    print(f"\nSelected seed {selected} (median validation strict accuracy "
          f"{per_seed[selected]['val']['strict']['accuracy']:.4f})")

    # Weight soup of the seeds (same init, same data): free at inference. It
    # replaces the median seed as the candidate only when its VALIDATION strict
    # accuracy is higher. Any soup failure leaves the run's results intact.
    soup_entry = None
    if soup and len(seed_list) > 1:
        try:
            print(f"Building the greedy weight soup of seeds {seed_list}...")
            soup_res = soup_eval.remote(seed_list, val_data, test_data, val_ctx, test_ctx,
                                        max_seq_length, True, "final_model_soup")
            s_val = soup_res["validation_inference"]["predictions"]
            s_test = soup_res["test_inference"]["predictions"]
            s_tau = mc.fit_acceptance_threshold(s_val, target_precision)
            soup_entry = {"selected_seeds": soup_res["selected_seeds"], "tau": s_tau,
                          "val": mc.score_predictions(s_val, tau=s_tau),
                          "test": mc.score_predictions(s_test, tau=s_tau),
                          "test_preds": s_test, "temperature": soup_res["calibration_temperature"],
                          "per_seed_val_accuracy": soup_res["per_seed_val_accuracy"]}
            if use_context and soup_res.get("test_ctx_inference"):
                soup_entry["ctx"] = _score_context_views(
                    soup_res["validation_ctx_inference"]["predictions"],
                    soup_res["test_ctx_inference"]["predictions"],
                    soup_res["validation_ctx_name_inference"]["predictions"],
                    soup_res["test_ctx_name_inference"]["predictions"], target_precision, mc)
                soup_entry["pairs"] = soup_entry["ctx"]["test_preds"]["ensemble"]
            print(f"Soup of {soup_entry['selected_seeds']}: validation strict "
                  f"{soup_entry['val']['strict']['accuracy']:.4f} vs selected seed "
                  f"{per_seed[selected]['val']['strict']['accuracy']:.4f}")
            # A one-seed soup is that seed's own weights; require a real merge
            # and a margin above inference noise before swapping candidates
            if (len(soup_entry["selected_seeds"]) > 1
                    and soup_entry["val"]["strict"]["accuracy"]
                    > per_seed[selected]["val"]["strict"]["accuracy"] + 0.002):
                cand_name = f"soup{soup_entry['selected_seeds']}"
                cand_test = soup_entry["test"]
                cand_preds = soup_entry["test_preds"]
                cand_save_name = "final_model_soup"
                cand_temperature = soup_entry["temperature"]
                cand_tau = soup_entry["tau"]
                cand_ctx = soup_entry.get("ctx")
                cand_pairs = soup_entry.get("pairs")
                print(f"Candidate: {cand_name} (validation beat the median seed)")
        except Exception as e:  # noqa: BLE001
            print(f"WARNING: soup evaluation failed ({type(e).__name__}: {e}); candidate stays seed {selected}")
            soup_entry = None

    if cand_pairs:
        cand_test = dict(cand_test)
        cand_test["pairs_ensemble"] = {k: v for k, v in cand_ctx["test"]["ensemble"].items()
                                       if k != "coverage_curve"}
        cand_test["pairs_context"] = {k: v for k, v in cand_ctx["test"]["context"].items()
                                      if k != "coverage_curve"}
        cand_test["pairs_name"] = {k: v for k, v in cand_ctx["test"]["name_on_pairs"].items()
                                   if k != "coverage_curve"}
    candidate_record = mc.build_metrics_record(
        cand_test, fingerprints, model=model_name, hf_repo=hf_repo, git_sha=_git_sha(),
        timestamp=results["timestamp"], seeds=seed_list, selected_seed=selected,
        extra={
            "candidate": cand_name,
            "candidate_save_name": cand_save_name,
            "calibration_temperature": cand_temperature,
            "acceptance_threshold": cand_tau,
            "target_precision": target_precision,
            "validation_metrics": {k: v for k, v in
                                   (soup_entry["val"] if cand_name.startswith("soup") else per_seed[selected]["val"]).items()
                                   if k != "coverage_curve"},
            "soup": None if soup_entry is None else {
                "selected_seeds": soup_entry["selected_seeds"],
                "per_seed_val_accuracy": soup_entry["per_seed_val_accuracy"],
                "val_strict": soup_entry["val"]["strict"]["accuracy"],
                "test_strict": soup_entry["test"]["strict"]["accuracy"],
                "test_lenient": soup_entry["test"]["lenient"]["accuracy"],
                "test_log1p_rows": soup_entry["test"]["strict"]["log1p_rows_accuracy"],
                "is_candidate": cand_name.startswith("soup")},
            "seed_summary": {str(s): {"val_strict": per_seed[s]["val"]["strict"]["accuracy"],
                                      "test_strict": per_seed[s]["test"]["strict"]["accuracy"],
                                      "test_f1_weighted": per_seed[s]["test"]["strict"]["f1_weighted"],
                                      "test_lenient": per_seed[s]["test"]["lenient"]["accuracy"],
                                      "tau": per_seed[s]["tau"]} for s in seed_list},
            "seed_agreement": agreement,
            "context": None if not use_context else {
                "context_version": context_version, "context_dropout": context_dropout,
                "field_dropout": field_dropout,
                "test": _ctx_summary(cand_ctx)},
            "ensemble": None if ensemble is None else {
                "test_strict": ensemble["test"]["strict"]["accuracy"],
                "test_lenient": ensemble["test"]["lenient"]["accuracy"],
                "val_strict": ensemble["val"]["strict"]["accuracy"],
                "test_log1p_rows": ensemble["test"]["strict"]["log1p_rows_accuracy"]},
        },
    )
    baseline_pairs = None
    if baseline is not None:
        bpp = baseline.get("predictions_pairs_path")
        if bpp and os.path.exists(bpp):
            baseline_pairs = mc.load_predictions_jsonl(bpp)
        try:
            decision = mc.promote_decision(candidate_record, baseline, cand_preds, baseline_preds,
                                           candidate_pairs=cand_pairs, baseline_pairs=baseline_pairs)
        except Exception as e:  # noqa: BLE001 -- a gate bug must never lose a run's results
            decision = {"passed": False, "reason": f"gate error: {type(e).__name__}: {e}", "axes": []}
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
        msg = (f"{model_name.split('/')[-1]}: {cand_name} of seeds {seed_list}, {epochs}ep, "
               f"bs{batch_size}x{gradient_accumulation}, lr{learning_rate}, "
               f"test strict {cand_test['strict']['accuracy']:.4f}")
        hf_url = push_saved_model.remote(cand_save_name, hf_repo, hf_token, msg)
        os.makedirs(os.path.dirname(baseline_path) or ".", exist_ok=True)
        pred_path = os.path.join(os.path.dirname(baseline_path) or ".", "best_predictions.jsonl")
        candidate_record["predictions_path"] = pred_path
        candidate_record["huggingface_url"] = hf_url
        if cand_pairs:
            pairs_path = os.path.join(os.path.dirname(baseline_path) or ".", "best_predictions_pairs.jsonl")
            candidate_record["predictions_pairs_path"] = pairs_path
            mc.write_predictions_jsonl(cand_pairs, pairs_path)
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
                                                  if k != "coverage_curve"},
                                         "context": _ctx_full(per_seed[s].get("ctx"))}
                                for s in seed_list},
                   "ensemble": None if ensemble is None else {
                       "tau": ensemble["tau"],
                       "validation": {k: v for k, v in ensemble["val"].items() if k != "coverage_curve"},
                       "test": {k: v for k, v in ensemble["test"].items() if k != "coverage_curve"}},
                   "soup": None if soup_entry is None else {
                       "selected_seeds": soup_entry["selected_seeds"], "tau": soup_entry["tau"],
                       "validation": {k: v for k, v in soup_entry["val"].items() if k != "coverage_curve"},
                       "test": {k: v for k, v in soup_entry["test"].items() if k != "coverage_curve"},
                       "context": _ctx_full(soup_entry.get("ctx"))},
                   "coverage_curve_test": cand_test.get("coverage_curve")},
                  f, indent=2, default=str)
    mc.write_predictions_jsonl(cand_preds, os.path.join(output_dir, f"{stem}_predictions.jsonl"))
    if cand_pairs:
        mc.write_predictions_jsonl(cand_pairs, os.path.join(output_dir, f"{stem}_predictions_pairs.jsonl"))

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
        if soup_entry is not None:
            f.write(fmt_block(f"soup{soup_entry['selected_seeds']} validation", soup_entry["val"]))
            f.write(fmt_block(f"soup{soup_entry['selected_seeds']} test      ", soup_entry["test"]))
        f.write(f"  candidate: {cand_name} (saved as {cand_save_name})\n")
        if agreement:
            f.write(f"  seed agreement on test: pairwise {agreement['pairwise_mean']:.3f}, "
                    f"unanimous {agreement['unanimous']:.3f}\n")
        if use_context:
            f.write("\nContext views on (name, context) pairs of the held-out sites "
                    "(strict / lenient / log1p-rows / n):\n")
            for s in seed_list:
                c = per_seed[s].get("ctx")
                if not c:
                    continue
                for view in ("name_on_pairs", "context", "ensemble"):
                    for split in ("val", "test"):
                        m = c[split][view]
                        f.write(f"  seed {s} {split:<4} {view:<14} {m['strict']['accuracy']:.4f} / "
                                f"{m['lenient']['accuracy']:.4f} / {m['strict']['log1p_rows_accuracy']:.4f} "
                                f"/ {m['strict']['n']}\n")
                for name, sl in c["test"]["context"]["slices"].items():
                    f.write(f"    test context slice {name:<24} strict {sl['strict']:.4f} "
                            f"lenient {sl['lenient']:.4f} n={sl['n']}\n")
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
    if soup_entry is not None:
        print(f"Soup {soup_entry['selected_seeds']}: val strict {soup_entry['val']['strict']['accuracy']:.4f} | "
              f"test strict {soup_entry['test']['strict']['accuracy']:.4f} "
              f"lenient {soup_entry['test']['lenient']['accuracy']:.4f}")
    print(f"Candidate: {cand_name}")
    print(f"Gate: {'PASSED' if decision['passed'] else 'FAILED'} ({decision['reason']})")
    print("=" * 60)


# =============================================================================
# Weight soup: average the seeds' fine-tuned weights (same init, same data)
# =============================================================================

@app.function(gpu="A100", timeout=60 * 60)
def soup_eval(seeds: list, val_data: list, test_data: list, val_ctx_data: list = None,
              test_ctx_data: list = None, max_seq_length: int = 64, greedy: bool = True,
              save_name: str = "final_model_soup") -> dict:
    """Uniform or greedy weight soup over final_model_seed{N} dirs on the
    volume. Greedy: start from the seed with the best validation accuracy,
    add seeds one at a time and keep them only if validation accuracy does
    not drop (Wortsman et al. 2022). Returns name-only (and context-view)
    predictions of the soup plus the per-seed validation accuracies, and
    saves the soup with the mean calibration temperature and the selected
    seeds recorded in its config."""
    import json
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    volume.reload()
    dirs = {s: volume_path / f"final_model_seed{s}" for s in seeds}
    missing = [str(d) for d in dirs.values() if not d.exists()]
    if missing:
        raise FileNotFoundError(f"missing soup ingredients on the volume: {missing}")
    tokenizer = AutoTokenizer.from_pretrained(str(dirs[seeds[0]]))
    models = {s: AutoModelForSequenceClassification.from_pretrained(str(d)).eval() for s, d in dirs.items()}
    id2label = {int(k): v for k, v in models[seeds[0]].config.id2label.items()}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def inputs_of(records):
        return [r["text"] if not r.get("context") else f"{r['text']} | {r['context']}" for r in records]

    @torch.no_grad()
    def logits_of(model, records):
        model.to(device)
        texts = inputs_of(records)
        out = []
        for i in range(0, len(texts), 256):
            enc = tokenizer(texts[i:i + 256], padding=True, truncation=True,
                            max_length=max_seq_length, return_tensors="pt").to(device)
            out.append(model(**enc).logits.float().cpu())
        model.to("cpu")
        return torch.cat(out)

    val_labels = torch.tensor([r["label_id"] for r in val_data])

    def val_acc(model):
        return float((logits_of(model, val_data).argmax(1) == val_labels).float().mean())

    per_seed_val = {s: val_acc(m) for s, m in models.items()}
    order = sorted(seeds, key=lambda s: -per_seed_val[s])
    print(f"Per-seed validation accuracy: {per_seed_val}")

    def average(selected):
        base = models[selected[0]]
        soup = AutoModelForSequenceClassification.from_pretrained(str(dirs[selected[0]])).eval()
        sd = {k: v.clone().float() for k, v in base.state_dict().items()}
        for s in selected[1:]:
            for k, v in models[s].state_dict().items():
                if sd[k].dtype.is_floating_point:
                    sd[k] += v.float()
        for k in sd:
            if sd[k].dtype.is_floating_point:
                sd[k] /= len(selected)
        soup.load_state_dict({k: v.to(base.state_dict()[k].dtype) for k, v in sd.items()})
        return soup

    if greedy:
        selected = [order[0]]
        best = per_seed_val[order[0]]
        for s in order[1:]:
            cand = average(selected + [s])
            acc = val_acc(cand)
            if acc >= best - 1e-9:
                selected.append(s)
                best = acc
                print(f"  + seed {s}: validation {acc:.4f} (kept)")
            else:
                print(f"  + seed {s}: validation {acc:.4f} (dropped)")
    else:
        selected = list(seeds)
    soup = average(selected)
    temps = [float(getattr(models[s].config, "calibration_temperature", 1.0) or 1.0) for s in selected]
    taus = [float(getattr(models[s].config, "acceptance_threshold", 1.0) or 1.0) for s in selected]
    temperature = sum(temps) / len(temps)
    soup.config.calibration_temperature = temperature
    soup.config.acceptance_threshold = sum(taus) / len(taus)
    soup.config.soup_seeds = selected
    soup_dir = volume_path / save_name
    soup_dir.mkdir(parents=True, exist_ok=True)
    soup.save_pretrained(str(soup_dir))
    tokenizer.save_pretrained(str(soup_dir))
    volume.commit()

    def predictions(model, records):
        probs = torch.softmax(logits_of(model, records) / temperature, dim=1)
        conf, ids = probs.max(dim=1)
        topk = probs.topk(min(10, probs.shape[1]), dim=1).indices.tolist()
        out = []
        for r, i, c, tk in zip(records, ids.tolist(), conf.tolist(), topk):
            rec = {"text": r["text"], "actual_label": r["label"], "actual_id": r["label_id"],
                   "predicted_label": id2label[i], "predicted_id": i, "confidence": c,
                   "correct": i == r["label_id"], "topk_labels": [id2label[j] for j in tk]}
            for key in ("site", "rows", "seen_in_train", "accept", "context", "pair_seen_in_train"):
                if key in r:
                    rec[key] = r[key]
            out.append(rec)
        return out

    result = {
        "selected_seeds": selected,
        "per_seed_val_accuracy": per_seed_val,
        "calibration_temperature": temperature,
        "save_name": save_name,
        "validation_inference": {"predictions": predictions(soup, val_data)},
        "test_inference": {"predictions": predictions(soup, test_data)},
    }
    if val_ctx_data:
        result["validation_ctx_inference"] = {"predictions": predictions(soup, val_ctx_data)}
        result["validation_ctx_name_inference"] = {
            "predictions": predictions(soup, [dict(r, context="") for r in val_ctx_data])}
    if test_ctx_data:
        result["test_ctx_inference"] = {"predictions": predictions(soup, test_ctx_data)}
        result["test_ctx_name_inference"] = {
            "predictions": predictions(soup, [dict(r, context="") for r in test_ctx_data])}
    return result


@app.local_entrypoint()
def soup(seeds: str = "42,43,44", greedy: bool = True, max_seq_length: int = 64,
         output_dir: str = "output", target_precision: float = 0.85,
         baseline_path: str = "output/best_metrics.json"):
    """Evaluate a weight soup of already-trained seeds (on the volume) with
    the same scorer and gate as a training run; writes
    output/<timestamp>_soup_metrics.json (+ predictions) and prints the
    composite comparison. Push manually with push_saved_model if adopted."""
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import metrics_core as mc

    seed_list = [int(x) for x in seeds.split(",") if x.strip()]
    val_data = _load_jsonl("data/validation.jsonl")
    test_data = _load_jsonl("data/test.jsonl")
    val_ctx = _load_jsonl("data/validation_ctx.jsonl") if os.path.exists("data/validation_ctx.jsonl") else None
    test_ctx = _load_jsonl("data/test_ctx.jsonl") if os.path.exists("data/test_ctx.jsonl") else None
    with open("data/label_mapping.json") as f:
        label2id = json.load(f)["label2id"]
    res = soup_eval.remote(seed_list, val_data, test_data, val_ctx, test_ctx, max_seq_length, greedy)
    val_preds = res["validation_inference"]["predictions"]
    test_preds = res["test_inference"]["predictions"]
    tau = mc.fit_acceptance_threshold(val_preds, target_precision)
    val_m = mc.score_predictions(val_preds, tau=tau)
    test_m = mc.score_predictions(test_preds, tau=tau)
    fp = mc.fingerprints(test_data, list(label2id))
    record = mc.build_metrics_record(test_m, fp, model=f"soup{res['selected_seeds']}",
                                     seeds=seed_list, selected_seed=None,
                                     timestamp=datetime.now().isoformat(), git_sha=_git_sha(),
                                     extra={"soup": {"selected_seeds": res["selected_seeds"],
                                                     "per_seed_val_accuracy": res["per_seed_val_accuracy"],
                                                     "greedy": greedy},
                                            "calibration_temperature": res["calibration_temperature"],
                                            "acceptance_threshold": tau,
                                            "validation_metrics": {k: v for k, v in val_m.items()
                                                                   if k != "coverage_curve"}})
    ctx = None
    if res.get("test_ctx_inference"):
        ctx = _score_context_views(res["validation_ctx_inference"]["predictions"],
                                   res["test_ctx_inference"]["predictions"],
                                   res["validation_ctx_name_inference"]["predictions"],
                                   res["test_ctx_name_inference"]["predictions"],
                                   target_precision, mc)
        record["context"] = {"test": _ctx_summary(ctx)}
    decision = {"passed": False, "reason": "no baseline", "axes": []}
    if os.path.exists(baseline_path):
        baseline = mc.legacy_baseline(baseline_path)
        bp = baseline.get("predictions_path")
        bpreds = mc.load_predictions_jsonl(bp) if bp and os.path.exists(bp) else None
        decision = mc.promote_decision(record, baseline, test_preds, bpreds)
    os.makedirs(output_dir, exist_ok=True)
    stem = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_soup"
    with open(os.path.join(output_dir, f"{stem}_metrics.json"), "w") as f:
        json.dump({"candidate": record, "gate": decision, "context": _ctx_full(ctx)}, f, indent=2,
                  default=str)
    mc.write_predictions_jsonl(test_preds, os.path.join(output_dir, f"{stem}_predictions.jsonl"))
    print(f"\nSoup of {res['selected_seeds']} (per-seed val {res['per_seed_val_accuracy']}):")
    print(f"  validation strict {val_m['strict']['accuracy']:.4f} | test strict "
          f"{test_m['strict']['accuracy']:.4f} lenient {test_m['lenient']['accuracy']:.4f} "
          f"log1p-rows {test_m['strict']['log1p_rows_accuracy']:.4f}")
    if ctx:
        for view in ("name_on_pairs", "context", "ensemble"):
            m = ctx["test"][view]
            print(f"  test pairs {view:<14} strict {m['strict']['accuracy']:.4f} "
                  f"lenient {m['lenient']['accuracy']:.4f}")
    print(mc.format_axes_table(decision))
    print(f"Saved: {output_dir}/{stem}_metrics.json (soup model on the volume as "
          f"{res['save_name']}; push with push_saved_model if adopted)")
