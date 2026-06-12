"""
Domain-Adaptive Pretraining (DAPT) for the point-name encoder
=============================================================

Continues masked-language-model pretraining of a transformer on UNLABELED
point names so the encoder learns BMS naming statistics (abbreviation
synonymy, equipment families) before classification fine-tuning. Labels
are never used; the corpus is just names.

Corpus sources (all normalized with the pipeline's normalize_text):
  - data/real_points.csv, TRAIN sites only -- held-out site exports are
    never read, so the cross-site benchmark stays honest (names + any
    descriptions)
  - data/cleaned_data.csv (synthetic templates)
  - data/synthetic_points.csv (generated eo66 variants, if present)
  - eo66 Display Names (public taxonomy text)
  - data/public_points.txt (real names from public research datasets;
    refresh with --fetch-public, which pulls the plastering benchmark's
    ground-truth files: SDH/SODA/IBM/GHC/UVA/UCB buildings)
  - --extra file(s): one raw point name per line

Per-site DAPT at deployment is the same mechanism pointed at a new
building's unlabeled point list -- there is no benchmark to protect there.

Workflow:
    python scripts/dapt.py --fetch-public      # refresh data/public_points.txt
    python scripts/dapt.py --corpus-only       # build + inspect the corpus
    modal run scripts/dapt.py --push-to-hub    # pretrain on Modal, push to HF
Then point `model:` in config/training.yml at the DAPT repo and push --
the fine-tune and its quality gate judge whether DAPT actually helped.

The DAPT artifact has no gate of its own: it is an encoder initialization,
scored only through the downstream fine-tune.
"""

import argparse
import json
import os
import random
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEFAULT_HF_REPO = "RyIoT33/deberta-v3-bms-base"
PUBLIC_POINTS_PATH = "data/public_points.txt"
TRAIN_TIMEOUT = 120 * 60

# Real point names from the plastering cross-building benchmark
# (github.com/plastering/plastering, the SDH/SODA/IBM/GHC/UVA/UCB buildings).
# Each entry: (url, format) where format selects the parser below.
# (ghc_full_parsing.json and the ucb_* files carry numeric IDs / vocabulary
# maps rather than point names, so they are deliberately absent)
PUBLIC_SOURCES = [
    ("https://raw.githubusercontent.com/plastering/plastering/master/groundtruth/SDH-GROUND-TRUTH", "groundtruth"),
    ("https://raw.githubusercontent.com/plastering/plastering/master/groundtruth/SODA-GROUND-TRUTH", "groundtruth"),
    ("https://raw.githubusercontent.com/plastering/plastering/master/groundtruth/IBM-GROUND-TRUTH", "groundtruth"),
    ("https://raw.githubusercontent.com/plastering/plastering/master/groundtruth/sdh_full_parsing.json", "json-keys"),
    ("https://raw.githubusercontent.com/plastering/plastering/master/groundtruth/uva_cse_point_map.csv", "csv-first-column"),
]


def _parse_public(text: str, fmt: str) -> list:
    """Extract raw point names from one public source."""
    if fmt == "groundtruth":
        # Alternating lines: a raw name, then its parse (which contains
        # ":c"/":v" tag markers). Keep the name lines.
        return [line.strip() for line in text.splitlines()
                if line.strip() and ':c' not in line and ':v' not in line]
    if fmt == "json-keys":
        return [str(k).strip() for k in json.loads(text)]
    if fmt == "csv-first-column":
        import csv
        import io
        rows = list(csv.reader(io.StringIO(text)))
        return [row[0].strip() for row in rows[1:] if row and row[0].strip()]
    if fmt == "lines":
        return [line.strip() for line in text.splitlines() if line.strip()]
    return []


def fetch_public(out_path=PUBLIC_POINTS_PATH):
    """Download public research point names into a committed text file."""
    names = []
    for url, fmt in PUBLIC_SOURCES:
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                text = resp.read().decode("utf-8", errors="replace")
            got = _parse_public(text, fmt)
            names.extend(got)
            print(f"  {len(got):6d} names  {url.rsplit('/', 1)[-1]}")
        except Exception as e:
            print(f"  WARNING: {url.rsplit('/', 1)[-1]}: {type(e).__name__}: {e}")
    # Names that are pure indices or JSON fragments carry no naming signal
    unique = sorted({n for n in names if n and not n.isdigit() and '"' not in n})
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(unique) + "\n")
    print(f"Saved {len(unique)} unique raw names -> {out_path}")
    return unique


def build_corpus(extra_paths=(), seed=42, eval_fraction=0.05):
    """Assemble the deduplicated, normalized DAPT corpus.

    Returns (train_texts, eval_texts). Held-out sites are excluded at the
    SOURCE level (their export rows are never read); generic strings that
    happen to recur across sites are legitimate corpus members.
    """
    import pandas as pd
    from clean_data import normalize_text
    from convert_to_jsonl import VAL_SITES, TEST_SITES

    sources = {}

    real = pd.read_csv("data/real_points.csv")
    real = real[~real["site"].isin(VAL_SITES + TEST_SITES)]
    names = {normalize_text(str(n)) for n in real["name"]}
    if "description" in real.columns:
        names |= {normalize_text(str(d)) for d in real["description"].fillna("")
                  if str(d).strip()}
    sources["real train sites"] = names

    synth = pd.read_csv("data/cleaned_data.csv")
    sources["synthetic templates"] = {str(t) for t in synth["text"]}

    if os.path.exists("data/synthetic_points.csv"):
        gen = pd.read_csv("data/synthetic_points.csv")
        sources["generated eo66 variants"] = {str(t) for t in gen["text"]}

    eo66 = pd.read_excel("data/eo66.xlsx")
    sources["eo66 display names"] = {
        normalize_text(str(n)) for n in eo66["Display Name"].dropna()}

    if os.path.exists(PUBLIC_POINTS_PATH):
        with open(PUBLIC_POINTS_PATH, encoding="utf-8") as f:
            sources["public research buildings"] = {
                normalize_text(line.strip()) for line in f if line.strip()}
    else:
        print(f"NOTE: {PUBLIC_POINTS_PATH} not found -- run --fetch-public to add "
              "real names from public research buildings")

    for path in extra_paths:
        with open(path, encoding="utf-8") as f:
            sources[f"extra: {path}"] = {
                normalize_text(line.strip()) for line in f if line.strip()}

    corpus = set()
    print("Corpus sources:")
    for name, texts in sources.items():
        texts = {t for t in texts if t}
        new = len(texts - corpus)
        corpus |= texts
        print(f"  {len(texts):6d} texts ({new:6d} new)  {name}")

    texts = sorted(corpus)
    rng = random.Random(seed)
    rng.shuffle(texts)
    n_eval = max(1, int(len(texts) * eval_fraction))
    eval_texts, train_texts = texts[:n_eval], texts[n_eval:]
    print(f"Total: {len(texts)} unique texts -> {len(train_texts)} train, "
          f"{len(eval_texts)} eval (perplexity tracking)")
    return train_texts, eval_texts


def _dapt_impl(
    train_texts: list,
    eval_texts: list,
    model_name: str = "microsoft/deberta-v3-base",
    epochs: int = 20,
    batch_size: int = 256,
    learning_rate: float = 5e-5,
    mlm_probability: float = 0.3,
    max_seq_length: int = 32,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.06,
    mixed_precision: str = "bf16",
    seed: int = 42,
    push_to_hub: bool = False,
    hf_repo: str = None,
    hf_token: str = None,
) -> dict:
    """Continued MLM pretraining. Runs on Modal; mirrors finetune.py's style."""
    import math

    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForMaskedLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if mixed_precision == "bf16" and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        print("WARNING: bf16 not supported on this GPU, falling back to fp16")
        mixed_precision = "fp16"

    print("=" * 60)
    print("Domain-Adaptive Pretraining (masked LM on point names)")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Corpus: {len(train_texts)} train / {len(eval_texts)} eval texts")
    print(f"Epochs: {epochs}  Batch: {batch_size}  LR: {learning_rate}  "
          f"Mask: {mlm_probability}  Seed: {seed}")
    print("=" * 60)

    # DeBERTa-v3 was pretrained with replaced-token detection; continued
    # pretraining attaches a fresh MLM head to the discriminator backbone
    # (standard practice -- the head is discarded at fine-tune time anyway)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForMaskedLM.from_pretrained(model_name, token=hf_token)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_seq_length)

    train_ds = Dataset.from_dict({"text": train_texts}).map(
        tokenize, batched=True, remove_columns=["text"])
    eval_ds = Dataset.from_dict({"text": eval_texts}).map(
        tokenize, batched=True, remove_columns=["text"])

    # Elevated masking rate: at ~6 tokens per name, 15% would mask barely
    # one token per example
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability)

    output_dir = volume_path / "dapt_output"
    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        fp16=mixed_precision == "fp16" and torch.cuda.is_available(),
        bf16=mixed_precision == "bf16" and torch.cuda.is_available(),
        seed=seed,
        report_to="none",
        dataloader_num_workers=2,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                      eval_dataset=eval_ds, data_collator=collator)

    before = trainer.evaluate()
    ppl_before = math.exp(before["eval_loss"])
    print(f"Held-out MLM perplexity BEFORE: {ppl_before:.2f}")

    train_result = trainer.train()

    after = trainer.evaluate()
    ppl_after = math.exp(after["eval_loss"])
    print(f"Held-out MLM perplexity AFTER:  {ppl_after:.2f} "
          f"(was {ppl_before:.2f})")

    hf_url = None
    final_path = volume_path / "dapt_model"
    final_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    if push_to_hub and hf_repo:
        if not hf_token:
            print("WARNING: HF_TOKEN not provided. Skipping push to Hub.")
        else:
            from huggingface_hub import HfApi, login
            login(token=hf_token)
            api = HfApi()
            api.create_repo(repo_id=hf_repo, private=True, exist_ok=True,
                            repo_type="model")
            api.upload_folder(
                folder_path=str(final_path), repo_id=hf_repo, repo_type="model",
                commit_message=(f"DAPT {model_name}: {epochs}ep on "
                                f"{len(train_texts)} names, "
                                f"ppl {ppl_before:.2f}->{ppl_after:.2f}"))
            hf_url = f"https://huggingface.co/{hf_repo}"
            print(f"DAPT checkpoint pushed to: {hf_url}")
    volume.commit()

    return {
        "timestamp": datetime.now().isoformat(),
        "base_model": model_name,
        "corpus_train": len(train_texts),
        "corpus_eval": len(eval_texts),
        "config": {
            "epochs": epochs, "batch_size": batch_size,
            "learning_rate": learning_rate, "mlm_probability": mlm_probability,
            "max_seq_length": max_seq_length, "seed": seed,
        },
        "perplexity_before": ppl_before,
        "perplexity_after": ppl_after,
        "train_runtime_seconds": train_result.metrics.get("train_runtime"),
        "huggingface_url": hf_url,
    }


# Modal app definitions (guarded so the corpus/fetch paths work without modal)
try:
    import modal

    image = (
        modal.Image.debian_slim(python_version="3.10")
        .pip_install(
            "torch==2.5.1",
            "transformers==4.46.3",
            "datasets==3.1.0",
            "scikit-learn==1.5.2",
            "accelerate==1.1.1",
            "huggingface_hub==0.26.2",
            "sentencepiece==0.2.0",
        )
    )
    volume = modal.Volume.from_name("deberta-finetune-vol", create_if_missing=True)
    volume_path = Path("/root") / "data"
    app = modal.App("deberta-dapt", image=image, volumes={volume_path: volume})

    @app.function(gpu="T4", timeout=TRAIN_TIMEOUT)
    def dapt_t4(**kwargs) -> dict:
        return _dapt_impl(**kwargs)

    @app.function(gpu="L4", timeout=TRAIN_TIMEOUT)
    def dapt_l4(**kwargs) -> dict:
        return _dapt_impl(**kwargs)

    @app.function(gpu="A100", timeout=TRAIN_TIMEOUT)
    def dapt_a100(**kwargs) -> dict:
        return _dapt_impl(**kwargs)

    @app.function(timeout=15 * 60)
    def push_saved(hf_repo: str, hf_token: str) -> str:
        """Push the last trained DAPT checkpoint from the Modal volume.

        Rescues runs that trained successfully but skipped the Hub push
        (e.g. missing token) -- no retraining needed.
        """
        from huggingface_hub import HfApi, login

        final_path = volume_path / "dapt_model"
        if not (final_path / "config.json").exists():
            raise RuntimeError("No saved DAPT model on the volume -- run training first")
        login(token=hf_token)
        api = HfApi()
        api.create_repo(repo_id=hf_repo, private=True, exist_ok=True, repo_type="model")
        api.upload_folder(folder_path=str(final_path), repo_id=hf_repo,
                          repo_type="model",
                          commit_message="DAPT checkpoint (pushed from Modal volume)")
        return f"https://huggingface.co/{hf_repo}"

    GPU_FUNCTIONS = {"T4": dapt_t4, "L4": dapt_l4, "A100": dapt_a100}

    @app.local_entrypoint()
    def main(
        model: str = "microsoft/deberta-v3-base",
        gpu: str = "L4",
        epochs: int = 20,
        batch_size: int = 256,
        learning_rate: float = 5e-5,
        mlm_probability: float = 0.3,
        seed: int = 42,
        extra: str = "",
        push_to_hub: bool = False,
        push_only: bool = False,
        hf_repo: str = DEFAULT_HF_REPO,
    ):
        """Build the corpus locally, pretrain on Modal, save a results file.

        --push-only skips training and pushes the checkpoint already saved
        on the Modal volume by a previous run.
        """
        # Token from the shell env or repo .env (same sources as the other scripts)
        try:
            from dotenv import load_dotenv
            load_dotenv(Path(__file__).parent.parent / ".env")
        except ImportError:
            pass
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_TOKEN")

        if push_only:
            if not hf_token:
                print("ERROR: --push-only needs HF_TOKEN (env or .env)")
                return
            url = push_saved.remote(hf_repo=hf_repo, hf_token=hf_token)
            print(f"DAPT checkpoint pushed to: {url}")
            print(f"Next: set `model: \"{hf_repo}\"` in config/training.yml and push.")
            return

        if push_to_hub and not hf_token:
            print("WARNING: --push-to-hub set but no HF_TOKEN in env or .env -- "
                  "training will complete and save to the volume; recover later "
                  "with `modal run scripts/dapt.py --push-only`")

        train_texts, eval_texts = build_corpus(
            extra_paths=[p for p in extra.split(",") if p], seed=seed)

        gpu_upper = gpu.upper()
        if gpu_upper not in GPU_FUNCTIONS:
            print(f"WARNING: unknown GPU '{gpu}', using L4")
            gpu_upper = "L4"

        results = GPU_FUNCTIONS[gpu_upper].remote(
            train_texts=train_texts,
            eval_texts=eval_texts,
            model_name=model,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            mlm_probability=mlm_probability,
            seed=seed,
            push_to_hub=push_to_hub,
            hf_repo=hf_repo,
            hf_token=hf_token,
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = f"output/{timestamp}_dapt.txt"
        os.makedirs("output", exist_ok=True)
        with open(out, "w") as f:
            f.write("DAPT Results\n" + "=" * 60 + "\n")
            f.write(json.dumps(results, indent=2, default=str))
        print(f"\nResults saved to: {out}")
        print(f"Perplexity: {results['perplexity_before']:.2f} -> "
              f"{results['perplexity_after']:.2f}")
        if results.get("huggingface_url"):
            print(f"\nNext: set `model: \"{hf_repo}\"` in config/training.yml "
                  "and push -- the fine-tune quality gate decides if DAPT ships.")

except ImportError:
    modal = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DAPT corpus tools (training runs via `modal run scripts/dapt.py`)")
    parser.add_argument("--fetch-public", action="store_true",
                        help=f"Refresh {PUBLIC_POINTS_PATH} from public research datasets")
    parser.add_argument("--corpus-only", action="store_true",
                        help="Build the corpus, print stats, write data/dapt_corpus.txt")
    parser.add_argument("--extra", default="", help="Comma-separated extra name files")
    parser.add_argument("--seed", type=int, default=42)
    cli = parser.parse_args()

    if cli.fetch_public:
        fetch_public()
    if cli.corpus_only:
        train_texts, eval_texts = build_corpus(
            extra_paths=[p for p in cli.extra.split(",") if p], seed=cli.seed)
        with open("data/dapt_corpus.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(train_texts + eval_texts) + "\n")
        print("Corpus written to data/dapt_corpus.txt for inspection")
    if not (cli.fetch_public or cli.corpus_only):
        parser.print_help()
