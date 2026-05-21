"""
bert_model.py
-------------
BERT fine-tuning pipeline for text classification using HuggingFace Transformers.
Supports GPU (CUDA / MPS) and CPU training with mixed-precision.
"""

import time
import logging
import argparse
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "bert-base-uncased"
LABEL_NAMES = ["Negative", "Positive"]


# -----------------------------------------------------------------------
# Device detection
# -----------------------------------------------------------------------

def get_device() -> str:
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logger.info(f"Using device: {device}")
    return device


# -----------------------------------------------------------------------
# Dataset preparation
# -----------------------------------------------------------------------

def load_and_tokenize(
    tokenizer,
    max_length: int = 128,
    max_train: int = 25000,
    max_test: int = 5000,
):
    """Load IMDb and tokenize for BERT."""
    logger.info("Loading IMDb dataset …")
    dataset = load_dataset("imdb")

    # Optionally subsample
    if max_train < len(dataset["train"]):
        dataset["train"] = dataset["train"].select(range(max_train))
    if max_test < len(dataset["test"]):
        dataset["test"] = dataset["test"].select(range(max_test))

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,  # dynamic padding via DataCollator
        )

    logger.info("Tokenizing …")
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch")
    return tokenized


# -----------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------

def compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
    }


# -----------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------

def train_bert(
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5,
    max_length: int = 128,
    max_train: int = 25000,
    max_test: int = 5000,
    output_dir: str = "../results/bert",
    warmup_ratio: float = 0.06,
    weight_decay: float = 0.01,
) -> Dict[str, Any]:

    device = get_device()
    fp16 = device == "cuda"  # mixed precision only on CUDA

    logger.info(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    tokenized_dataset = load_and_tokenize(
        tokenizer, max_length=max_length,
        max_train=max_train, max_test=max_test,
    )

    logger.info(f"Loading model: {MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={0: "Negative", 1: "Positive"},
        label2id={"Negative": 0, "Positive": 1},
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        fp16=fp16,
        report_to="none",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    logger.info("\n🚀 Starting BERT fine-tuning …")
    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0

    # --- Evaluate on test set ---
    logger.info("Evaluating on test set …")
    t1 = time.time()
    pred_output = trainer.predict(tokenized_dataset["test"])
    infer_time = time.time() - t1

    preds = np.argmax(pred_output.predictions, axis=-1)
    labels = pred_output.label_ids

    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    cm = confusion_matrix(labels, preds).tolist()

    logger.info(f"\n{classification_report(labels, preds, target_names=LABEL_NAMES)}")

    n_test = len(tokenized_dataset["test"])
    results = {
        "model": "BERT (fine-tuned)",
        "base_model": MODEL_NAME,
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "train_time_sec": round(train_time, 2),
        "infer_time_sec": round(infer_time, 4),
        "infer_ms_per_sample": round((infer_time / n_test) * 1000, 4),
        "confusion_matrix": cm,
        "hyperparameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "max_length": max_length,
            "warmup_ratio": warmup_ratio,
            "weight_decay": weight_decay,
        },
    }

    logger.info(f"Accuracy      : {acc:.4f}")
    logger.info(f"F1 (weighted) : {f1:.4f}")
    logger.info(f"Train time    : {train_time / 60:.1f} min")
    logger.info(f"Infer latency : {results['infer_ms_per_sample']:.4f} ms/sample")

    # Save model
    model_path = Path(output_dir) / "best_model"
    trainer.save_model(str(model_path))
    logger.info(f"Model saved to {model_path}")

    return results


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main(args):
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = train_bert(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length,
        output_dir=str(output_dir / "bert_checkpoints"),
    )

    out_path = output_dir / "bert_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\n✅ Results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune BERT for text classification")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--output", default="../results", help="Output directory")
    args = parser.parse_args()
    main(args)
