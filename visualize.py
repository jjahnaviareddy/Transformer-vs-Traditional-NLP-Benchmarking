"""
visualize.py
------------
Generate comparison charts and plots from benchmark results.
Produces publication-quality figures saved to results/figures/.
"""

import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Styling ──────────────────────────────────────────────────────────────────
PALETTE = ["#4C72B0", "#55A868", "#C44E52"]   # blue / green / red
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


def _load_results(results_dir: Path) -> List[Dict[str, Any]]:
    summary_path = results_dir / "metrics_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            return json.load(f)["models"]

    # Fallback: load individual files
    results = []
    for path in sorted(results_dir.glob("*results.json")):
        with open(path) as f:
            data = json.load(f)
        results.extend(data if isinstance(data, list) else [data])
    return results


# ── Plot 1: Metric Bar Chart ──────────────────────────────────────────────────

def plot_metric_comparison(results: List[Dict], out_dir: Path) -> None:
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
    models = [r["model"] for r in results]
    x = np.arange(len(metrics))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (r, color) in enumerate(zip(results, PALETTE)):
        vals = [r.get(m, 0) for m in metrics]
        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=r["model"], color=color, alpha=0.88)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.70, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison", fontweight="bold", pad=14)
    ax.legend(loc="lower right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    path = out_dir / "metric_comparison.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {path}")


# ── Plot 2: Train Time vs Accuracy Scatter ────────────────────────────────────

def plot_accuracy_vs_time(results: List[Dict], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))

    for r, color in zip(results, PALETTE):
        ax.scatter(
            r.get("train_time_sec", 0),
            r.get("accuracy", 0),
            color=color, s=180, zorder=5, edgecolors="white", linewidth=1.5,
            label=r["model"],
        )
        ax.annotate(
            r["model"].split("(")[0].strip(),
            xy=(r.get("train_time_sec", 0), r.get("accuracy", 0)),
            xytext=(8, 4), textcoords="offset points", fontsize=9,
        )

    ax.set_xlabel("Training Time (seconds, log scale)")
    ax.set_ylabel("Test Accuracy")
    ax.set_xscale("log")
    ax.set_title("Accuracy vs. Training Cost", fontweight="bold", pad=14)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend()

    path = out_dir / "accuracy_vs_time.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {path}")


# ── Plot 3: Confusion Matrices ────────────────────────────────────────────────

def plot_confusion_matrices(results: List[Dict], out_dir: Path) -> None:
    models_with_cm = [r for r in results if "confusion_matrix" in r]
    if not models_with_cm:
        logger.warning("No confusion matrices found in results.")
        return

    n = len(models_with_cm)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    class_names = ["Negative", "Positive"]
    for ax, r in zip(axes, models_with_cm):
        cm = np.array(r["confusion_matrix"])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        sns.heatmap(
            cm_norm, annot=True, fmt=".2%", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            ax=ax, cbar=False, linewidths=0.5,
        )
        ax.set_title(r["model"], fontweight="bold", fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    fig.suptitle("Confusion Matrices (Normalized)", fontweight="bold", y=1.02)
    path = out_dir / "confusion_matrices.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {path}")


# ── Plot 4: Inference Latency ─────────────────────────────────────────────────

def plot_inference_latency(results: List[Dict], out_dir: Path) -> None:
    models = [r["model"].split("(")[0].strip() for r in results]
    latencies = [r.get("infer_ms_per_sample", 0) for r in results]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(models, latencies, color=PALETTE[:len(models)], alpha=0.88)
    for bar, val in zip(bars, latencies):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f} ms", va="center", fontsize=9)

    ax.set_xlabel("Inference Latency (ms per sample)")
    ax.set_title("Inference Speed Comparison", fontweight="bold", pad=14)
    ax.xaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    path = out_dir / "inference_latency.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    results_dir = Path(args.results_dir)
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    results = _load_results(results_dir)
    if not results:
        logger.error("No results found. Run evaluate.py first.")
        return

    plot_metric_comparison(results, figures_dir)
    plot_accuracy_vs_time(results, figures_dir)
    plot_confusion_matrices(results, figures_dir)
    plot_inference_latency(results, figures_dir)

    logger.info(f"\n✅ All plots saved to {figures_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize NLP benchmark results")
    parser.add_argument("--results_dir", default="../results")
    args = parser.parse_args()
    main(args)
