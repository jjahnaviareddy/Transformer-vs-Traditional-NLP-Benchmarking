"""
evaluate.py
-----------
Aggregates results from traditional and BERT models,
prints comparison tables, and saves a unified metrics summary.
"""

import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

METRICS = ["accuracy", "precision", "recall", "f1_score",
           "train_time_sec", "infer_ms_per_sample"]

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision (W)",
    "recall": "Recall (W)",
    "f1_score": "F1-Score (W)",
    "train_time_sec": "Train Time (s)",
    "infer_ms_per_sample": "Infer (ms/sample)",
}


def load_results(results_dir: Path) -> List[Dict[str, Any]]:
    """Load all *_results.json files from results directory."""
    all_results = []
    for path in sorted(results_dir.glob("*results.json")):
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            all_results.extend(data)
        else:
            all_results.append(data)
    return all_results


def print_comparison_table(results: List[Dict[str, Any]]) -> None:
    """Pretty-print a comparison table to stdout."""
    col_w = 26
    metric_w = 18

    header = f"{'Model':<{col_w}}" + "".join(
        f"{METRIC_LABELS[m]:>{metric_w}}" for m in METRICS
    )
    sep = "─" * len(header)

    print(f"\n{'NLP MODEL BENCHMARK COMPARISON':^{len(header)}}")
    print(sep)
    print(header)
    print(sep)

    for r in results:
        row = f"{r['model']:<{col_w}}"
        for m in METRICS:
            val = r.get(m, "N/A")
            if isinstance(val, float):
                if m in ("accuracy", "precision", "recall", "f1_score"):
                    row += f"{val:>{metric_w}.4f}"
                else:
                    row += f"{val:>{metric_w}.2f}"
            else:
                row += f"{str(val):>{metric_w}}"
        print(row)

    print(sep)

    # Highlight winner per metric
    print("\n🏆  Winners per metric:")
    for m in ["accuracy", "f1_score"]:
        best = max(results, key=lambda r: r.get(m, 0))
        print(f"   {METRIC_LABELS[m]:<20}: {best['model']}  ({best[m]:.4f})")
    for m in ["train_time_sec", "infer_ms_per_sample"]:
        best = min(results, key=lambda r: r.get(m, float("inf")))
        print(f"   {METRIC_LABELS[m]:<20}: {best['model']}  ({best[m]:.2f})")


def compute_improvement(results: List[Dict[str, Any]]) -> None:
    """Print relative improvement of BERT over best traditional model."""
    traditional = [r for r in results if "BERT" not in r["model"]]
    bert = next((r for r in results if "BERT" in r["model"]), None)

    if not traditional or bert is None:
        return

    best_trad = max(traditional, key=lambda r: r.get("f1_score", 0))
    delta_acc = bert["accuracy"] - best_trad["accuracy"]
    delta_f1 = bert["f1_score"] - best_trad["f1_score"]

    print(f"\n📊  BERT vs Best Traditional ({best_trad['model']}):")
    print(f"   Accuracy improvement : {delta_acc:+.4f} ({delta_acc*100:+.2f}%)")
    print(f"   F1 improvement       : {delta_f1:+.4f} ({delta_f1*100:+.2f}%)")
    cost_ratio = bert["train_time_sec"] / max(best_trad["train_time_sec"], 0.01)
    print(f"   Train time ratio     : {cost_ratio:.1f}× slower\n")


def save_summary(results: List[Dict[str, Any]], output_dir: Path) -> None:
    out = {
        "models": results,
        "summary": {
            "best_accuracy": max(results, key=lambda r: r.get("accuracy", 0))["model"],
            "best_f1": max(results, key=lambda r: r.get("f1_score", 0))["model"],
            "fastest_train": min(results, key=lambda r: r.get("train_time_sec", float("inf")))["model"],
            "fastest_infer": min(results, key=lambda r: r.get("infer_ms_per_sample", float("inf")))["model"],
        }
    }
    path = output_dir / "metrics_summary.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"Summary saved to {path}")


def main(args):
    results_dir = Path(args.results_dir)
    results = load_results(results_dir)

    if not results:
        logger.error("No result files found. Run traditional_models.py and bert_model.py first.")
        return

    print_comparison_table(results)
    compute_improvement(results)
    save_summary(results, results_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare NLP model results")
    parser.add_argument("--results_dir", default="../results", help="Directory with *_results.json files")
    args = parser.parse_args()
    main(args)
