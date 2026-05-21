# 🤖 Transformer vs Traditional NLP — Benchmarking Study

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-FFD21F)](https://huggingface.co/transformers)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A rigorous empirical comparison of **traditional ML models** (Logistic Regression, Naive Bayes) against **transformer-based architectures** (BERT) for text classification — covering accuracy, F1, computational cost, and practical trade-offs.

---

## 📌 Project Overview

This project provides an end-to-end benchmarking framework to evaluate NLP models across multiple dimensions:

| Model | Type | Accuracy | F1-Score | Train Time |
|---|---|---|---|---|
| Naive Bayes (TF-IDF) | Traditional | 83.2% | 0.831 | ~2s |
| Logistic Regression (TF-IDF) | Traditional | 87.6% | 0.874 | ~5s |
| **BERT (fine-tuned)** | **Transformer** | **93.4%** | **0.933** | ~18 min |

> ✅ **BERT achieves ~6.8% higher accuracy** over the best traditional baseline, with state-of-the-art F1.

---

## 🗂️ Project Structure

```
nlp-benchmark/
├── src/
│   ├── preprocess.py          # Text cleaning & tokenization
│   ├── traditional_models.py  # LR & Naive Bayes pipelines
│   ├── bert_model.py          # BERT fine-tuning pipeline
│   ├── evaluate.py            # Metrics & comparison utilities
│   └── visualize.py           # Result plots & charts
├── notebooks/
│   └── full_benchmark.ipynb   # End-to-end walkthrough notebook
├── data/
│   └── README.md              # Dataset instructions
├── results/
│   └── metrics_summary.json   # Saved benchmark results
├── tests/
│   └── test_pipeline.py       # Unit tests
├── requirements.txt
├── setup.py
└── README.md
```

---

## ⚙️ Setup & Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/nlp-benchmark.git
cd nlp-benchmark

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### Run Traditional Models
```bash
python src/traditional_models.py --dataset imdb --max_features 50000
```

### Fine-tune BERT
```bash
python src/bert_model.py --dataset imdb --epochs 3 --batch_size 16
```

### Full Benchmark Comparison
```bash
python src/evaluate.py --compare all --output results/
```

---

## 📊 Methodology

### Dataset
- **IMDb Movie Reviews** (50,000 samples) — binary sentiment classification
- 80/10/10 train/validation/test split, stratified

### Traditional Pipeline
1. **Preprocessing** — lowercasing, punctuation removal, stopword filtering, lemmatization
2. **Vectorization** — TF-IDF with unigrams + bigrams (max 50k features)
3. **Models** — Multinomial Naive Bayes, Logistic Regression (L2 regularization)
4. **Tuning** — GridSearchCV with 5-fold cross-validation

### BERT Pipeline
1. **Tokenization** — `bert-base-uncased` WordPiece tokenizer (max length 128)
2. **Architecture** — `BertForSequenceClassification` with classification head
3. **Fine-tuning** — AdamW optimizer, linear warmup schedule, 3 epochs
4. **Hardware** — NVIDIA GPU (CUDA) with mixed-precision training

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score (macro & weighted)
- Confusion Matrix
- ROC-AUC Curve
- Training time & inference latency

---

## 📈 Key Findings

- **BERT outperforms** traditional models by **~6.8% accuracy** on the IMDb benchmark
- **Logistic Regression** is a strong baseline — within ~6% of BERT at 1/200th the compute
- **Naive Bayes** trains in seconds and is surprisingly competitive for simpler datasets
- **BERT inference** is ~40× slower per sample than LR, making it costly in production
- For resource-constrained environments, **LR + TF-IDF** offers the best accuracy/cost ratio

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 📚 References

- Devlin et al. (2019) — [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [HuggingFace Transformers](https://huggingface.co/transformers)
- Maas et al. (2011) — IMDb Large Movie Review Dataset

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

*Built as part of an NLP research study comparing classical and modern text classification approaches.*
