"""
test_pipeline.py
----------------
Unit tests for preprocessing, model pipelines, and evaluation utilities.
Run with:  pytest tests/ -v
"""

import sys
import json
import pytest
from pathlib import Path

# Make src importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocess import TextPreprocessor, get_default_preprocessor


# ─────────────────────────────────────────────────────────────────────────────
# TextPreprocessor tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTextPreprocessor:

    def test_lowercase(self):
        p = TextPreprocessor(lowercase=True, remove_stopwords=False,
                             remove_punctuation=False, lemmatize=False)
        assert p.clean("Hello World") == "hello world"

    def test_remove_html(self):
        p = TextPreprocessor(remove_html=True, remove_stopwords=False,
                             remove_punctuation=False, lemmatize=False)
        assert "<br/>" not in p.clean("Hello <br/> World")

    def test_remove_urls(self):
        p = TextPreprocessor(remove_urls=True, remove_stopwords=False,
                             remove_punctuation=False, lemmatize=False)
        result = p.clean("Visit https://example.com for more.")
        assert "https" not in result

    def test_remove_punctuation(self):
        p = TextPreprocessor(remove_punctuation=True, remove_stopwords=False, lemmatize=False)
        result = p.clean("Hello, world!")
        assert "," not in result and "!" not in result

    def test_tokenize_returns_list(self):
        p = get_default_preprocessor()
        tokens = p.tokenize("The quick brown fox jumps over the lazy dog")
        assert isinstance(tokens, list)
        assert len(tokens) > 0

    def test_stopword_removal(self):
        p = TextPreprocessor(remove_stopwords=True, lemmatize=False, remove_punctuation=False)
        tokens = p.tokenize("this is a test sentence")
        stop_words = {"this", "is", "a"}
        for sw in stop_words:
            assert sw not in tokens, f"Stopword '{sw}' should be removed"

    def test_process_returns_string(self):
        p = get_default_preprocessor()
        result = p.process("Great movie, really enjoyed it!")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_process_batch(self):
        p = get_default_preprocessor()
        texts = ["Great film!", "Terrible waste of time.", "Average at best."]
        results = p.process_batch(texts)
        assert len(results) == len(texts)
        assert all(isinstance(r, str) for r in results)

    def test_empty_string(self):
        p = get_default_preprocessor()
        result = p.process("")
        assert result == ""

    def test_non_string_input(self):
        p = get_default_preprocessor()
        assert p.clean(None) == ""
        assert p.clean(123) == ""

    def test_min_token_length(self):
        p = TextPreprocessor(min_token_length=3, remove_stopwords=False, lemmatize=False)
        tokens = p.tokenize("a bb ccc dddd")
        for t in tokens:
            assert len(t) >= 3, f"Token '{t}' is shorter than min_token_length=3"

    def test_lemmatization(self):
        p = TextPreprocessor(lemmatize=True, remove_stopwords=False,
                             remove_punctuation=False)
        tokens = p.tokenize("running runs runner")
        # All forms should reduce to 'run' (or similar)
        assert any("run" in t for t in tokens)

    def test_html_and_url_combined(self):
        p = get_default_preprocessor()
        text = '<a href="https://example.com">Click here</a>'
        result = p.process(text)
        assert "http" not in result
        assert "<" not in result


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate utilities tests
# ─────────────────────────────────────────────────────────────────────────────

from evaluate import print_comparison_table, compute_improvement

MOCK_RESULTS = [
    {
        "model": "Naive Bayes (TF-IDF)",
        "accuracy": 0.832,
        "precision": 0.834,
        "recall": 0.832,
        "f1_score": 0.831,
        "train_time_sec": 2.1,
        "infer_ms_per_sample": 0.05,
        "confusion_matrix": [[2100, 300], [250, 2350]],
    },
    {
        "model": "Logistic Regression (TF-IDF)",
        "accuracy": 0.876,
        "precision": 0.878,
        "recall": 0.876,
        "f1_score": 0.874,
        "train_time_sec": 5.3,
        "infer_ms_per_sample": 0.02,
        "confusion_matrix": [[2200, 200], [220, 2380]],
    },
    {
        "model": "BERT (fine-tuned)",
        "accuracy": 0.934,
        "precision": 0.935,
        "recall": 0.934,
        "f1_score": 0.933,
        "train_time_sec": 1080.0,
        "infer_ms_per_sample": 8.2,
        "confusion_matrix": [[2350, 50], [130, 2470]],
    },
]


class TestEvaluate:

    def test_print_comparison_table_runs(self, capsys):
        print_comparison_table(MOCK_RESULTS)
        captured = capsys.readouterr()
        assert "BERT" in captured.out
        assert "Logistic Regression" in captured.out

    def test_compute_improvement_runs(self, capsys):
        compute_improvement(MOCK_RESULTS)
        captured = capsys.readouterr()
        assert "BERT" in captured.out

    def test_bert_has_highest_accuracy(self):
        best = max(MOCK_RESULTS, key=lambda r: r["accuracy"])
        assert "BERT" in best["model"]

    def test_nb_has_fastest_train(self):
        fastest = min(MOCK_RESULTS, key=lambda r: r["train_time_sec"])
        assert "Naive Bayes" in fastest["model"]

    def test_lr_has_fastest_infer(self):
        fastest = min(MOCK_RESULTS, key=lambda r: r["infer_ms_per_sample"])
        assert "Logistic Regression" in fastest["model"]
