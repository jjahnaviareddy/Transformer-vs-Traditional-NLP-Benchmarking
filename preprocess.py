"""
preprocess.py
-------------
Text cleaning and preprocessing utilities shared across all model pipelines.
"""

import re
import string
import logging
from typing import List, Optional

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK data on first use
def _download_nltk_resources():
    resources = ["punkt", "stopwords", "wordnet", "omw-1.4"]
    for r in resources:
        try:
            nltk.download(r, quiet=True)
        except Exception as e:
            logging.warning(f"Could not download NLTK resource '{r}': {e}")

_download_nltk_resources()

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    Configurable text preprocessor with options for:
    - Lowercasing
    - HTML tag removal
    - URL / mention / number removal
    - Punctuation stripping
    - Stopword removal
    - Lemmatization
    """

    def __init__(
        self,
        lowercase: bool = True,
        remove_html: bool = True,
        remove_urls: bool = True,
        remove_punctuation: bool = True,
        remove_stopwords: bool = True,
        lemmatize: bool = True,
        min_token_length: int = 2,
    ):
        self.lowercase = lowercase
        self.remove_html = remove_html
        self.remove_urls = remove_urls
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.min_token_length = min_token_length

        self._stop_words = set(stopwords.words("english")) if remove_stopwords else set()
        self._lemmatizer = WordNetLemmatizer() if lemmatize else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clean(self, text: str) -> str:
        """Return a cleaned string (no tokenization)."""
        if not isinstance(text, str):
            return ""

        if self.lowercase:
            text = text.lower()
        if self.remove_html:
            text = self._strip_html(text)
        if self.remove_urls:
            text = self._strip_urls(text)

        text = self._strip_numbers(text)

        if self.remove_punctuation:
            text = text.translate(str.maketrans("", "", string.punctuation))

        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, text: str) -> List[str]:
        """Clean → tokenize → filter → lemmatize."""
        text = self.clean(text)
        tokens = word_tokenize(text)

        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self._stop_words]

        tokens = [t for t in tokens if len(t) >= self.min_token_length]

        if self.lemmatize and self._lemmatizer:
            tokens = [self._lemmatizer.lemmatize(t) for t in tokens]

        return tokens

    def process(self, text: str) -> str:
        """Full pipeline: returns space-joined token string (for TF-IDF)."""
        return " ".join(self.tokenize(text))

    def process_batch(self, texts: List[str], verbose: bool = False) -> List[str]:
        """Process a list of texts."""
        from tqdm import tqdm
        iterator = tqdm(texts, desc="Preprocessing") if verbose else texts
        return [self.process(t) for t in iterator]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_html(text: str) -> str:
        return re.sub(r"<[^>]+>", " ", text)

    @staticmethod
    def _strip_urls(text: str) -> str:
        return re.sub(r"http\S+|www\.\S+", " ", text)

    @staticmethod
    def _strip_numbers(text: str) -> str:
        return re.sub(r"\d+", " ", text)


# -----------------------------------------------------------------------
# Convenience function
# -----------------------------------------------------------------------

def get_default_preprocessor() -> TextPreprocessor:
    """Return a preprocessor with the default settings used in benchmarks."""
    return TextPreprocessor(
        lowercase=True,
        remove_html=True,
        remove_urls=True,
        remove_punctuation=True,
        remove_stopwords=True,
        lemmatize=True,
        min_token_length=2,
    )
