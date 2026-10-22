"""Offline, reusable text normalization and tokenization helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


WORD_PATTERN = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)
SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")
STEMMER = PorterStemmer()


@dataclass(frozen=True)
class TokenSpan:
    """A normalized token and its character offsets in the original text."""

    value: str
    start: int
    end: int


@dataclass(frozen=True)
class PreprocessedText:
    """Original and normalized representations used by analysis and display."""

    original: str
    normalized: str
    tokens: tuple[str, ...]
    filtered_tokens: tuple[str, ...]
    stemmed_tokens: tuple[str, ...]
    sentences: tuple[str, ...]
    token_spans: tuple[TokenSpan, ...]


def normalize_text(text: str) -> str:
    """Lowercase text, replace punctuation with spaces, and collapse whitespace."""
    without_punctuation = re.sub(r"[^\w\s]", " ", text.casefold(), flags=re.UNICODE)
    return re.sub(r"\s+", " ", without_punctuation).strip()


def tokenize_words(text: str) -> tuple[str, ...]:
    """Tokenize words while retaining internal apostrophes."""
    return tuple(match.group(0).casefold() for match in WORD_PATTERN.finditer(text))


def preprocess_text(text: str, remove_stopwords: bool = True) -> PreprocessedText:
    """Build normalized, token, sentence, stop-word, and stemmed views."""
    original = text or ""
    spans = tuple(
        TokenSpan(match.group(0).casefold(), match.start(), match.end())
        for match in WORD_PATTERN.finditer(original)
    )
    tokens = tuple(span.value for span in spans)
    filtered = (
        tuple(token for token in tokens if token not in ENGLISH_STOP_WORDS)
        if remove_stopwords
        else tokens
    )
    sentences = tuple(
        sentence.strip()
        for sentence in SENTENCE_PATTERN.split(original.strip())
        if sentence.strip()
    )
    return PreprocessedText(
        original=original,
        normalized=normalize_text(original),
        tokens=tokens,
        filtered_tokens=filtered,
        stemmed_tokens=tuple(STEMMER.stem(token) for token in filtered),
        sentences=sentences,
        token_spans=spans,
    )