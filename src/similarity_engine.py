"""Pairwise document similarity using three deterministic text measures."""

from __future__ import annotations

from itertools import combinations
from typing import Any, Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.text_preprocessor import normalize_text, tokenize_words


def _word_ngrams(tokens: Sequence[str], size: int) -> set[tuple[str, ...]]:
    return {tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1)}


def compare_documents(
    documents: Sequence[dict[str, str]], ngram_size: int = 3
) -> list[dict[str, Any]]:
    """Compare each unique document pair and return component and combined scores.

    Combined similarity is the unweighted mean of TF-IDF cosine, unique n-gram
    overlap (intersection divided by the larger set), and word-set Jaccard.
    """
    if len(documents) < 2:
        return []
    ngram_size = max(1, int(ngram_size))
    texts = [document.get("text", "") or "" for document in documents]
    tokens = [tokenize_words(text) for text in texts]
    normalized = [normalize_text(text) for text in texts]

    try:
        vectorizer = TfidfVectorizer(
            lowercase=False,
            stop_words="english",
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b\w\w+\b",
        )
        vectors = vectorizer.fit_transform(normalized)
    except ValueError:
        vectors = None

    shingles = [_word_ngrams(document_tokens, ngram_size) for document_tokens in tokens]
    word_sets = [set(document_tokens) for document_tokens in tokens]
    results: list[dict[str, Any]] = []

    for left_index, right_index in combinations(range(len(documents)), 2):
        if vectors is None:
            tfidf_score = 0.0
        else:
            tfidf_score = float(
                cosine_similarity(vectors[left_index], vectors[right_index])[0, 0]
            )
            tfidf_score = min(1.0, max(0.0, tfidf_score))

        shared_ngrams = shingles[left_index] & shingles[right_index]
        largest_ngram_set = max(len(shingles[left_index]), len(shingles[right_index]))
        ngram_score = len(shared_ngrams) / largest_ngram_set if largest_ngram_set else 0.0

        shared_words = word_sets[left_index] & word_sets[right_index]
        all_words = word_sets[left_index] | word_sets[right_index]
        jaccard_score = len(shared_words) / len(all_words) if all_words else 0.0

        component_scores = (tfidf_score, ngram_score, jaccard_score)
        results.append(
            {
                "left_name": documents[left_index].get("name", f"Document {left_index + 1}"),
                "right_name": documents[right_index].get("name", f"Document {right_index + 1}"),
                "left_index": left_index,
                "right_index": right_index,
                "tfidf_cosine": tfidf_score,
                "ngram_overlap": ngram_score,
                "jaccard": jaccard_score,
                "combined_score": sum(component_scores) / len(component_scores),
                "shared_ngrams": [" ".join(gram) for gram in sorted(shared_ngrams)],
                "left_word_count": len(tokens[left_index]),
                "right_word_count": len(tokens[right_index]),
                "sufficient_text": min(
                    len(tokens[left_index]), len(tokens[right_index])
                ) >= 3,
            }
        )
    return results