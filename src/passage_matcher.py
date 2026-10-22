"""Find exact shared word sequences and map them back to original text."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.text_preprocessor import preprocess_text


def _approximate_sentence_number(text: str, character_offset: int) -> int:
    return max(1, sum(text[:character_offset].count(mark) for mark in ".!?") + 1)


def find_matching_passages(
    left_name: str,
    left_text: str,
    right_name: str,
    right_text: str,
    min_words: int = 5,
    max_matches: int = 40,
) -> list[dict[str, Any]]:
    """Return maximal exact token runs of at least ``min_words`` on both sides."""
    min_words = max(1, int(min_words))
    left = preprocess_text(left_text, remove_stopwords=False)
    right = preprocess_text(right_text, remove_stopwords=False)
    left_tokens = left.token_spans
    right_tokens = right.token_spans
    if len(left_tokens) < min_words or len(right_tokens) < min_words:
        return []

    right_positions: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for index in range(len(right_tokens) - min_words + 1):
        gram = tuple(token.value for token in right_tokens[index : index + min_words])
        right_positions[gram].append(index)

    found: dict[tuple[int, int, int, int], int] = {}
    for left_index in range(len(left_tokens) - min_words + 1):
        gram = tuple(token.value for token in left_tokens[left_index : left_index + min_words])
        for right_index in right_positions.get(gram, []):
            if (
                left_index > 0
                and right_index > 0
                and left_tokens[left_index - 1].value == right_tokens[right_index - 1].value
            ):
                continue
            length = min_words
            while (
                left_index + length < len(left_tokens)
                and right_index + length < len(right_tokens)
                and left_tokens[left_index + length].value
                == right_tokens[right_index + length].value
            ):
                length += 1
            key = (left_index, left_index + length, right_index, right_index + length)
            found[key] = length
            if len(found) >= max_matches * 3:
                break
        if len(found) >= max_matches * 3:
            break

    ordered = sorted(found.items(), key=lambda item: (-item[1], item[0]))[:max_matches]
    matches: list[dict[str, Any]] = []
    for (left_start_index, left_end_index, right_start_index, right_end_index), length in ordered:
        left_start = left_tokens[left_start_index].start
        left_end = left_tokens[left_end_index - 1].end
        right_start = right_tokens[right_start_index].start
        right_end = right_tokens[right_end_index - 1].end
        matches.append(
            {
                "left_name": left_name,
                "right_name": right_name,
                "phrase": " ".join(token.value for token in left_tokens[left_start_index:left_end_index]),
                "left_snippet": left_text[left_start:left_end],
                "right_snippet": right_text[right_start:right_end],
                "left_start": left_start,
                "left_end": left_end,
                "right_start": right_start,
                "right_end": right_end,
                "left_sentence": _approximate_sentence_number(left_text, left_start),
                "right_sentence": _approximate_sentence_number(right_text, right_start),
                "word_count": length,
                "match_type": "Exact normalized word sequence",
            }
        )
    return matches