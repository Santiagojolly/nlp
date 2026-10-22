from src.passage_matcher import find_matching_passages
from src.similarity_engine import compare_documents


def test_identical_documents_score_one_and_are_compared_once():
    documents = [
        {"name": "A", "text": "A careful comparison of several research methods improves the final analysis."},
        {"name": "B", "text": "A careful comparison of several research methods improves the final analysis."},
    ]

    results = compare_documents(documents)

    assert len(results) == 1
    assert results[0]["combined_score"] == 1.0
    assert results[0]["tfidf_cosine"] == 1.0
    assert results[0]["ngram_overlap"] == 1.0
    assert results[0]["jaccard"] == 1.0


def test_disjoint_and_short_documents_are_handled_without_errors():
    results = compare_documents(
        [{"name": "A", "text": "apple"}, {"name": "B", "text": "ocean"}]
    )

    assert len(results) == 1
    assert results[0]["combined_score"] == 0.0
    assert not results[0]["sufficient_text"]
    assert compare_documents([{"name": "only", "text": "one"}]) == []


def test_partial_overlap_has_shared_ngram_and_highlight_offsets():
    left = "Fresh water supports healthy communities around the world."
    right = "Researchers found that fresh water supports healthy communities in cities."

    comparison = compare_documents(
        [{"name": "Left", "text": left}, {"name": "Right", "text": right}], ngram_size=3
    )[0]
    matches = find_matching_passages("Left", left, "Right", right, min_words=4)

    assert 0 < comparison["combined_score"] < 1
    assert comparison["shared_ngrams"]
    assert matches
    assert matches[0]["left_snippet"].casefold() == "fresh water supports healthy communities"
    assert left[matches[0]["left_start"] : matches[0]["left_end"]] == matches[0]["left_snippet"]


def test_empty_documents_return_zero_scores_and_no_exact_matches():
    results = compare_documents([{"name": "A", "text": ""}, {"name": "B", "text": ""}])

    assert results[0]["combined_score"] == 0.0
    assert not results[0]["sufficient_text"]
    assert find_matching_passages("A", "", "B", "") == []