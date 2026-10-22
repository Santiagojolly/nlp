from src.text_preprocessor import normalize_text, preprocess_text, tokenize_words


def test_normalization_lowercases_removes_punctuation_and_collapses_space():
    assert normalize_text("  HELLO,   World!  ") == "hello world"


def test_preprocessing_exposes_sentences_stop_words_and_stems():
    result = preprocess_text("The studies were running quickly. Another sentence follows.")

    assert result.original.startswith("The studies")
    assert result.sentences == (
        "The studies were running quickly.",
        "Another sentence follows.",
    )
    assert "the" in result.tokens
    assert "the" not in result.filtered_tokens
    assert "studi" in result.stemmed_tokens
    assert result.token_spans[0].start == 0


def test_word_tokenization_handles_apostrophes_and_empty_input():
    assert tokenize_words("Students' work isn't empty.") == ("students", "work", "isn't", "empty")
    assert preprocess_text("").tokens == ()