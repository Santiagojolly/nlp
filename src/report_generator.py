"""Create downloadable CSV and plain-text analysis reports."""

from __future__ import annotations

import csv
from io import StringIO
from typing import Any, Sequence


LIMITATIONS = (
    "This comparison only checks the documents supplied to this application. It is not a "
    "web or corpus search, may miss paraphrased material, and may flag common phrases. "
    "Extraction quality can also affect results. Similarity is an indicator, not a "
    "definitive plagiarism verdict or finding of academic misconduct."
)


def generate_csv_report(comparisons: Sequence[dict[str, Any]]) -> str:
    """Return one CSV row per document pair, including the component scores."""
    output = StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(
        [
            "Document A",
            "Document B",
            "Combined similarity (%)",
            "TF-IDF cosine (%)",
            "N-gram overlap (%)",
            "Jaccard (%)",
            "Sufficient text",
            "Exact matching phrases",
            "Method",
        ]
    )
    for comparison in comparisons:
        phrases = " | ".join(match["phrase"] for match in comparison.get("matches", []))
        writer.writerow(
            [
                comparison["left_name"],
                comparison["right_name"],
                round(comparison["combined_score"] * 100, 2),
                round(comparison["tfidf_cosine"] * 100, 2),
                round(comparison["ngram_overlap"] * 100, 2),
                round(comparison["jaccard"] * 100, 2),
                "yes" if comparison["sufficient_text"] else "no",
                phrases,
                "Unweighted mean of TF-IDF cosine, n-gram overlap, and word-set Jaccard",
            ]
        )
    return output.getvalue()


def generate_text_report(comparisons: Sequence[dict[str, Any]]) -> str:
    """Return a readable report with pair scores, exact passages, and limitations."""
    lines = [
        "DOCUMENT SIMILARITY ANALYSIS",
        "=" * 29,
        "",
        "Scores are similarity indicators, not plagiarism verdicts.",
        "Combined score: arithmetic mean of TF-IDF cosine, n-gram overlap, and Jaccard.",
        "",
    ]
    if not comparisons:
        lines.append("No document pairs were analyzed.")
    for comparison in comparisons:
        lines.extend(
            [
                f"{comparison['left_name']}  <->  {comparison['right_name']}",
                f"  Combined similarity: {comparison['combined_score']:.1%}",
                f"  TF-IDF cosine:       {comparison['tfidf_cosine']:.1%}",
                f"  N-gram overlap:      {comparison['ngram_overlap']:.1%}",
                f"  Word-set Jaccard:    {comparison['jaccard']:.1%}",
                f"  Sufficient text:     {'yes' if comparison['sufficient_text'] else 'no'}",
            ]
        )
        matches = comparison.get("matches", [])
        lines.append("  Exact matching passages:")
        if not matches:
            lines.append("    None found at the configured minimum phrase length.")
        for match in matches:
            lines.extend(
                [
                    f"    {match['word_count']} words; approximately sentence "
                    f"{match['left_sentence']} / {match['right_sentence']}:",
                    f"      {match['left_name']}: {match['left_snippet']}",
                    f"      {match['right_name']}: {match['right_snippet']}",
                ]
            )
        lines.append("")
    lines.extend(["Limitations", "-----------", LIMITATIONS, ""])
    return "\n".join(lines)