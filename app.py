"""Streamlit interface for comparing uploaded and pasted documents."""

from __future__ import annotations

import hashlib
import html
from itertools import combinations
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.document_parser import DocumentParseError, extract_text
from src.passage_matcher import find_matching_passages
from src.report_generator import generate_csv_report, generate_text_report
from src.similarity_engine import compare_documents


st.set_page_config(page_title="Text Similarity Review", page_icon="▤", layout="wide")
st.markdown(
    """
    <style>
    :root { --ink: #172a27; --muted: #5e706b; --paper: #f5f7f3; --line: #d9e1db; --accent: #167663; }
    .stApp { background: var(--paper); color: var(--ink); }
    h1, h2, h3 { color: var(--ink); font-family: Georgia, 'Times New Roman', serif; }
    h1 { letter-spacing: 0; font-size: 2.45rem; }
    .intro { color: var(--muted); max-width: 850px; font-size: 1.02rem; }
    .metric-label { color: var(--muted); font-size: .88rem; }
    .passage { background: #fff; border: 1px solid var(--line); border-left: 4px solid var(--accent); padding: .75rem .9rem; border-radius: 4px; line-height: 1.6; overflow-wrap: anywhere; }
    .match { background: #ffe5a8; color: #3d2c00; font-weight: 700; padding: 1px 2px; }
    div[data-testid="stMetric"] { background: #fff; border: 1px solid var(--line); padding: 14px 16px; border-radius: 4px; }
    div.stButton > button[kind="primary"] { background: var(--accent); border-color: var(--accent); }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Document Similarity Review")
st.markdown(
    '<p class="intro">Compare the documents you provide, inspect exact shared passages, and export a transparent analysis. '
    "Similarity scores are indicators for review, not proof of plagiarism.</p>",
    unsafe_allow_html=True,
)

st.subheader("Add documents")
upload_col, paste_col = st.columns([1, 1], gap="large")
with upload_col:
    uploaded_files = st.file_uploader(
        "Upload TXT, PDF, or DOCX files",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True,
        help="Text is extracted locally in this application. Password-protected files are not supported.",
    )
with paste_col:
    pasted_text = st.text_area(
        "Or paste a document",
        height=145,
        placeholder="Paste text here. It will be included as a separate document.",
    )

documents: list[dict[str, str]] = []
messages: list[tuple[str, str]] = []
used_names: dict[str, int] = {}


def unique_name(name: str) -> str:
    count = used_names.get(name, 0) + 1
    used_names[name] = count
    return name if count == 1 else f"{name} ({count})"


def highlighted_excerpt(text: str, start: int, end: int, context_chars: int = 90) -> str:
    excerpt_start = max(0, start - context_chars)
    excerpt_end = min(len(text), end + context_chars)
    prefix = "..." if excerpt_start else ""
    suffix = "..." if excerpt_end < len(text) else ""
    return (
        html.escape(prefix + text[excerpt_start:start])
        + '<span class="match">'
        + html.escape(text[start:end])
        + "</span>"
        + html.escape(text[end:excerpt_end] + suffix)
    )


for uploaded_file in uploaded_files or []:
    try:
        text = extract_text(uploaded_file.name, uploaded_file.getvalue())
        if not text.strip():
            messages.append(("warning", f"{uploaded_file.name}: no readable text was found."))
            continue
        documents.append({"name": unique_name(uploaded_file.name), "text": text})
    except DocumentParseError as error:
        messages.append(("error", str(error)))

if pasted_text.strip():
    documents.append({"name": unique_name("Pasted text"), "text": pasted_text})

for level, message in messages:
    getattr(st, level)(message)

if documents:
    with st.expander(f"Review extracted text ({len(documents)} document{'s' if len(documents) != 1 else ''})"):
        for document in documents:
            st.text_area(
                document["name"],
                value=document["text"],
                height=150,
                key=f"preview-{document['name']}",
                disabled=True,
            )

input_fingerprint = hashlib.sha256(
    "\0".join(f"{document['name']}\0{document['text']}" for document in documents).encode("utf-8")
).hexdigest()
if st.button("Analyze documents", type="primary", use_container_width=True):
    if len(documents) < 2:
        st.warning("Add at least two readable documents to begin a comparison.")
        st.session_state.pop("analysis", None)
    else:
        comparisons = compare_documents(documents)
        for comparison in comparisons:
            comparison["matches"] = find_matching_passages(
                documents[comparison["left_index"]]["name"],
                documents[comparison["left_index"]]["text"],
                documents[comparison["right_index"]]["name"],
                documents[comparison["right_index"]]["text"],
            )
        st.session_state["analysis"] = {
            "fingerprint": input_fingerprint,
            "documents": documents,
            "comparisons": comparisons,
        }

analysis = st.session_state.get("analysis")
if analysis and analysis["fingerprint"] != input_fingerprint:
    st.info("Inputs changed. Select **Analyze documents** to refresh the results.")
elif analysis:
    comparisons: list[dict[str, Any]] = analysis["comparisons"]
    analyzed_documents = analysis["documents"]
    average_score = (
        sum(item["combined_score"] for item in comparisons) / len(comparisons)
        if comparisons
        else 0.0
    )

    st.divider()
    st.subheader("Analysis overview")
    metric_columns = st.columns(3)
    metric_columns[0].metric("Documents analyzed", len(analyzed_documents))
    metric_columns[1].metric("Document pairs", len(comparisons))
    metric_columns[2].metric("Mean pair similarity", f"{average_score:.1%}")

    if not any(item["sufficient_text"] for item in comparisons):
        st.warning(
            "At least one document in every pair contains fewer than three words. "
            "Scores are shown for completeness but are not meaningful for such short text."
        )
    elif any(not item["sufficient_text"] for item in comparisons):
        st.info("Some pairs contain very little text; interpret those scores cautiously.")

    st.caption(
        "Combined score is the unweighted mean of TF-IDF cosine, unique word n-gram overlap, "
        "and word-set Jaccard. The methods are complementary indicators and do not establish intent or source."
    )

    names = [document["name"] for document in analyzed_documents]
    matrix = np.full((len(names), len(names)), np.nan)
    score_by_pair = {
        (item["left_index"], item["right_index"]): item["combined_score"]
        for item in comparisons
    }
    for left_index, right_index in combinations(range(len(names)), 2):
        value = score_by_pair[(left_index, right_index)] * 100
        matrix[left_index, right_index] = value
        matrix[right_index, left_index] = value

    st.subheader("Pairwise similarity matrix")
    heatmap = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=names,
            y=names,
            zmin=0,
            zmax=100,
            colorscale=[[0, "#edf1eb"], [0.5, "#9bc8b5"], [1, "#167663"]],
            colorbar={"title": "Similarity %"},
            hovertemplate="%{y} vs %{x}<br>%{z:.1f}%<extra></extra>",
            text=np.where(np.isnan(matrix), "", np.vectorize(lambda value: f"{value:.1f}%")(np.nan_to_num(matrix))),
            texttemplate="%{text}",
        )
    )
    heatmap.update_layout(height=max(320, 70 * len(names)), margin={"l": 20, "r": 20, "t": 20, "b": 20})
    st.plotly_chart(heatmap, use_container_width=True)

    st.subheader("Document comparison")
    comparison_table = pd.DataFrame(
        [
            {
                "Document A": item["left_name"],
                "Document B": item["right_name"],
                "Combined": f"{item['combined_score']:.1%}",
                "TF-IDF cosine": f"{item['tfidf_cosine']:.1%}",
                "N-gram overlap": f"{item['ngram_overlap']:.1%}",
                "Jaccard": f"{item['jaccard']:.1%}",
                "Exact passages": len(item["matches"]),
                "Text length": "Sufficient" if item["sufficient_text"] else "Too short",
            }
            for item in comparisons
        ]
    )
    st.dataframe(comparison_table, hide_index=True, use_container_width=True)

    st.subheader("Similarity by pair")
    figure, axis = plt.subplots(figsize=(max(7, len(comparisons) * 1.1), 3.4))
    colors = ["#167663" if item["sufficient_text"] else "#c4a35a" for item in comparisons]
    labels = [f"{item['left_name']} / {item['right_name']}" for item in comparisons]
    axis.bar(range(len(comparisons)), [item["combined_score"] * 100 for item in comparisons], color=colors)
    axis.set_ylabel("Combined similarity (%)")
    axis.set_ylim(0, 100)
    axis.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    axis.grid(axis="y", color="#d9e1db", linewidth=0.7)
    axis.set_axisbelow(True)
    figure.tight_layout()
    st.pyplot(figure, clear_figure=True)
    plt.close(figure)

    st.subheader("Matching passages")
    st.caption("Exact word-sequence matches are highlighted. Sentence numbers are approximate locations, not PDF page numbers.")
    any_matches = False
    for item in comparisons:
        with st.expander(f"{item['left_name']} ↔ {item['right_name']} · {len(item['matches'])} exact passage(s)"):
            if not item["matches"]:
                st.write("No exact phrase of five or more words was found. Similarity scores can still reflect partial or distributed overlap.")
            for match in item["matches"]:
                any_matches = True
                left_text = analyzed_documents[item["left_index"]]["text"]
                right_text = analyzed_documents[item["right_index"]]["text"]
                left_highlight = highlighted_excerpt(
                    left_text, match["left_start"], match["left_end"]
                )
                right_highlight = highlighted_excerpt(
                    right_text, match["right_start"], match["right_end"]
                )
                left_col, right_col = st.columns(2)
                left_col.caption(f"{item['left_name']} · approximately sentence {match['left_sentence']}")
                left_col.markdown(f'<div class="passage">{left_highlight}</div>', unsafe_allow_html=True)
                right_col.caption(f"{item['right_name']} · approximately sentence {match['right_sentence']}")
                right_col.markdown(f'<div class="passage">{right_highlight}</div>', unsafe_allow_html=True)
                st.caption(f"Exact normalized phrase: “{match['phrase']}” ({match['word_count']} words)")
    if not any_matches and comparisons:
        st.caption("No exact matching passages were identified in the analyzed pairs.")

    st.subheader("Export analysis")
    export_left, export_right = st.columns(2)
    with export_left:
        st.download_button(
            "Download CSV report",
            data=generate_csv_report(comparisons),
            file_name="similarity_report.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with export_right:
        st.download_button(
            "Download readable TXT report",
            data=generate_text_report(comparisons),
            file_name="similarity_report.txt",
            mime="text/plain",
            use_container_width=True,
        )

st.divider()
st.caption(
    "This local tool only compares submitted documents. It does not search the web or a reference corpus; "
    "paraphrases may be missed and common phrases can produce false positives. Similarity is not a definitive plagiarism verdict."
)