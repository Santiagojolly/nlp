# Document Similarity Review

A local Streamlit application for comparing two or more TXT, PDF, and DOCX documents, reviewing exact shared passages, and exporting an analysis. It uses no paid APIs and sends no document text to a remote service.

## Project structure

```text
plagiarism-detector/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── document_parser.py
│   ├── text_preprocessor.py
│   ├── similarity_engine.py
│   ├── passage_matcher.py
│   └── report_generator.py
├── tests/
│   ├── test_preprocessing.py
│   └── test_similarity.py
└── sample_documents/
    ├── research_a.txt
    └── research_b.txt
```

## Dependencies

`requirements.txt` installs Streamlit for the interface; NLTK for Porter stemming; scikit-learn for TF-IDF and cosine similarity; NumPy and pandas for analysis tables; Plotly and Matplotlib for visualizations; pypdf and python-docx for document extraction; and pytest for unit tests.

Python 3.10 or newer is recommended. PDF extraction works for text-based PDFs; scanned image-only PDFs need OCR, which this application does not perform.

## Setup and run

Run these commands from the `plagiarism-detector` directory.
On Debian/Ubuntu, if Python reports that `ensurepip` is unavailable, install the venv support package first: `sudo apt update && sudo apt install python3-venv`.

### Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

Run the tests in a second terminal (activate `.venv` there as well):

```bash
cd plagiarism-detector
source .venv/bin/activate
python -m pytest
```

### Windows PowerShell

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

Run the tests in a second PowerShell terminal:

```powershell
cd plagiarism-detector
.\.venv\Scripts\Activate.ps1
python -m pytest
```

If PowerShell blocks virtual-environment activation, use Command Prompt instead:

```bat
py -3 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

## Using the application

Upload multiple supported files, paste an additional document if useful, review extracted text, then select **Analyze documents**. The dashboard shows each pair's combined score, TF-IDF cosine, n-gram overlap, word-set Jaccard, a pairwise heatmap, a comparison chart, exact phrase matches with approximate sentence locations, and CSV/TXT report downloads. The `sample_documents/` files can be uploaded to try the workflow; they are not loaded automatically.

## How scores work

- **TF-IDF cosine:** cosine similarity of document TF-IDF vectors, using unigrams and bigrams with English stop words excluded.
- **N-gram overlap:** shared unique word shingles divided by the larger document shingle-set size (three-word shingles by default).
- **Jaccard:** intersection-over-union of the documents' unique normalized word sets.
- **Combined similarity:** the unweighted arithmetic mean of those three scores.
- **Exact passages:** matching sequences of at least five normalized words, highlighted against original text. The displayed sentence number is approximate; PDF page coordinates are not retained.

The preprocessing module also exposes stop-word-filtered tokens and NLTK Porter-stemmed tokens for further experiments. Current scores and exact passage detection use unstemmed words so that the displayed phrase corresponds directly to source text.

## Limitations

This is document-to-document similarity analysis, not a universal plagiarism search engine. It only compares documents supplied to the app, may miss paraphrases and translated text, and can flag common phrases. Extraction quality affects results. A similarity score is an indicator for human review, not a definitive plagiarism verdict or a determination of academic misconduct. Very short documents receive a clear warning because their scores are not meaningful.