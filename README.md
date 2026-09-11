# PDF RAG Study Assistant — Similarity Threshold Edition

A small PDF question-answering app built to practice the main RAG steps:

1. Extract text from local PDFs.
2. Split text into overlapping chunks.
3. Create embeddings with `all-MiniLM-L6-v2`.
4. Store embeddings and page metadata in ChromaDB.
5. Retrieve the most relevant chunks for a question.
6. Reject weak matches with a configurable similarity threshold.
7. Ask Groq to answer only from accepted PDF context.

The project supports a command-line workflow and a polished local Streamlit web UI. The UI can upload PDFs, rebuild the index, choose one PDF or all PDFs, tune the threshold live, keep question history for the current session, and show source page numbers. Answers are generated in Banglish, while source PDF names and page numbers are shown separately.

## What changed in this version

- Added one shared retrieval module: `src/retrieval.py`.
- Added a distance-based relevance threshold. ChromaDB distance is lower for better matches.
- Default threshold is `0.75`.
- If no retrieved chunk has `distance <= threshold`, the app does not call Groq and reports that the answer was not found in the PDF.
- Streamlit and CLI output show accepted sources/pages.
- Streamlit UI supports local PDF upload, PDF selection, live threshold tuning, and session question history.
- Uploaded PDFs are saved under `data/`; click **Rebuild vector index** after uploading before asking questions.
- Embedding model loading is cached during a process.
- Added threshold unit tests and a smaller, readable `requirements.txt`.

> This is technically a maximum **distance** threshold (lower is better), often called a similarity threshold in the UI. `0.75` is an initial value based on this embedding model and should be tuned for your PDFs.

## Folder structure

```text
pdf-rag-study-assistant-threshold-v2/
├── app.py
├── data/                         # Put local PDFs here; PDFs are ignored by Git
├── src/
│   ├── read_pdf.py               # PDF text extraction
│   ├── chunk_pdf.py               # Overlapping chunk creation
│   ├── store_embeddings.py        # Build/rebuild ChromaDB
│   ├── retrieval.py               # Shared retrieval + threshold filtering
│   ├── search_pdf.py              # Inspect accepted/rejected matches
│   └── rag_answer.py               # Groq-grounded answer generation
├── tests/test_retrieval.py
├── .env.example
├── .gitignore
└── requirements.txt
```

## Windows PowerShell setup

Python 3.11 or 3.12 is recommended for the ML dependencies on Windows. If your current Python 3.14 environment shows a `torchvision` or DLL error, install one of those versions and create a fresh virtual environment.

Run these commands from the project root (the folder containing `app.py`):

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Copy one or more text-based PDFs into `data/`, then build the vector database:

```powershell
python src\store_embeddings.py
```

Set the Groq key for the current PowerShell window:

```powershell
$env:GROQ_API_KEY="your-groq-api-key"
```

Run the web app:

```powershell
streamlit run app.py
```

Or use the CLI:

```powershell
python src\rag_answer.py
```

## Tune the threshold

Inspect distances first:

```powershell
python src\search_pdf.py
```

Then change the value for the current terminal session. Lower values are stricter; higher values allow more context:

```powershell
$env:RAG_DISTANCE_THRESHOLD="0.85"
streamlit run app.py
```

Use a few questions whose answers are definitely in the PDFs and a few unrelated questions. A good threshold accepts the first group and rejects the second. The threshold affects retrieval only; it does not change the embedding model.

## Linux/macOS commands

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/store_embeddings.py
export GROQ_API_KEY="your-groq-api-key"
export RAG_DISTANCE_THRESHOLD="0.75"
streamlit run app.py
```

## Important notes

- Run `store_embeddings.py` again whenever PDFs are added or changed.
- Keep API keys out of Git. `.env.example` is only a template; never commit a real key.
- Scanned/image-only PDFs need OCR before text extraction can read them.
- The current generator uses Groq (`openai/gpt-oss-20b`); the threshold layer is independent of the LLM provider.

## Tests

```powershell
python -m pytest -q
```
