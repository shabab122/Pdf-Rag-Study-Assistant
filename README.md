# PDF RAG Study Assistant

A web and command-line study assistant that reads local PDF notes, splits their
text into overlapping chunks, stores semantic embeddings in ChromaDB, retrieves
the most relevant passages, and asks a Groq-hosted language model to answer in
natural Banglish using the retrieved PDF content.

## Features

- Reads one or more text-based PDF files.
- Preserves the source filename and page number for every chunk.
- Stores embeddings locally in a persistent ChromaDB database.
- Supports semantic search without calling an LLM.
- Generates answers grounded in the retrieved PDF passages.
- Includes a polished Streamlit chat interface with PDF upload and indexing.
- Shows source pages and the retrieved passages below each web answer.
- Loads the Groq API key from a local `.env` file.
- Includes clear checks for missing PDFs, scanned PDFs, and a missing index.

## How it works

1. PDF text is extracted with `pypdf`.
2. Text is divided into 500-character chunks with 100-character overlap.
3. `all-MiniLM-L6-v2` converts every chunk into an embedding.
4. ChromaDB stores the embeddings, text, filename, and page number locally.
5. A question is embedded and matched with the nearest PDF chunks.
6. Groq generates an answer from those retrieved chunks.

## Project structure

```text
pdf-rag-study-assistant-main/
├── .streamlit/config.toml  # Web theme and upload limit
├── app.py                  # Streamlit web interface
├── data/                   # Put local PDF files here
├── src/
│   ├── config.py           # Shared paths and model settings
│   ├── embedding_model.py  # Cached embedding-model loader
│   ├── read_pdf.py         # Extract and preview PDF text
│   ├── chunk_pdf.py        # Split extracted text into chunks
│   ├── store_embeddings.py # Build the local ChromaDB index
│   ├── search_pdf.py       # Search the index without an LLM
│   └── rag_answer.py       # Retrieve context and generate an answer
├── tests/                  # Lightweight unit tests
├── .env.example            # Environment-variable template
├── .gitignore
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.11 or 3.12 is recommended.
- Internet access is needed during installation, the first embedding-model
  download, and Groq answer generation.
- A Groq API key is required only for `rag_answer.py`.

## Setup on Windows PowerShell

Run these commands from the project folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Open `.env` and replace the placeholder with your real Groq API key:

```dotenv
GROQ_API_KEY=your_real_api_key
```

If PowerShell blocks virtual-environment activation, run this once in the same
terminal and then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

This changes the policy only for the current PowerShell process.

## Setup on Linux or macOS

Run these commands from the project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

Then edit `.env` and replace the placeholder with your real Groq API key.

## Use the assistant

### Recommended: launch the web interface

```bash
streamlit run app.py
```

Streamlit prints a local address, normally `http://localhost:8501`. Open that
address in your browser. From the sidebar you can:

1. Upload one or more text-based PDFs.
2. Save the uploaded PDFs into `data/`.
3. Build or rebuild the knowledge base.
4. Ask questions through the chat box.

The first index build can take longer because the embedding model must be
downloaded. The app keeps the loaded embedding model in memory so later
questions are faster.

### Command-line workflow

#### 1. Add PDF files

Place one or more `.pdf` files inside the `data` folder. Local PDFs are ignored
by Git so private study documents are not accidentally committed.

#### 2. Optional: preview extracted text

```bash
python -m src.read_pdf
```

#### 3. Optional: preview text chunks

```bash
python -m src.chunk_pdf
```

#### 4. Build or rebuild the vector database

```bash
python -m src.store_embeddings
```

Run this command again whenever PDFs are added, removed, or changed. Rebuilding
replaces the previous `course_notes` collection, preventing duplicate chunks.

#### 5. Search without generating an answer

```bash
python -m src.search_pdf "What is a shell script?"
```

To request a different number of matching chunks:

```bash
python -m src.search_pdf "What is a shell script?" --results 5
```

#### 6. Ask the RAG assistant

Interactive mode:

```bash
python -m src.rag_answer
```

Or pass the question directly:

```bash
python -m src.rag_answer "What is a shell script?"
```

## Run tests

```bash
python -m unittest discover -s tests -v
```

## Common problems

### No PDF found

Make sure at least one file ending in `.pdf` is directly inside `data/`, then
run the command again.

### No extractable text was found

The current project does not perform OCR. A scanned image-only PDF must first be
converted to a searchable/text-based PDF with an OCR tool.

### The vector database or PDF index is missing

Build it before searching or asking questions:

```bash
python -m src.store_embeddings
```

### `GROQ_API_KEY` was not found

Create `.env` from `.env.example`, add the key, save the file, and run the
assistant again. Never commit `.env` or paste a real API key into source code.

### The first run is slow

The embedding model is downloaded on first use. Later runs reuse the local
model cache.

## Current limitations

- Image-only PDFs need OCR before this project can read them.
- Chunking is character-based, so it can split a sentence between chunks.
- Answers depend on the quality of PDF extraction and retrieval.
- The web interface is intended for local use and does not include user accounts.
- The model may still make mistakes, so verify important answers against the
  cited source filename and page in the retrieved context.

## Security and privacy

- `.env`, local PDFs, and `chroma_db/` are excluded from Git.
- PDF text sent as context to Groq leaves the local machine during answer
  generation. Use `search_pdf.py` if the documents must remain fully local.
- Treat model output as study assistance, not as an authoritative source.
