# PDF RAG Study Assistant Pro v3

## Windows

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Put PDFs inside `data/`.

Build embeddings:
```powershell
python src\store_embeddings.py
```

Add Groq key:

Option 1:
Edit `.env`
```
GROQ_API_KEY=your_key
```

Option 2:
```
$env:GROQ_API_KEY="your_key"
```

Run:
```
streamlit run app.py
```

Get Groq key from Groq Console:
https://console.groq.com/

## Linux/macOS

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/store_embeddings.py
export GROQ_API_KEY="your_key"
streamlit run app.py
```
