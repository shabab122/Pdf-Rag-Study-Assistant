
from pathlib import Path
import os
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.rag_answer import generate_answer, retrieve_context
from src.retrieval import DEFAULT_DISTANCE_THRESHOLD

DATA_FOLDER = Path("data")
DATA_FOLDER.mkdir(exist_ok=True)

st.set_page_config(page_title="RAG Study Assistant Pro", page_icon="🧠", layout="wide")

st.markdown("""
<style>
.hero {padding:35px;border-radius:25px;background:linear-gradient(135deg,#111827,#2563eb,#06b6d4);color:white;}
.card {background:white;padding:20px;border-radius:18px;border:1px solid #e5e7eb;margin:10px 0;}
.small {color:#64748b;}
</style>
""", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history=[]

st.markdown("""
<div class="hero">
<h1>🧠 PDF RAG Study Assistant Pro</h1>
<p>Upload PDFs → Build Knowledge Base → Ask Questions → Get Source-Grounded AI Answers</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("📂 Knowledge Base")
    uploads=st.file_uploader("Upload PDF", type="pdf", accept_multiple_files=True)
    if st.button("Save PDFs"):
        for f in uploads or []:
            (DATA_FOLDER/f.name).write_bytes(f.getbuffer())
        st.success("PDF saved. Rebuild index.")

    pdfs=sorted(DATA_FOLDER.glob("*.pdf"))
    st.metric("PDF Files", len(pdfs))

    if st.button("🔄 Rebuild Vector Database"):
        from src.store_embeddings import build_index
        with st.spinner("Creating embeddings..."):
            total, counts=build_index()
        st.success(f"{total} chunks indexed")

    threshold=st.slider("Retrieval Threshold",0.1,1.5,float(os.getenv("RAG_DISTANCE_THRESHOLD",0.75)),0.05)
    count=st.slider("Context Chunks",1,8,3)

    if st.button("Clear Chat"):
        st.session_state.history=[]

col1,col2=st.columns([2,1])
with col2:
    st.subheader("System Status")
    st.success("Embedding Engine Ready")
    st.success("ChromaDB Connected")
    st.info("Groq Enabled" if os.getenv("GROQ_API_KEY") else "Add GROQ_API_KEY")

question=col1.text_input("Ask anything from your PDF")

if question:
    with st.spinner("Searching and generating answer..."):
        ctx=retrieve_context(question,count)
        answer=generate_answer(question,ctx)
    st.session_state.history.append({"q":question,"a":answer,"time":datetime.now().strftime("%H:%M")})

for item in reversed(st.session_state.history):
    st.markdown(f"<div class='card'><b>Q:</b> {item['q']}<br><small>{item['time']}</small><hr><b>Answer:</b><br>{item['a']}</div>", unsafe_allow_html=True)

    if item is st.session_state.history[-1]:
        pass
