import os
import streamlit as st
import requests
from bs4 import BeautifulSoup
from groq import Groq
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json  # NEW: for parsing .ipynb

st.set_page_config(page_title="POLARIN Assistant", page_icon="❄️", layout="wide")
st.title("❄️ POLARIN Polar Data RAG Assistant + Code Learner")
st.caption("Now learns from Jupyter notebooks too!")

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("Settings")
    groq_api_key = st.secrets.get("GROQ_API_KEY") or st.text_input("Groq API Key", type="password")

    if st.button("🔄 Scrape POLARIN Site"):
        with st.spinner("Scraping..."):
            num = scrape_polarin()
            st.success(f"Added {num} chunks!")

    uploaded_file = st.file_uploader("Upload .ipynb to learn from", type=["ipynb"])
    if uploaded_file and st.button("📘 Learn from Notebook"):
        with st.spinner("Extracting code & explanations..."):
            num_chunks = learn_from_notebook(uploaded_file)
            st.success(f"Learned {num_chunks} new chunks from notebook!")

    if st.button("🗑️ Clear Knowledge Base"):
        st.session_state.documents = []
        st.session_state.metadatas = []
        if "faiss_index" in st.session_state:
            del st.session_state.faiss_index
        st.success("Knowledge base cleared.")

# ====================== EMBEDDING MODEL ======================
@st.cache_resource
def load_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedder = load_embedder()

if "documents" not in st.session_state:
    st.session_state.documents = []
    st.session_state.metadatas = []

# ====================== NEW: NOTEBOOK PARSER ======================
def learn_from_notebook(uploaded_file):
    notebook = json.load(uploaded_file)
    new_chunks = 0

    for cell in notebook.get("cells", []):
        if cell["cell_type"] == "code":
            code = "".join(cell["source"])
            if code.strip():
                st.session_state.documents.append(code)
                st.session_state.metadatas.append({
                    "source": "notebook",
                    "type": "code_example",
                    "language": "python"
                })
                new_chunks += 1

        elif cell["cell_type"] == "markdown":
            text = "".join(cell["source"])
            if text.strip():
                st.session_state.documents.append(text)
                st.session_state.metadatas.append({
                    "source": "notebook",
                    "type": "explanation"
                })
                new_chunks += 1

    if new_chunks > 0:
        rebuild_faiss_index()
    return new_chunks

# ====================== REST OF THE CODE (unchanged) ======================
def scrape_polarin():
    urls = ["https://s4polarin.eu/", "https://s4polarin.eu/virtual-access/", "https://s4polarin.eu/data-catalog/"]
    new_chunks = 0
    for url in urls:
        try:
            resp = requests.get(url, timeout=20)
            soup = BeautifulSoup(resp.text, "html.parser")
            text = soup.get_text(separator="\n", strip=True)
            chunks = [text[i:i+850] for i in range(0, len(text), 700)]
            for chunk in chunks:
                if len(chunk.strip()) > 60:
                    st.session_state.documents.append(chunk)
                    st.session_state.metadatas.append({"source": url})
                    new_chunks += 1
        except:
            st.warning(f"Failed {url}")
    rebuild_faiss_index()
    return new_chunks

def rebuild_faiss_index():
    if not st.session_state.documents:
        return
    embeddings = embedder.encode(st.session_state.documents, convert_to_numpy=True).astype(np.float32)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    st.session_state.faiss_index = index

def get_context(question: str, k=6):
    if "faiss_index" not in st.session_state or not st.session_state.documents:
        return "Please load knowledge first (Scrape or upload notebook)."
    query_vec = embedder.encode([question], convert_to_numpy=True).astype(np.float32)
    _, indices = st.session_state.faiss_index.search(query_vec, k)
    context = "\n\n".join([st.session_state.documents[i] for i in indices[0]])
    return context

def generate_response(question: str, context: str):
    if not groq_api_key:
        return "Please provide Groq API key."
    client = Groq(api_key=groq_api_key)
    system = """You are a polar data expert. Use real code patterns from notebooks (erddapy, xarray, cartopy, pandas).
Always return clean, commented, ready-to-run Python code."""

    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Context (includes notebook code):\n{context}\n\nQuestion: {question}"}
        ],
        temperature=0.3,
        max_tokens=2048
    )
    return resp.choices[0].message.content

# ====================== CHAT ======================
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("E.g.: Plot sea ice temperature from ERDDAP like in the notebook"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Using learned notebook patterns..."):
            context = get_context(prompt)
            answer = generate_response(prompt, context)
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

st.sidebar.caption("FAISS RAG + Notebook Learner")
