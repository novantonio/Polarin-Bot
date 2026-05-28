import os
import streamlit as st
import requests
from bs4 import BeautifulSoup
from groq import Groq
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="POLARIN Assistant", page_icon="❄️", layout="wide")
st.title("❄️ POLARIN Polar Data RAG Assistant")
st.caption("Ask for Python code to access and plot polar data via ERDDAP")

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("Settings")
    groq_api_key = st.text_input("Groq API Key", type="password", value=os.getenv("GROQ_API_KEY", ""))
    
    if st.button("🔄 Scrape POLARIN Site"):
        with st.spinner("Scraping documentation..."):
            num_chunks = scrape_polarin()
            st.success(f"Loaded {num_chunks} document chunks!")

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

# ====================== SESSION STATE ======================
if "documents" not in st.session_state:
    st.session_state.documents = []
    st.session_state.metadatas = []

def scrape_polarin():
    urls = [
        "https://s4polarin.eu/",
        "https://s4polarin.eu/virtual-access/",
        "https://s4polarin.eu/data-catalog/",
    ]
    
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
        except Exception as e:
            st.warning(f"Failed scraping {url}")
    
    rebuild_faiss_index()
    return new_chunks

def rebuild_faiss_index():
    if len(st.session_state.documents) == 0:
        return
    embeddings = embedder.encode(st.session_state.documents, convert_to_numpy=True).astype(np.float32)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    st.session_state.faiss_index = index

# ====================== RAG ======================
def get_context(question: str, k=5):
    if "faiss_index" not in st.session_state or len(st.session_state.documents) == 0:
        return "Please click 'Scrape POLARIN Site' first."
    
    query_vec = embedder.encode([question], convert_to_numpy=True).astype(np.float32)
    _, indices = st.session_state.faiss_index.search(query_vec, k)
    context = "\n\n".join([st.session_state.documents[i] for i in indices[0]])
    return context

# ====================== LLM ======================
def generate_response(question: str, context: str):
    if not groq_api_key:
        return "Please enter your Groq API key in the sidebar."
    
    client = Groq(api_key=groq_api_key)
    
    system_prompt = """You are an expert on polar research data (Arctic and Antarctic).
You help scientists access and visualize data from POLARIN using Python.
Always give clean, well-commented code with erddapy, xarray, matplotlib or cartopy."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context from POLARIN:\n{context}\n\nUser question: {question}"}
        ],
        temperature=0.3,
        max_tokens=2048
    )
    return response.choices[0].message.content

# ====================== CHAT ======================
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("E.g.: Show me Python code to plot sea ice concentration"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            context = get_context(prompt)
            answer = generate_response(prompt, context)
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

st.sidebar.caption("FAISS RAG Bot - Stable Version")
