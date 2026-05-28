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
st.caption("Get Python code to access & plot polar data from ERDDAP / POLARIN")

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("Settings")
    groq_api_key = st.text_input("Groq API Key", type="password", value=os.getenv("GROQ_API_KEY", ""))
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Scrape POLARIN"):
            with st.spinner("Scraping..."):
                num = scrape_polarin()
                st.success(f"Added {num} chunks!")
    with col2:
        if st.button("Clear KB"):
            st.session_state.documents = []
            st.session_state.metadatas = []
            if "faiss_index" in st.session_state:
                del st.session_state.faiss_index
            st.success("Knowledge base cleared.")

# ====================== EMBEDDINGS ======================
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
        "https://s4polarin.eu/erddap",
    ]
    new_chunks = 0
    for url in urls:
        try:
            resp = requests.get(url, timeout=20)
            soup = BeautifulSoup(resp.text, "html.parser")
            text = soup.get_text(separator="\n", strip=True)
            chunks = [text[i:i+900] for i in range(0, len(text), 700)]
            for chunk in chunks:
                if len(chunk.strip()) > 50:
                    st.session_state.documents.append(chunk)
                    st.session_state.metadatas.append({"source": url})
                    new_chunks += 1
        except Exception as e:
            st.warning(f"Failed to scrape {url}")
    
    rebuild_faiss_index()
    return new_chunks

def rebuild_faiss_index():
    if not st.session_state.documents:
        return
    embeddings = embedder.encode(st.session_state.documents, convert_to_numpy=True).astype(np.float32)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    st.session_state.faiss_index = index

# ====================== RAG ======================
def query_rag(question: str, k=5):
    if "faiss_index" not in st.session_state or not st.session_state.documents:
        return "Please click 'Scrape POLARIN' first to build the knowledge base."
    
    query_vec = embedder.encode([question], convert_to_numpy=True).astype(np.float32)
    _, indices = st.session_state.faiss_index.search(query_vec, k)
    context = "\n\n".join(st.session_state.documents[i] for i in indices[0])
    return context

# ====================== LLM ======================
def generate_answer(question: str, context: str):
    if not groq_api_key:
        return "Enter your Groq API key in the sidebar."
    
    client = Groq(api_key=groq_api_key)
    system = """You are a polar data specialist. Help users access and plot data from POLARIN/ERDDAP.
Always provide clean, commented Python code using erddapy, xarray, matplotlib, cartopy."""

    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
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

if prompt := st.chat_input("Example: Show me code to plot sea ice extent from POLARIN"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            context = query_rag(prompt)
            answer = generate_answer(prompt, context)
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

st.sidebar.caption("FAISS RAG • Stable on Streamlit Cloud")
