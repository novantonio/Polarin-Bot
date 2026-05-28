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
st.caption("Ask how to access & plot polar data — get Python code with erddapy, xarray, cartopy")

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("Settings")
    groq_api_key = st.text_input("Groq API Key", type="password", value=os.getenv("GROQ_API_KEY", ""))
    
    if st.button("🔄 Scrape POLARIN Knowledge Base"):
        with st.spinner("Scraping site..."):
            num_docs = scrape_polarin()
            st.success(f"Added {num_docs} document chunks!")

# ====================== LOAD EMBEDDING MODEL ======================
@st.cache_resource
def load_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedder = load_embedder()

# ====================== FAISS + DOCUMENTS ======================
if "documents" not in st.session_state:
    st.session_state.documents = []   # list of text chunks
    st.session_state.metadatas = []   # list of dicts with source

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
            resp = requests.get(url, timeout=15)
            soup = BeautifulSoup(resp.text, "html.parser")
            text = soup.get_text(separator="\n", strip=True)
            
            # Simple chunking
            chunks = [text[i:i+850] for i in range(0, len(text), 700)]
            
            for chunk in chunks:
                if chunk.strip():
                    st.session_state.documents.append(chunk)
                    st.session_state.metadatas.append({"source": url})
                    new_chunks += 1
        except Exception as e:
            st.warning(f"Could not scrape {url}: {e}")
    
    # Rebuild FAISS index
    rebuild_faiss_index()
    return new_chunks

def rebuild_faiss_index():
    if not st.session_state.documents:
        return
    embeddings = embedder.encode(st.session_state.documents, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    st.session_state.faiss_index = index

# ====================== RAG QUERY ======================
def query_rag(question: str, k=6):
    if "faiss_index" not in st.session_state or not st.session_state.documents:
        return "No knowledge base yet. Please click 'Scrape POLARIN' first."
    
    query_vec = embedder.encode([question], convert_to_numpy=True)
    distances, indices = st.session_state.faiss_index.search(query_vec, k)
    
    context = "\n\n".join([st.session_state.documents[i] for i in indices[0]])
    return context

# ====================== LLM ======================
def generate_answer(question: str, context: str):
    if not groq_api_key:
        return "Please enter your Groq API key in the sidebar."
    
    client = Groq(api_key=groq_api_key)
    
    system_prompt = """You are a polar data expert specialized in POLARIN and ERDDAP.
Provide clear answers and always include well-commented, ready-to-run Python code using:
- erddapy
- xarray
- matplotlib / cartopy
- pandas"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context from POLARIN:\n{context}\n\nQuestion: {question}"}
        ],
        temperature=0.3,
        max_tokens=2048
    )
    return response.choices[0].message.content

# ====================== CHAT INTERFACE ======================
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("E.g. How do I plot Arctic sea ice concentration?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching POLARIN knowledge + generating code..."):
            context = query_rag(prompt)
            answer = generate_answer(prompt, context)
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

# Footer
st.sidebar.markdown("---")
if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
st.sidebar.caption("FAISS + Sentence-Transformers + Groq")
