import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
import requests
from bs4 import BeautifulSoup
import time
from groq import Groq
import os
from urllib.parse import urljoin

st.set_page_config(page_title="POLARIN Polar Data Assistant", page_icon="❄️", layout="wide")
st.title("❄️ POLARIN Polar Data RAG Assistant")
st.markdown("Ask questions about polar data access, get ready-to-run Python code using **erddapy**, **xarray**, **cartopy**, etc.")

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("Settings")
    groq_api_key = st.text_input("Groq API Key", type="password", value=os.getenv("GROQ_API_KEY", ""))
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3)
    
    if st.button("🔄 Re-scrape POLARIN Knowledge Base"):
        with st.spinner("Scraping POLARIN site..."):
            scrape_polarin_data()
        st.success("Knowledge base updated!")

# ====================== INITIALIZATION ======================
@st.cache_resource
def get_chroma_client():
    return chromadb.PersistentClient(path="./polarin_chroma")

@st.cache_resource
def get_groq_client():
    return Groq(api_key=groq_api_key) if groq_api_key else None

client = get_chroma_client()
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

collection = client.get_or_create_collection(
    name="polarin_docs",
    embedding_function=embedding_func
)

# ====================== SCRAPING FUNCTION ======================
def scrape_polarin_data():
    urls = [
        "https://s4polarin.eu/",
        "https://s4polarin.eu/virtual-access/",
        "https://s4polarin.eu/data-catalog/",
        # Add more pages if you discover them
    ]
    
    documents = []
    metadatas = []
    ids = []
    
    for url in urls:
        try:
            resp = requests.get(url, timeout=10)
            soup = BeautifulSoup(resp.text, 'html.parser')
            text = soup.get_text(separator="\n", strip=True)
            
            # Chunking
            chunks = [text[i:i+800] for i in range(0, len(text), 700)]
            
            for i, chunk in enumerate(chunks):
                documents.append(chunk)
                metadatas.append({
                    "source": url,
                    "type": "documentation" if "catalog" in url else "general"
                })
                ids.append(f"{url}_{i}")
                
        except Exception as e:
            st.warning(f"Failed to scrape {url}: {e}")
    
    # Add general ERDDAP knowledge
    add_erddap_knowledge()
    
    if documents:
        collection.add(documents=documents, metadatas=metadatas, ids=ids)
        return len(documents)
    return 0

def add_erddap_knowledge():
    erddap_examples = [
        ("POLARIN uses ERDDAP servers for polar datasets. Use erddapy to connect.", {"source": "erddap_guide", "type": "code_example"}),
        ("Basic example: from erddapy import ERDDAP; e = ERDDAP(server='https://polarin-erddap.example.org/erddap', protocol='griddap')", {"source": "erddap_guide", "type": "code_example"}),
    ]
    # Add more static knowledge here as needed
    docs, metas, id_list = zip(*[(text, meta, f"erddap_{i}") for i, (text, meta) in enumerate(erddap_examples)])
    collection.add(documents=list(docs), metadatas=list(metas), ids=list(id_list))

# ====================== RAG QUERY ======================
def query_rag(question: str, n_results=5):
    results = collection.query(
        query_texts=[question],
        n_results=n_results,
        include=["documents", "metadatas"]
    )
    context = "\n\n".join(results['documents'][0])
    return context

# ====================== LLM RESPONSE ======================
def generate_response(question: str, context: str):
    client = get_groq_client()
    if not client:
        return "Please provide your Groq API key in the sidebar."
    
    system_prompt = """You are a helpful polar data expert.
You help researchers access and plot polar data (Arctic/Antarctic) from POLARIN and ERDDAP servers.
Always provide clean, well-commented Python code using erddapy, xarray, matplotlib/cartopy, pandas.
Include installation instructions when needed.
Be accurate and safe (no arbitrary code execution)."""

    user_prompt = f"""Context from POLARIN documentation:
{context}

Question: {question}

Provide a clear answer + working Python code example."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature,
        max_tokens=2048
    )
    return response.choices[0].message.content

# ====================== MAIN APP ======================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if prompt := st.chat_input("Ask about polar data, ERDDAP, plotting..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking + retrieving knowledge..."):
            context = query_rag(prompt)
            response = generate_response(prompt, context)
            st.markdown(response)
            
            # Nice code block highlighting
            if "```python" in response:
                st.success("Copy the code above 👆")

    st.session_state.messages.append({"role": "assistant", "content": response})

# ====================== UTILITIES ======================
st.sidebar.markdown("---")
if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []

if st.sidebar.button("Show Knowledge Base Size"):
    count = collection.count()
    st.sidebar.info(f"Documents in DB: **{count}**")

st.caption("Built for POLARIN Data Hub • Uses Groq + ChromaDB + erddapy patterns")
