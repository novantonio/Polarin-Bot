import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"   # ← FIX for protobuf error

import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
import requests
from bs4 import BeautifulSoup
from groq import Groq

st.set_page_config(page_title="POLARIN Assistant", page_icon="❄️", layout="wide")
st.title("❄️ POLARIN Polar Data RAG Assistant")

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("Configuration")
    groq_api_key = st.text_input("Groq API Key", type="password", value=os.getenv("GROQ_API_KEY", ""))
    if st.button("🔄 Rescrape POLARIN"):
        with st.spinner("Scraping..."):
            scrape_polarin_data()
        st.success("Done!")

# ====================== CLIENTS ======================
@st.cache_resource
def get_chroma():
    client = chromadb.PersistentClient(path="./chroma_polarin")
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    return client.get_or_create_collection(
        name="polarin_docs",
        embedding_function=embedding_func
    )

@st.cache_resource
def get_groq():
    return Groq(api_key=groq_api_key) if groq_api_key else None

collection = get_chroma()

# ====================== SCRAPING ======================
def scrape_polarin_data():
    urls = [
        "https://s4polarin.eu/",
        "https://s4polarin.eu/virtual-access/",
        "https://s4polarin.eu/data-catalog/",
        "https://s4polarin.eu/erddap",
    ]
    docs = []
    metas = []
    ids = []

    for url in urls:
        try:
            r = requests.get(url, timeout=15)
            soup = BeautifulSoup(r.text, "html.parser")
            text = soup.get_text(separator="\n", strip=True)
            chunks = [text[i:i+900] for i in range(0, len(text), 700)]
            
            for i, chunk in enumerate(chunks):
                docs.append(chunk)
                metas.append({"source": url})
                ids.append(f"{url.split('//')[1]}_{i}")
        except Exception as e:
            st.error(f"Failed {url}: {e}")

    if docs:
        collection.add(documents=docs, metadatas=metas, ids=ids)

# ====================== RAG + LLM ======================
def query_rag(question, n=6):
    results = collection.query(query_texts=[question], n_results=n)
    return "\n\n".join(results['documents'][0]) if results['documents'] else ""

def generate_answer(question, context):
    groq = get_groq()
    if not groq:
        return "Please add your Groq API key."

    system = """You are an expert on polar data (Arctic & Antarctic). 
Help users access and visualize data from POLARIN / ERDDAP using Python.
Always return clean, commented code with erddapy + xarray + matplotlib/cartopy."""

    response = groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ],
        temperature=0.3,
        max_tokens=2000
    )
    return response.choices[0].message.content

# ====================== CHAT ======================
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("How do I plot sea ice concentration from POLARIN?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving POLARIN knowledge..."):
            context = query_rag(prompt)
            answer = generate_answer(prompt, context)
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

st.sidebar.caption("POLARIN RAG Bot • ChromaDB + Groq")
