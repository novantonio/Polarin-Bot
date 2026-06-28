"""
POLARIN Bot — Extended Edition
================================
Estensioni rispetto alla versione originale:
  1. Scraping automatico dei 17 capitoli di literacy.s4oceanice.eu
  2. Generazione di notebook Colab (.ipynb) eseguibili a partire da una risposta
  3. Categorizzazione delle query per tema (ERDDAP, Argo, Sea Ice, CTD, WMS, CORA…)
  4. Pannello "Colab Builder" con download diretto del .ipynb
  5. Integrazione con GitHub per recuperare i notebook sorgente (raw)
"""

import os
import json
import re
import time
import base64
import textwrap
from datetime import datetime

import requests
import numpy as np
import streamlit as st
from bs4 import BeautifulSoup
from groq import Groq
import faiss
from sentence_transformers import SentenceTransformer

# ─── pagina ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="POLARIN Assistant", page_icon="❄️", layout="wide")
st.title("❄️ POLARIN Polar Data RAG Assistant — Extended")
st.caption("Scraping OCEAN:ICE Literacy Book + Colab Notebook Generator")

# ─── catalogo capitoli OCEAN:ICE ────────────────────────────────────────────
OCEANICE_CHAPTERS = {
    "ch01_tabledap":  ("ERDDAP tabledap querying",
                       "https://literacy.s4oceanice.eu/chapters/chapter1/oceanice_erddap_querying_tabledap.html"),
    "ch02_griddap":   ("ERDDAP griddap querying",
                       "https://literacy.s4oceanice.eu/chapters/chapter2/oceanice_erddap_querying_griddap.html"),
    "ch03_platforms": ("Mapping Ocean Observation Platforms – Southern Ocean",
                       "https://literacy.s4oceanice.eu/chapters/chapter3/oceanice_platforms.html"),
    "ch04_argo_ctd":  ("CTD data from Argo profiling floats",
                       "https://literacy.s4oceanice.eu/chapters/chapter4/oceanice_argo_floats.html"),
    "ch05_argo_heat": ("Argo Float Observation – Interactive Maps and Profiles",
                       "https://literacy.s4oceanice.eu/chapters/chapter5/oceanice_argo_heatmap.html"),
    "ch06_meop":      ("Animal-Borne Sensors – MEOP",
                       "https://literacy.s4oceanice.eu/chapters/chapter6/oceanice_meop_animal_borne_profiles.html"),
    "ch07_ice":       ("Antarctic Sea Ice Extension – Annual Maxima and Minima",
                       "https://literacy.s4oceanice.eu/chapters/chapter7/oceanice_ice_sheet.html"),
    "ch08_cchdo":     ("Ocean Circulation, Carbon Uptake – CCHDO Bottle",
                       "https://literacy.s4oceanice.eu/chapters/chapter8/oceanice_cchdo_bottle.html"),
    "ch09_bottle":    ("Interactive visualization of CCHDO Bottle Data",
                       "https://literacy.s4oceanice.eu/chapters/chapter9/oceanice_bottle_interactive.html"),
    "ch10_iadc":      ("CTD Explorer – Temperature & Salinity (IADC)",
                       "https://literacy.s4oceanice.eu/chapters/chapter10/oceanice_iadc.html"),
    "ch11_mld":       ("Southern Ocean Mixed Layer Depth – Argo Floats",
                       "https://literacy.s4oceanice.eu/chapters/chapter11/oceanice_mixed_layer_depth.html"),
    "ch12_sealevel":  ("Antarctic Sea Level – Monthly Mean",
                       "https://literacy.s4oceanice.eu/chapters/chapter12/oceanice_sea_level.html"),
    "ch13_cora":      ("Global Ocean Data Explorer – T/S/Currents (CORA)",
                       "https://literacy.s4oceanice.eu/chapters/chapter13/oceanice_cora_analysis.html"),
    "ch14_cora_pt":   ("Gridded T/S Viewer 1960–Present (CORA point)",
                       "https://literacy.s4oceanice.eu/chapters/chapter14/oceanice_cora_point.html"),
    "ch15_glorys":    ("GLORYS12V1 Ocean Potential Temperature – WMS",
                       "https://literacy.s4oceanice.eu/chapters/chapter15/oceanice_wms_bbox.html"),
    "ch16_overlay":   ("In-Situ Temperature Datasets with WMS Overlay",
                       "https://literacy.s4oceanice.eu/chapters/chapter16/oceanice_cora_overlay.html"),
    "ch17_catalogue": ("OCEAN ICE Data Catalogue",
                       "https://literacy.s4oceanice.eu/chapters/chapter17/oceanice_catalogue.html"),
}

# URL GitHub raw dei notebook sorgente (s4oceanice/literacy.s4oceanice)
OCEANICE_NB_BASE = (
    "https://raw.githubusercontent.com/s4oceanice/literacy.s4oceanice/main/"
)
OCEANICE_NB_PATHS = {
    "ch01_tabledap":  "chapters/chapter1/oceanice_erddap_querying_tabledap.ipynb",
    "ch02_griddap":   "chapters/chapter2/oceanice_erddap_querying_griddap.ipynb",
    "ch03_platforms": "chapters/chapter3/oceanice_platforms.ipynb",
    "ch04_argo_ctd":  "chapters/chapter4/oceanice_argo_floats.ipynb",
    "ch05_argo_heat": "chapters/chapter5/oceanice_argo_heatmap.ipynb",
    "ch06_meop":      "chapters/chapter6/oceanice_meop_animal_borne_profiles.ipynb",
    "ch07_ice":       "chapters/chapter7/oceanice_ice_sheet.ipynb",
    "ch08_cchdo":     "chapters/chapter8/oceanice_cchdo_bottle.ipynb",
    "ch09_bottle":    "chapters/chapter9/oceanice_bottle_interactive.ipynb",
    "ch10_iadc":      "chapters/chapter10/oceanice_iadc.ipynb",
    "ch11_mld":       "chapters/chapter11/oceanice_mixed_layer_depth.ipynb",
    "ch12_sealevel":  "chapters/chapter12/oceanice_sea_level.ipynb",
    "ch13_cora":      "chapters/chapter13/oceanice_cora_analysis.ipynb",
    "ch14_cora_pt":   "chapters/chapter14/oceanice_cora_point.ipynb",
    "ch15_glorys":    "chapters/chapter15/oceanice_wms_bbox.ipynb",
    "ch16_overlay":   "chapters/chapter16/oceanice_cora_overlay.ipynb",
    "ch17_catalogue": "chapters/chapter17/oceanice_catalogue.ipynb",
}

# ─── embedding model ────────────────────────────────────────────────────────
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# ─── session state ──────────────────────────────────────────────────────────
for key, default in [
    ("documents", []),
    ("metadatas", []),
    ("messages", []),
    ("last_code", ""),
    ("nb_notebooks_fetched", set()),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─── FAISS helpers ──────────────────────────────────────────────────────────
def rebuild_faiss_index():
    if not st.session_state.documents:
        return
    emb = embedder.encode(
        st.session_state.documents, convert_to_numpy=True
    ).astype(np.float32)
    idx = faiss.IndexFlatL2(emb.shape[1])
    idx.add(emb)
    st.session_state.faiss_index = idx


def get_context(question: str, k: int = 8) -> str:
    if "faiss_index" not in st.session_state or not st.session_state.documents:
        return "Knowledge base vuota. Premi 'Scrape OCEAN:ICE' nella sidebar."
    qvec = embedder.encode([question], convert_to_numpy=True).astype(np.float32)
    _, idxs = st.session_state.faiss_index.search(qvec, k)
    snippets = []
    for i in idxs[0]:
        meta = st.session_state.metadatas[i]
        src = meta.get("source", "?")
        typ = meta.get("type", "text")
        prefix = f"[{typ.upper()} | {src}]\n"
        snippets.append(prefix + st.session_state.documents[i])
    return "\n\n---\n\n".join(snippets)

# ─── scraping ───────────────────────────────────────────────────────────────
def _chunk_text(text: str, chunk_size: int = 900, overlap: int = 150):
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i : i + chunk_size]
        if len(chunk.strip()) > 60:
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def scrape_polarin():
    """Scrapa il sito POLARIN originale."""
    urls = [
        "https://s4polarin.eu/",
        "https://s4polarin.eu/virtual-access/",
        "https://s4polarin.eu/data-catalog/",
    ]
    n = 0
    for url in urls:
        try:
            r = requests.get(url, timeout=20)
            soup = BeautifulSoup(r.text, "html.parser")
            text = soup.get_text(separator="\n", strip=True)
            for chunk in _chunk_text(text):
                st.session_state.documents.append(chunk)
                st.session_state.metadatas.append({"source": url, "type": "web_text"})
                n += 1
        except Exception as e:
            st.warning(f"Errore su {url}: {e}")
    rebuild_faiss_index()
    return n


def scrape_oceanice_literacy(selected_chapters: list | None = None):
    """
    Scrapa i capitoli del literacy book OCEAN:ICE.
    Se selected_chapters è None, scrapa tutti i 17 capitoli.
    """
    chapters = selected_chapters or list(OCEANICE_CHAPTERS.keys())
    n = 0
    progress = st.progress(0)
    for i, ch_key in enumerate(chapters):
        title, url = OCEANICE_CHAPTERS[ch_key]
        try:
            r = requests.get(url, timeout=30)
            soup = BeautifulSoup(r.text, "html.parser")
            # rimuovi nav/footer
            for tag in soup(["nav", "footer", "script", "style"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
            for chunk in _chunk_text(text):
                st.session_state.documents.append(chunk)
                st.session_state.metadatas.append({
                    "source": url,
                    "chapter": ch_key,
                    "chapter_title": title,
                    "type": "literacy_text",
                })
                n += 1
            time.sleep(0.3)  # rate limiting gentile
        except Exception as e:
            st.warning(f"Errore capitolo {ch_key}: {e}")
        progress.progress((i + 1) / len(chapters))
    rebuild_faiss_index()
    return n


def fetch_oceanice_notebook(ch_key: str) -> int:
    """
    Recupera il notebook .ipynb dal repo GitHub s4oceanice
    e ne indicizza code + markdown cells.
    Restituisce il numero di chunk aggiunti.
    """
    if ch_key in st.session_state.nb_notebooks_fetched:
        return 0
    nb_path = OCEANICE_NB_PATHS.get(ch_key)
    if not nb_path:
        return 0
    url = OCEANICE_NB_BASE + nb_path
    try:
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            st.warning(f"Notebook {ch_key} non trovato su GitHub (status {r.status_code})")
            return 0
        nb = r.json()
    except Exception as e:
        st.warning(f"Errore fetching notebook {ch_key}: {e}")
        return 0

    title, _ = OCEANICE_CHAPTERS[ch_key]
    n = 0
    for cell in nb.get("cells", []):
        src = "".join(cell.get("source", []))
        if not src.strip():
            continue
        cell_type = cell.get("cell_type", "code")
        st.session_state.documents.append(src)
        st.session_state.metadatas.append({
            "source": f"github:{nb_path}",
            "chapter": ch_key,
            "chapter_title": title,
            "type": "code_example" if cell_type == "code" else "explanation",
            "language": "python" if cell_type == "code" else "markdown",
        })
        n += 1

    if n > 0:
        rebuild_faiss_index()
        st.session_state.nb_notebooks_fetched.add(ch_key)
    return n

# ─── notebook parser (upload locale) ────────────────────────────────────────
def learn_from_notebook(uploaded_file) -> int:
    nb = json.load(uploaded_file)
    n = 0
    for cell in nb.get("cells", []):
        src = "".join(cell.get("source", []))
        if not src.strip():
            continue
        st.session_state.documents.append(src)
        st.session_state.metadatas.append({
            "source": "upload",
            "type": "code_example" if cell["cell_type"] == "code" else "explanation",
            "language": "python" if cell["cell_type"] == "code" else "markdown",
        })
        n += 1
    if n:
        rebuild_faiss_index()
    return n

# ─── categorie tematiche ────────────────────────────────────────────────────
TOPIC_KEYWORDS = {
    "ERDDAP / tabledap / griddap": [
        "erddap", "tabledap", "griddap", "erddapy", "opendap", "dataset_id",
        "get_var", "constraints", "to_pandas", "to_xarray",
    ],
    "Argo Floats / CTD": [
        "argo", "float", "ctd", "profiling", "profile", "wmo", "temperature",
        "salinity", "pressure", "mixed layer", "mld",
    ],
    "Sea Ice": [
        "sea ice", "ice extent", "ice sheet", "antarctic ice", "nsidc",
        "ice concentration", "sic", "sea_ice",
    ],
    "Animal-Borne / MEOP": [
        "meop", "animal", "seal", "elephant seal", "tag", "animal-borne",
        "biologger",
    ],
    "CCHDO / Bottle Data": [
        "cchdo", "bottle", "woce", "go-ship", "hydrography", "cruise",
        "oxygen", "carbon", "dic",
    ],
    "CORA / Gridded": [
        "cora", "gridded", "climatology", "reanalysis", "cmems", "copernicus",
        "temperature salinity", "depth level",
    ],
    "WMS / GLORYS": [
        "wms", "glorys", "getmap", "bbox", "owslib", "wms layer", "glorys12",
    ],
    "Sea Level": [
        "sea level", "altimetry", "tide gauge", "sl", "ssh", "msla",
    ],
    "Platforms / Maps": [
        "platform", "map", "cartopy", "leaflet", "folium", "buoy", "mooring",
        "glider", "drifter", "soosmap",
    ],
}


def detect_topic(text: str) -> str:
    text_lc = text.lower()
    scores = {}
    for topic, kws in TOPIC_KEYWORDS.items():
        scores[topic] = sum(kw in text_lc for kw in kws)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "General Ocean Data Science"

# ─── Colab notebook builder ──────────────────────────────────────────────────
_COLAB_BADGE = (
    "[![Open In Colab]"
    "(https://colab.research.google.com/assets/colab-badge.svg)]"
    "(https://colab.research.google.com/)"
)

def _extract_code_blocks(text: str) -> list[str]:
    """Estrae blocchi ```python ... ``` dalla risposta del LLM."""
    pattern = r"```(?:python)?\n(.*?)```"
    return re.findall(pattern, text, re.DOTALL)


def build_colab_notebook(
    question: str,
    answer: str,
    topic: str,
    extra_context_code: str = "",
) -> dict:
    """
    Costruisce un dizionario notebook Jupyter (.ipynb) pronto per Colab.
    Struttura:
      - Cella markdown: titolo + badge Colab + domanda
      - Cella markdown: install dipendenze
      - Cella codice: pip install
      - Cella markdown: spiegazione (testo della risposta senza codice)
      - Cella/e codice: codice estratto dalla risposta
      - Cella codice opzionale: contesto aggiuntivo dai notebook OCEAN:ICE
    """
    timestamp = datetime.now().strftime("%Y-%m-%d")

    # Separa testo dal codice nella risposta
    code_blocks = _extract_code_blocks(answer)
    explanation = re.sub(r"```(?:python)?\n.*?```", "", answer, flags=re.DOTALL).strip()

    # Dipendenze comuni rilevate dalle parole chiave
    deps_map = {
        "erddapy": ["erddapy"],
        "xarray": ["xarray", "netcdf4", "h5netcdf"],
        "cartopy": ["cartopy"],
        "folium": ["folium"],
        "plotly": ["plotly"],
        "gsw": ["gsw"],  # TEOS-10
        "owslib": ["owslib"],
        "cmocean": ["cmocean"],
        "seaborn": ["seaborn"],
        "scipy": ["scipy"],
    }
    all_code = "\n".join(code_blocks) + extra_context_code
    needed = []
    for pkg, variants in deps_map.items():
        if any(v in all_code.lower() for v in variants):
            needed.append(pkg)

    pip_line = "!pip install -q " + " ".join(needed) if needed else "# No extra packages needed"

    def md_cell(source: str) -> dict:
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": source.splitlines(keepends=True),
        }

    def code_cell(source: str, collapsed: bool = False) -> dict:
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"collapsed": collapsed},
            "outputs": [],
            "source": source.splitlines(keepends=True),
        }

    cells = []

    # 1. Header
    cells.append(md_cell(
        f"# {topic}\n\n"
        f"{_COLAB_BADGE}\n\n"
        f"**Generated by POLARIN Bot** — {timestamp}\n\n"
        f"---\n\n"
        f"**Query:** {question}\n"
    ))

    # 2. Install
    cells.append(md_cell("## Setup\n\nInstalla le dipendenze necessarie:"))
    cells.append(code_cell(pip_line))

    # 3. Imports standard
    cells.append(md_cell("## Imports"))
    cells.append(code_cell(textwrap.dedent("""\
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import warnings
        warnings.filterwarnings('ignore')
    """)))

    # 4. Spiegazione
    if explanation:
        cells.append(md_cell(f"## Descrizione\n\n{explanation}"))

    # 5. Codice estratto
    if code_blocks:
        cells.append(md_cell("## Codice"))
        for i, block in enumerate(code_blocks, 1):
            if len(code_blocks) > 1:
                cells.append(md_cell(f"### Parte {i}"))
            cells.append(code_cell(block.strip()))
    else:
        # Se non ci sono blocchi espliciti, metti la risposta raw in una cella
        cells.append(md_cell("## Codice generato"))
        cells.append(code_cell(f"# Risposta del modello (formattare se necessario)\n# {answer[:300]}"))

    # 6. Contesto notebook OCEAN:ICE se presente
    if extra_context_code.strip():
        cells.append(md_cell(
            "## Codice di riferimento dai notebook OCEAN:ICE\n\n"
            "_Estratto automaticamente dalla knowledge base._"
        ))
        cells.append(code_cell(extra_context_code.strip()))

    # 7. Footer
    cells.append(md_cell(
        "---\n\n"
        "**Risorse utili:**\n"
        "- [OCEAN:ICE Literacy Book](https://literacy.s4oceanice.eu/intro.html)\n"
        "- [POLARIN Virtual Access](https://s4polarin.eu/virtual-access/)\n"
        "- [EMODnet Physics ERDDAP](https://erddap.emodnet-physics.eu/erddap/)\n"
    ))

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0",
            },
            "colab": {
                "provenance": [],
                "toc_visible": True,
            },
        },
        "cells": cells,
    }
    return notebook


def notebook_download_link(nb: dict, filename: str) -> str:
    """Restituisce un link HTML per il download del notebook."""
    nb_str = json.dumps(nb, indent=2)
    b64 = base64.b64encode(nb_str.encode()).decode()
    href = (
        f'<a href="data:application/json;base64,{b64}" '
        f'download="{filename}">📥 Scarica {filename}</a>'
    )
    return href

# ─── LLM ────────────────────────────────────────────────────────────────────
def generate_response(question: str, context: str, topic: str) -> str:
    if not groq_api_key:
        return "⚠️ Inserisci la Groq API key nella sidebar."
    client = Groq(api_key=groq_api_key)
    system = f"""You are a polar and Southern Ocean data science expert.
The user's question is about: **{topic}**.

Use real patterns from the OCEAN:ICE literacy notebooks (erddapy, xarray, cartopy, pandas,
folium, plotly, gsw, owslib) found in the context.

Rules:
- Always output clean, well-commented, ready-to-run Python code in ```python blocks.
- Prefer erddapy for ERDDAP access; xarray for gridded data; pandas for tabular.
- Add !pip install cells only when strictly necessary.
- If the context contains relevant notebook code, reuse and adapt it.
- Explain briefly what the code does before each block.
- The code must be Google Colab-compatible (no local file dependencies).
"""
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": f"Context (notebook code + docs):\n{context}\n\nQuestion: {question}",
            },
        ],
        temperature=0.25,
        max_tokens=3000,
    )
    return resp.choices[0].message.content


def get_extra_context_code(metadatas_subset: list) -> str:
    """Filtra i chunk di tipo code_example dal contesto recuperato."""
    idx_docs = st.session_state.documents
    idx_meta = st.session_state.metadatas
    code_chunks = [
        idx_docs[i]
        for i, m in enumerate(idx_meta)
        if m.get("type") == "code_example" and len(idx_docs[i]) > 80
    ]
    # prendi i primi 3 più rilevanti (già ordinati per similarità)
    return "\n\n# ─────────────────────────────\n\n".join(code_chunks[:3])

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Impostazioni")
    groq_api_key = st.secrets.get("GROQ_API_KEY", "") or st.text_input(
        "Groq API Key", type="password"
    )

    st.divider()
    st.subheader("📚 Knowledge Base")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🌐 Scrape POLARIN"):
            with st.spinner("Scraping s4polarin.eu…"):
                n = scrape_polarin()
                st.success(f"+{n} chunks POLARIN")

    with col2:
        if st.button("🧊 Scrape OCEAN:ICE"):
            with st.spinner("Scraping tutti i 17 capitoli…"):
                n = scrape_oceanice_literacy()
                st.success(f"+{n} chunks OCEAN:ICE")

    st.caption("Oppure seleziona capitoli specifici:")
    ch_options = {f"{k}: {v[0][:40]}…": k for k, v in OCEANICE_CHAPTERS.items()}
    selected_labels = st.multiselect(
        "Capitoli da scrapare", list(ch_options.keys()), label_visibility="collapsed"
    )
    if st.button("📖 Scrape capitoli selezionati") and selected_labels:
        selected_keys = [ch_options[l] for l in selected_labels]
        with st.spinner(f"Scraping {len(selected_keys)} capitoli…"):
            n = scrape_oceanice_literacy(selected_keys)
            st.success(f"+{n} chunks")

    st.divider()
    st.subheader("🐙 Notebook GitHub OCEAN:ICE")
    nb_options = {f"{k}: {v[0][:35]}…": k for k, v in OCEANICE_CHAPTERS.items()}
    nb_selected = st.multiselect(
        "Carica notebook dal repo", list(nb_options.keys()), label_visibility="collapsed"
    )
    if st.button("⬇️ Fetch notebook selezionati") and nb_selected:
        total = 0
        for label in nb_selected:
            ch_key = nb_options[label]
            with st.spinner(f"Fetching {ch_key}…"):
                n = fetch_oceanice_notebook(ch_key)
                total += n
        st.success(f"+{total} chunk da {len(nb_selected)} notebook")

    st.divider()
    st.subheader("📂 Upload notebook locale")
    uploaded_file = st.file_uploader("Carica .ipynb", type=["ipynb"])
    if uploaded_file and st.button("📘 Impara dal notebook"):
        n = learn_from_notebook(uploaded_file)
        st.success(f"+{n} chunks dal notebook caricato")

    st.divider()
    if st.button("🗑️ Svuota knowledge base"):
        st.session_state.documents = []
        st.session_state.metadatas = []
        st.session_state.nb_notebooks_fetched = set()
        if "faiss_index" in st.session_state:
            del st.session_state.faiss_index
        st.success("Knowledge base azzerata.")

    st.divider()
    n_docs = len(st.session_state.documents)
    n_code = sum(1 for m in st.session_state.metadatas if m.get("type") == "code_example")
    st.metric("Chunk totali", n_docs)
    st.metric("Di cui codice", n_code)

# ─── TAB LAYOUT ──────────────────────────────────────────────────────────────
tab_chat, tab_colab, tab_info = st.tabs(["💬 Chat", "🚀 Colab Builder", "ℹ️ Info capitoli"])

# ── TAB CHAT ─────────────────────────────────────────────────────────────────
with tab_chat:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input(
        "Es: Plot temperature profiles from Argo floats near Antarctica"
    ):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        topic = detect_topic(prompt)
        with st.chat_message("assistant"):
            with st.spinner(f"🔍 Topic rilevato: **{topic}** — generando risposta…"):
                context = get_context(prompt)
                answer = generate_response(prompt, context, topic)
                st.markdown(f"**🏷️ Topic:** `{topic}`\n\n{answer}")

                # Salva ultimo codice per il Colab Builder
                code_blocks = _extract_code_blocks(answer)
                if code_blocks:
                    st.session_state.last_code = "\n\n".join(code_blocks)
                    st.session_state.last_question = prompt
                    st.session_state.last_topic = topic
                    st.session_state.last_answer = answer
                    st.info(
                        "💡 Vai alla tab **🚀 Colab Builder** per scaricare il notebook eseguibile!"
                    )

        st.session_state.messages.append({"role": "assistant", "content": answer})

# ── TAB COLAB BUILDER ────────────────────────────────────────────────────────
with tab_colab:
    st.header("🚀 Colab Notebook Generator")
    st.write(
        "Genera un notebook `.ipynb` eseguibile su Google Colab a partire "
        "dall'ultima risposta del bot, oppure da una domanda libera qui sotto."
    )

    col_a, col_b = st.columns([3, 1])
    with col_a:
        colab_query = st.text_area(
            "Domanda per il notebook",
            value=st.session_state.get("last_question", ""),
            height=100,
            placeholder="Es: Retrieve Argo float profiles from ERDDAP and plot T/S diagram",
        )
    with col_b:
        nb_filename = st.text_input("Nome file", value="polarin_notebook.ipynb")
        include_context_code = st.checkbox("Includi codice OCEAN:ICE", value=True)

    generate_colab = st.button("⚡ Genera Notebook Colab", type="primary")

    if generate_colab and colab_query:
        topic = detect_topic(colab_query)
        st.info(f"🏷️ Topic rilevato: **{topic}**")

        with st.spinner("Recupero contesto e genero il notebook…"):
            context = get_context(colab_query)
            answer = st.session_state.get("last_answer", "")
            if not answer or colab_query != st.session_state.get("last_question", ""):
                answer = generate_response(colab_query, context, topic)

            extra_code = ""
            if include_context_code:
                extra_code = get_extra_context_code(st.session_state.metadatas)

            nb = build_colab_notebook(colab_query, answer, topic, extra_code)

        st.success("✅ Notebook generato!")
        dl_link = notebook_download_link(nb, nb_filename)
        st.markdown(dl_link, unsafe_allow_html=True)

        # Anteprima celle
        with st.expander("📋 Anteprima celle del notebook"):
            for i, cell in enumerate(nb["cells"]):
                src = "".join(cell["source"])
                if cell["cell_type"] == "markdown":
                    st.markdown(src)
                else:
                    st.code(src, language="python")
                if i < len(nb["cells"]) - 1:
                    st.divider()

    elif generate_colab and not colab_query:
        st.warning("Inserisci una domanda per generare il notebook.")

    # Colab link diretto ai notebook OCEAN:ICE originali
    st.divider()
    st.subheader("📘 Notebook OCEAN:ICE originali su Colab")
    st.write("Apri direttamente i notebook sorgente del literacy book su Google Colab:")
    colab_base = "https://colab.research.google.com/github/s4oceanice/literacy.s4oceanice/blob/main/"
    cols = st.columns(2)
    for i, (ch_key, (ch_title, _)) in enumerate(OCEANICE_CHAPTERS.items()):
        nb_path = OCEANICE_NB_PATHS.get(ch_key, "")
        if nb_path:
            url = colab_base + nb_path
            cols[i % 2].markdown(f"[🔗 {ch_title[:55]}]({url})")

# ── TAB INFO ─────────────────────────────────────────────────────────────────
with tab_info:
    st.header("📚 Capitoli OCEAN:ICE Literacy Book")
    for ch_key, (title, url) in OCEANICE_CHAPTERS.items():
        fetched = ch_key in st.session_state.nb_notebooks_fetched
        badge = "✅" if fetched else "⬜"
        with st.expander(f"{badge} **{ch_key}** — {title}"):
            st.markdown(f"**URL:** [{url}]({url})")
            colab_url = (
                "https://colab.research.google.com/github/s4oceanice/"
                f"literacy.s4oceanice/blob/main/{OCEANICE_NB_PATHS.get(ch_key, '')}"
            )
            st.markdown(f"**Colab:** [{colab_url}]({colab_url})")
            if st.button(f"⬇️ Fetch notebook {ch_key}", key=f"fetch_{ch_key}"):
                n = fetch_oceanice_notebook(ch_key)
                st.success(f"+{n} chunk aggiunti dalla knowledge base")

    st.divider()
    st.subheader("🏷️ Topic classifier")
    test_q = st.text_input("Testa il rilevamento topic:", "plot sea ice extent from NSIDC")
    if test_q:
        st.info(f"Topic rilevato: **{detect_topic(test_q)}**")
