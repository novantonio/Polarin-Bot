"""
POLARIN ERDDAP Scientific Assistant
====================================
Versione completa con analisi grafiche e report PDF
"""

import os
import io
import gc
import warnings
from datetime import datetime

import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from groq import Groq

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak, HRFlowable
from reportlab.lib.enums import TA_CENTER

warnings.filterwarnings("ignore")

# ====================== CONFIG ======================
st.set_page_config(page_title="POLARIN Assistant", page_icon="❄️", layout="wide")
st.title("❄️ POLARIN ERDDAP Scientific Assistant")
st.caption("Analisi dataset oceanografici polari • s4polarin.eu")

ERDDAP_BASE = "https://erddap.s4polarin.eu/erddap"
PDF_OUTPUT = "polarin_report.pdf"

# ====================== GROQ CLIENT ======================
client = Groq(api_key=os.getenv("GROQ_API_KEY"))  # imposta la variabile d'ambiente o mettila qui

# ====================== ERDDAP FUNCTIONS ======================
@st.cache_data(ttl=3600)
def search_datasets(keyword: str = ""):
    url = f"{ERDDAP_BASE}/search/index.csv?&searchFor={keyword}"
    try:
        df = pd.read_csv(url)
        df = df[df['Dataset ID'].notna() & (df['Dataset ID'] != 'Dataset ID')].reset_index(drop=True)
        return df[['Title', 'Dataset ID']]
    except:
        return pd.DataFrame()

@st.cache_data
def download_dataset(dataset_id: str):
    url = f"{ERDDAP_BASE}/tabledap/{dataset_id}.csv?&time>=1900-01-01"
    st.info(f"📥 Scaricando {dataset_id}...")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), skiprows=[1])
    df.columns = [c.lower() for c in df.columns]
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
    return df

# ====================== PLOTTING ======================
def elabora_visualizzazioni_dataset(df_data):
    figures = []
    exclude = {"time", "latitude", "longitude", "depth", "station", "id", "platformcode"}
    params = [col for col in df_data.columns if col not in exclude and not col.endswith("_qc")]
    
    if not params:
        st.warning("Nessun parametro numerico trovato.")
        return figures

    df = df_data.copy()
    has_depth = "depth" in df.columns and df["depth"].nunique() > 1
    if not has_depth:
        df["depth"] = 1.0

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["month"] = df["time"].dt.month

    has_coords = all(c in df.columns for c in ["latitude", "longitude"])

    for param in params[:3]:   # limite per performance
        fig, axs = plt.subplots(2, 3, figsize=(18, 9), gridspec_kw={"hspace": 0.4, "wspace": 0.3})
        fig.suptitle(f"{param.upper()} — {df.get('platformcode', pd.Series(['Dataset'])).iloc[0]}", 
                     fontsize=14, fontweight="bold")

        # 1. Mappa
        if has_coords:
            ax = plt.subplot(2, 3, 1, projection=ccrs.PlateCarree())
            coords = df[["latitude", "longitude"]].dropna()
            ax.scatter(coords["longitude"], coords["latitude"], s=8, color="red", alpha=0.6, transform=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.LAND, facecolor="#eeeeee")
            ax.set_title("Posizione")

        # 2. Time series
        ax = axs[0, 1]
        ax.scatter(df["time"], df[param], s=6, alpha=0.7)
        ax.set_title("Serie Temporale")
        ax.grid(True, alpha=0.3)

        # 3. Media mensile
        ax = axs[1, 1]
        monthly = df.groupby("month")[param].mean()
        ax.plot(monthly.index, monthly.values, marker='o')
        ax.set_title("Media Mensile")
        ax.set_xticks(range(1,13))
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        figures.append((fig, param, "Platform"))
        st.pyplot(fig)
        plt.close(fig)

    return figures

# ====================== PDF REPORT ======================
def _fig_to_rl_image(fig, width_cm=16):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return RLImage(buf, width=width_cm*cm)

def genera_report_pdf(record_list):
    doc = SimpleDocTemplate(PDF_OUTPUT, pagesize=A4, leftMargin=1.5*cm, rightMargin=1.5*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    
    story = []
    story.append(Paragraph("POLARIN Dataset Analysis Report", styles["Title"]))
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"Generato il {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
    story.append(HRFlowable(width="100%", thickness=2))
    story.append(Spacer(1, 30))

    for rec in record_list:
        story.append(Paragraph(f"Dataset: {rec['dataset_id']}", styles["Heading1"]))
        story.append(Paragraph(rec.get('title', ''), styles["Normal"]))
        story.append(Spacer(1, 10))
        
        for fig, param, _ in rec.get("figures", []):
            story.append(Paragraph(f"Parametro: {param.upper()}", styles["Heading2"]))
            story.append(_fig_to_rl_image(fig))
            story.append(Spacer(1, 15))

        story.append(PageBreak())

    doc.build(story)
    return PDF_OUTPUT

# ====================== LLM HELPER ======================
def genera_domande_scientifiche():
    prompt = """Sei un ricercatore polare esperto del progetto POLARIN.
    Genera 6 domande scientifiche interessanti, indagabili con dataset ERDDAP di s4polarin.eu.
    Focalizzati su: temperatura, salinità, ghiaccio marino, profili CTD, trend climatici, interazioni oceano-cryosfera."""
    
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=800
    )
    return response.choices[0].message.content

# ====================== STREAMLIT UI ======================
tab1, tab2, tab3 = st.tabs(["🔍 Esplora Dataset", "📊 Analisi", "🤖 Assistente AI"])

with tab1:
    col1, col2 = st.columns([3,1])
    with col1:
        keyword = st.text_input("Cerca dataset", value="polarin")
    if st.button("Cerca"):
        df = search_datasets(keyword)
        if not df.empty:
            st.dataframe(df, use_container_width=True)
            st.session_state.datasets = df
        else:
            st.warning("Nessun dataset trovato.")

with tab2:
    dataset_id = st.text_input("Dataset ID da analizzare", placeholder="es. POLARIN_SOME_DATASET")
    if st.button("Analizza e genera grafici"):
        if dataset_id:
            try:
                df = download_dataset(dataset_id)
                st.success(f"Caricate {len(df)} righe")
                
                figures = elabora_visualizzazioni_dataset(df)
                
                record = {
                    "dataset_id": dataset_id,
                    "title": dataset_id,
                    "figures": figures,
                    "n_records": len(df)
                }
                
                if st.button("📄 Genera Report PDF"):
                    pdf_path = genera_report_pdf([record])
                    with open(pdf_path, "rb") as f:
                        st.download_button("Scarica Report PDF", f, file_name=pdf_path)
            except Exception as e:
                st.error(f"Errore: {e}")
        else:
            st.warning("Inserisci un Dataset ID")

with tab3:
    st.subheader("Assistente Scientifico")
    if st.button("💡 Genera domande di ricerca PolarIn"):
        with st.spinner("Pensando..."):
            domande = genera_domande_scientifiche()
            st.markdown(domande)
    
    query = st.text_area("Fai una domanda scientifica", 
        "Quali trend di temperatura si osservano nel Mare di Ross?")
    
    if st.button("Invia domanda"):
        st.info("L'LLM analizzerà la domanda e suggerirà dataset (funzionalità in espansione).")

st.caption("POLARIN ERDDAP Assistant • Basato su ASPIM_analysis.ipynb")
