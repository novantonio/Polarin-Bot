"""
POLARIN ERDDAP Scientific Assistant
====================================
English version - Optimized for Streamlit Cloud
"""

import os
import io
import warnings
from datetime import datetime

import requests
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from groq import Groq

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak, HRFlowable

warnings.filterwarnings("ignore")

# ====================== CONFIG ======================
st.set_page_config(page_title="POLARIN Assistant", page_icon="❄️", layout="wide")
st.title("❄️ POLARIN ERDDAP Scientific Assistant")
st.caption("Polar oceanographic data analysis • s4polarin.eu")

ERDDAP_BASE = "https://erddap.s4polarin.eu/erddap"
PDF_OUTPUT = "polarin_report.pdf"

# ====================== GROQ CLIENT ======================
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ====================== ERDDAP FUNCTIONS ======================
@st.cache_data(ttl=3600)
def search_datasets(keyword: str = "polarin"):
    """Search datasets on PolarIn ERDDAP server"""
    url = f"{ERDDAP_BASE}/search/index.csv?&searchFor={keyword}"
    try:
        df = pd.read_csv(url)
        df = df[df['Dataset ID'].notna() & (df['Dataset ID'] != 'Dataset ID')].reset_index(drop=True)
        return df[['Title', 'Dataset ID']]
    except:
        st.error("Failed to connect to ERDDAP server")
        return pd.DataFrame()

@st.cache_data
def download_dataset(dataset_id: str):
    """Download dataset from ERDDAP"""
    url = f"{ERDDAP_BASE}/tabledap/{dataset_id}.csv?&time>=1900-01-01"
    st.info(f"📥 Downloading {dataset_id}...")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), skiprows=[1])
    df.columns = [c.lower() for c in df.columns]
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
    return df

# ====================== PLOTTING (No Cartopy) ======================
def generate_plots(df_data):
    """Generate analysis plots"""
    figures = []
    exclude = {"time", "latitude", "longitude", "depth", "station", "id", "platformcode"}
    params = [col for col in df_data.columns if col not in exclude and not col.endswith("_qc")]
    
    if not params:
        st.warning("No numeric parameters found in this dataset.")
        return figures

    df = df_data.copy()
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")

    for param in params[:3]:
        fig, axs = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle(f"{param.upper()} — {df.get('platformcode', pd.Series(['Dataset'])).iloc[0]}", 
                     fontsize=14, fontweight="bold")

        # Time Series
        ax = axs[0, 0]
        ax.scatter(df["time"], df[param], s=8, alpha=0.7)
        ax.set_title("Time Series")
        ax.grid(True, alpha=0.3)

        # Daily Average
        ax = axs[0, 1]
        if "time" in df.columns:
            daily = df.set_index("time")[param].resample("D").mean()
            daily.plot(ax=ax)
            ax.set_title("Daily Average")
            ax.grid(True, alpha=0.3)

        # Histogram
        ax = axs[1, 0]
        ax.hist(df[param].dropna(), bins=30, alpha=0.7, color="teal")
        ax.set_title("Distribution")

        # Monthly Boxplot
        ax = axs[1, 1]
        if "time" in df.columns:
            df["month"] = df["time"].dt.month
            df.boxplot(column=param, by="month", ax=ax)
            ax.set_title("Monthly Boxplot")
            ax.set_xlabel("Month")

        plt.tight_layout()
        figures.append((fig, param, "Platform"))
        st.pyplot(fig)
        plt.close(fig)

    return figures

# ====================== PDF REPORT ======================
def generate_pdf_report(record_list):
    """Generate PDF report"""
    doc = SimpleDocTemplate(PDF_OUTPUT, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("POLARIN Dataset Analysis Report", styles["Title"]))
    story.append(Spacer(1, 20))

    for rec in record_list:
        story.append(Paragraph(f"Dataset: {rec['dataset_id']}", styles["Heading1"]))
        for fig, param, _ in rec.get("figures", []):
            story.append(Paragraph(f"Parameter: {param.upper()}", styles["Heading2"]))
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            buf.seek(0)
            story.append(RLImage(buf, width=16*cm))
            story.append(Spacer(1, 15))
        story.append(PageBreak())

    doc.build(story)
    return PDF_OUTPUT

# ====================== MAIN INTERFACE ======================
tab1, tab2, tab3 = st.tabs(["🔍 Dataset Explorer", "📊 Analysis", "🤖 AI Assistant"])

with tab1:
    st.subheader("Search PolarIn Datasets")
    if st.button("Search All Datasets"):
        df = search_datasets("")
        if not df.empty:
            st.dataframe(df, use_container_width=True)
            st.session_state.datasets = df

with tab2:
    st.subheader("Dataset Analysis")
    dataset_id = st.text_input("Enter Dataset ID", placeholder="e.g. ASPIM_PR_CNDC_Portofino")
    
    if st.button("Analyze Dataset"):
        if dataset_id:
            try:
                df = download_dataset(dataset_id)
                st.success(f"✅ Loaded {len(df):,} records")
                
                figures = generate_plots(df)
                
                if st.button("📄 Generate PDF Report"):
                    record = {"dataset_id": dataset_id, "figures": figures}
                    pdf_path = generate_pdf_report([record])
                    with open(pdf_path, "rb") as f:
                        st.download_button("Download PDF Report", f, file_name=pdf_path)
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Please enter a Dataset ID")

with tab3:
    st.subheader("AI Scientific Assistant")
    if st.button("Generate Scientific Questions"):
        with st.spinner("Thinking..."):
            prompt = "Generate 6 interesting scientific questions about polar regions using ERDDAP oceanographic datasets."
            response = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[{"role": "user", "content": prompt}]
            )
            st.markdown(response.choices[0].message.content)

st.caption("POLARIN ERDDAP Assistant | Optimized for Streamlit Cloud")
