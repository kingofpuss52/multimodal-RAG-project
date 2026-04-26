import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import torch
import streamlit as st
import time
from PIL import Image

from src.vision_engine import VisionEngine
from src.rag_engine import RAGEngine
from src.mlflow_utils import MedicalTracking

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# inisialisasi tracker eksperimen
tracker = MedicalTracking()

# konfigurasi halaman
st.set_page_config(
  page_title="NeuroScan AI - Brain Tumor Analysis",
  page_icon="🧠",
  layout="wide"
)

# konfigurasi path secara dinamis
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT_DIR, "models")
DATA_DIR = os.path.join(ROOT_DIR, "data")
FAISS_PATH = os.path.join(DATA_DIR, "faiss_index")

# load engine
@st.cache_resource
def load_engines():
  """load model inferensi dan RAG engine untuk optimasi performa"""
  vision_model_path = os.path.join(MODELS_DIR, "model_terbaik_final_Bayesian.keras")
  vision = VisionEngine(vision_model_path)
  
  # load RAG engine
  rag = RAGEngine()
  
  if os.path.exists(FAISS_PATH):
    # pakai model embedding yang sama waktu build database
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # load vector db
    vector_db = FAISS.load_local(
      FAISS_PATH,
      embeddings,
      allow_dangerous_deserialization=True
    )
    
    # masukkan database ke dalam RAG engine
    rag.vector_db = vector_db
    
  else:
    st.error(f"Folder '{FAISS_PATH}' tidak ditemukan. Jalankan script indexing dahulu.")
  
  return vision, rag

# menu sidebar dan disclaimer
with st.sidebar:
  st.image("https://cdn-icons-png.flaticon.com/512/2491/2491281.png", width=100)
  st.title("NeuroScan AI")
  st.markdown("---")
  
  st.header("Informasi Pengembang")
  st.write("**Nathasya Utami Hakim**")
  st.markdown("---")
  
  # buat disclaimer
  st.warning("**Medical Disclaimer**")
  st.caption("""
    Sistem AI ini dikembangkan dengan bantuan keputusan (**Decision Support System**). Sistem ini **Bukan merupakan diagnosa medis resmi.**
    
    Harap verifikasi hasil dengan ahli radiologi atau dokter spesialis sebelum melakukan tindakan klinis.
  """
  )
  st.divider()
  st.info("Status Sistem: **Stabil (Hybrid CPU/GPU)**")
  
# interface utama
st.title("Sistem Analisis MRI MultiModal (RAG)")
st.write("Klasifikasi Tumor Otak (EfficientNetB5) dan  Otomasi Pelaporan Klinis (Qwen2.5-3B)")

# inisialisasi engine
try:
  with st.spinner("Menginisialisasi model AI..."):
    vision_engine, rag_engine = load_engines()
    
  st.toast("Sistem siap digunakan.", icon="✅")
  
except Exception as e:
  st.error(f"Gagal memuat sistem: {e}")
  
# upload gambar MRI
uploaded_file = st.file_uploader("Pilih Gambar MRI (format .jpg, .png, .jpeg)...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
  # buat 2 kolom
  col1, col2 = st.columns([1, 1])
  
  with col1:
    st.subheader("Gambar MRI Input")
    image = Image.open(uploaded_file)
    st.image(image, use_column_width=True, caption="Gambar yang telah diupload")
    
  with col2:
    st.subheader("Hasil Analisis AI")
    
    # jalankan prediksi dari model klasifikasi
    with st.spinner("Menganalisis gambar..."):
      start_pred = time.time()
      prediction = vision_engine.predict(image)
      pred_time = time.time() - start_pred
      
    label = prediction['label']
    confidence = prediction['confidence']
    
    # tampilkan hasil diagnosa
    if label in ["Normal", "No Tumor"]:
      st.success(f"**Diagnosa: {label}**")
    
    else:
      st.error(f"**Diagnosa: {label}**")
      
    st.metric("Confidence Score", f"{confidence:.2%}")
    st.progress(confidence)
    
    # jalankan RAG report
    st.markdown("---")
    st.subheader("Integrated Clinical Analysis")
    
    with st.spinner("Generate ringkasan klinis (RAG)..."):
      start_rag = time.time()
      report = rag_engine.query_clinical_report(label, confidence)
      rag_time = time.time() - start_rag
      
    st.info(report)
    
    with st.expander("Lihat Detail"):
      st.write(f"Vision Inference: `{pred_time:.2f}s`")
      st.write(f"RAG Generation: `{rag_time:.2f}s`")
      st.write(f"Device: `NVIDIA GeForce RTX 3050 Laptop GPU (4-bit Quantized)`")
    
    # log ke mlflow
    tracker.log_full_analysis(
      vision_results={"label": label, "confidence": confidence, "time": pred_time},
      rag_results={"report": report, "time": rag_time, "context_len": 800},
      hardware_info={"device": "NVIDIA GeForce RTX 3050 Laptop GPU"}
    )
    
# menu footer
st.markdown("---")
st.markdown(
  "<center><small>© 2026 Nathasya Utami Hakim - Brain Tumor Classification Multimodal Project</small></center>", unsafe_allow_html=True
)