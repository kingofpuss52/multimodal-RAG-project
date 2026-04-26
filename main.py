import os
import time
from PIL import Image
from src.vision_engine import VisionEngine
from src.rag_engine import RAGEngine
from src.mlflow_utils import MedicalTracking
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def run_multimodal_rag():
  # konfigurasi path secara dinamis
  ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
  MODEL_PATH = os.path.join(ROOT_DIR, "models/model_terbaik_final_Bayesian.keras")
  INDEX_PATH = os.path.join(ROOT_DIR, "data/faiss_index")
  
  # path gambar untuk uji coba
  IMAGE_TEST = os.path.join(ROOT_DIR, "data/image (52).jpg")
  
  # inisialisasi tracking eksperimen
  tracker = MedicalTracking(experiment_name="Multimodal-RAG-CLI-Test")
  
  print("Inisialisasi Sistem Hybrid (Inferensi CPU | LLM GPU)..")
    
  # inisialisasi vision engine (model inferensi)
  vision = VisionEngine(MODEL_PATH)
  dummy_img = Image.new('RGB', (224, 224))
  vision.predict(dummy_img) 
  print("Vision Engine siap.")
    
  # inisialisasi RAG engine
  rag = RAGEngine()
    
  # load vector db (FAISS)
  if os.path.exists(INDEX_PATH):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    rag.setup_knowledge_base(vector_db)
    
    print("RAG Engine dan Vector DB siap digunakan.")
    
  else:
    print(f"Error: Index FAISS tidak ditemukan di {INDEX_PATH}")
    return
    
  # eksekusi pipeline multimidal
  if os.path.exists(IMAGE_TEST):
    print(f"Menganalisis gambar MRI: {os.path.basename(IMAGE_TEST)}")
    
    # 1. prediksi visual
    start_pred = time.time()
    vision_results = vision.predict(IMAGE_TEST)
    pred_time = time.time() - start_pred
    
    diagnosis = vision_results['label']
    confidence = vision_results['confidence']
    
    # 2. generate laporan dari RAG
    print(f"Generate laporan klinis untuk: {diagnosis}...")
    start_rag = time.time()
    clinical_report = rag.query_clinical_report(diagnosis, confidence)
    rag_time = time.time() - start_rag
    
    # tampilkan hasil
    print("\n" + "=" * 60)
    print("           INTEGRATED CLINICAL ANALYSIS REPORT           ")
    print("-" * 60)
    print(f"DIAGNOSIS    : {diagnosis}")
    print(f"CONFIDENCE   : {confidence:.2%}")
    print(f"ANALYSIS     :\n{clinical_report}")
    print("-" * 60)
    print(f"PERFORMANCE  : Vision {pred_time:.2f}s | RAG {rag_time:.2f}s")
    print("=" * 60 + "\n")
    
    # catat hasilnya ke mlflow melalui class MedicalTracker
    tracker.log_full_analysis(
      vision_results={
        "label": diagnosis,
        "confidence": confidence,
        "time": pred_time
      },
      rag_results={
        "report": clinical_report,
        "time": rag_time,
        "context_len": 800
      },
      hardware_info={"device": "NVIDIA GeForce RTX 3050 Laptop GPU"}
    )
    
    print(f"Data analisis berhasil dicatat ke MLflow.")
    
  else:
    print(f"Gambar uji coba tidak ditemukan di {IMAGE_TEST}.")
    
    
if __name__ == "__main__":
  run_multimodal_rag()