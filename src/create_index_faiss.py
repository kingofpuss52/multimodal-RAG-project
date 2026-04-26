import os
import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

class MedicalIndexer:
  """
  Mengatur pembuatan database vector FAISS.
  Integrasi hyperparameter yang telah dioptimasi.
  """
  def __init__(self):
    # gunakan path relatif
    self.root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    self.kb_paths = [
      os.path.join(self.root_dir, "data/knowledge_base.txt"),
      os.path.join(self.root_dir, "data/pubmed_knowledge.txt")
    ]
    self.config_path = os.path.join(self.root_dir, "configs/best_params.json")
    self.index_save_path = os.path.join(self.root_dir, "data/faiss_index")
    
    # set parameter default jika terjadi fallback
    self.default_params = {"chunk_size": 600, "chunk_overlap": 100}
    
  def _load_optimal_params(self):
    """load parameter yang telah dioptimasi. jika tidak ada, set parameter menjadi default"""
    if os.path.exists(self.config_path):
      try:
        with open(self.config_path, "r") as f:
          return json.load(f)
        
      except Exception as e:
        print(f"[Warning] Tidak dapat membaca konfigurasi terbaik: {e}. Menggunakan konfigurasi Default.")
        
    return self.default_params
    
    
  def _load_documents_content(self):
    """Kumpulkan teks dari file knowledge base (database)"""
    all_content = ""
    
    for path in self.kb_paths:
      if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
          all_content += f.read() + "\n\n"
          
    return all_content
  
  def run_indexing(self):
    """Jalankan indexing dari knowledge base"""
    print("Menjalankan Indexing Knowledge...")
    
    # load parameter
    params = self._load_optimal_params()
    
    # kumpulkan dokumen dan chunking
    content = self._load_documents_content()
    
    if not content:
      print("[Error] File knowledge base tidak ditemukan. Periksa folder data.")
      return
    
    # potong potong teks (chunking) agar pas untuk LLM
    text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=params.get("chunk_size"), 
      chunk_overlap=params.get("chunk_overlap"),
      separators=["---", "\n\n", ".", " ", ""]
    )
    docs = text_splitter.create_documents([content])
  
    # 3. inisialisasi model embedding
    print(f"Generate embedding untuk {len(docs)} chunks...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
  
    # 4. buat vector db, simpan secara lokal
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(self.index_save_path)
  
    print(f"Indexing selesai. Vector index disimpan di {self.index_save_path}")
  
if __name__ == "__main__":
  indexer = MedicalIndexer()
  indexer.run_indexing()
  