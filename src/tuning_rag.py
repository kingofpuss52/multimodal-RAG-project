import optuna
import mlflow
import os
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

class RAGTuner:
  """
  Tuning hyperparameter untuk proses RAG menggunakan Optuna, implementasi MLOps menggunakan MLflow.
  Mengoptimasi chunk size dan overlap berdasarkan akurasi keyword retrieval.
  """
  def __init__(self, data_path=None):
    self.data_path = data_path or [
      "data/knowledge_base.txt",
      "data/pubmed_knowledge.txt"
    ]
    self.experiment_name = "RAG_Hyperparameter_Tuning"
    
  def _load_raw_data(self):
    """Load dan gabungkan data teks dari database dokumen"""
    content = ""
    
    for path in self.data_path:
      if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
          content += f.read() + "\n\n"
          
    return content
  
  def evaluate_retrieval(self, vector_db, query, expected_keyword):
    """hitung kualitas retrieval berdasarkan keyword yang muncul"""
    results = vector_db.similarity_search_with_score(query, k=3)
    score = 0
    
    for doc, dist in results:
      if expected_keyword.lower() in doc.page_content.lower():
        # semakin kecil jarak, semakin besar skor
        score += (1 / (dist + 1e-6)) 
    return score
    
  def objective(self, trial):
    """setup hyperparameter yang akan dituning"""
    chunk_size = trial.suggest_int("chunk_size", 300, 800, step=100)
    chunk_overlap = trial.suggest_int("chunk_overlap", 50, 150, step=50)
    
    # track experiment dengan mlflow
    with mlflow.start_run(nested=True):
      raw_text = self._load_raw_data()
      text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
      )
      docs = text_splitter.create_documents([raw_text])
      
      # build temporary index agar VRAM aman selama tuning
      embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
      db = FAISS.from_documents(docs, embeddings)
      
      # jalankan test query
      test_queries = [
        {"q": "Bagaimana protokol penanganan Glioma?", "key": "radioterapi"},
        {"q": "Apa ciri-ciri Meningioma di MRI?", "key": "dural tail"},
        {"q": "Apa itu Pituitary Adenoma?", "key": "hormon"}
      ]
      
      total_score = sum(self.evaluate_retrieval(db, t['q'], t['key']) for t in test_queries)
      avg_score = total_score / len(test_queries)
      
      # log hasil metrik ke mlflow
      mlflow.log_params(trial.params)
      mlflow.log_metric("retrieval_score", avg_score)
      
      return avg_score
  
  def run_tuner(self, n_trials=10):
    """Jalankan Optuna dan simpan hasil ke dalam JSON"""
    mlflow.set_experiment(self.experiment_name)
    study = optuna.create_study(direction="maximize")
    study.optimize(self.objective, n_trials=n_trials)
    
    print(f"Optimasi berhasil. Hasil Terbaik {study.best_value:.4f}")
    
    # simpan hasil terbaik
    config_path = "configs/best_params.json"
    os.makedirs("configs", exist_ok=True)
    
    with open(config_path, "w") as f:
      json.dump(study.best_params, f, indent=4)
      
    print(f"Parameter terbaik disimpan ke dalam {config_path}")
    
    
if __name__ == "__main__":
  tuner = RAGTuner()
  tuner.run_tuner(n_trials=10)