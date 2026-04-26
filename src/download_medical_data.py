import os
from datasets import load_dataset
from typing import Dict, List

class DownloadMedicaldata:
  """
  Mengambil dan filter dataset PubMed QA secara otomatis,
  bertujuan untuk membuat knowledge dengan domain spesifik terhadap RAG.
  """
  def __init__(self, output_filename: str="pubmed_knowledge.txt"):
    # konfigurasi relative path
    self.root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    self.output_path = os.path.join(self.root_dir, "data", output_filename)
    
    # pastikan folder 'data' ada
    os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
    
  def _format_text(self, item: Dict, label: str) -> str:
    """Seragamkan format teks untuk indexing"""
    return (
      f"CATEGORY: {label}\n"
      f"MEDICAL CONTEXT: {item['context']['contexts'][0]}\n"
      f"CLINICAL QUESTION: {item['question']}\n"
      f"EXPERT ANSWER: {item['long_answer']}"
    )

  def download_and_filter(self, limit_per_class: int=50):
    """mulai download dataset dari PubMed QA dan filter berdasarkan keyword tumor otak dan otak normal"""
    
    print(f"Memulai download dataset PubMed (Limit: {limit_per_class} data per kategori)...")
    
    # 1. load dataset PubMed QA dari Hugging Face
    dataset = load_dataset("pubmed_qa", "pqa_labeled", split="train")
    
    # 2. keyword untuk semantic filtering
    keyword_tumor = ["tumor", "brain", "glioma", "meningioma", "pituitary", "malignant", "adenoma"]
    keyword_normal = ["normal brain", "healthy control", "unremarkable", "no abnormality", "no tumor", "non-pathological"]
    
    tumor_data = []
    normal_data = []
    
    for item in dataset:
      # gabungkan konteks dan pertanyaan untuk mengecek keyword
      context_text = item['context']['contexts'][0].lower()
      question_text = item['question'].lower()
      full_text = context_text + " " + question_text
      
      # filter untuk kelompok normal
      if any(key in full_text for key in keyword_normal) and len(normal_data) < limit_per_class:
        normal_data.append(self._format_text(item, "NORMAL/NO TUMOR"))
        
      # filter untuk kelompok tumor
      elif any(key in full_text for key in keyword_tumor) and len(tumor_data) < limit_per_class:
        tumor_data.append(self._format_text(item, "TUMOR DETECTED"))
        
      # hentikan jika kedua kelompok sudah memenuhi limit
      if len(normal_data) >= limit_per_class and len(tumor_data) >= limit_per_class:
        break
      
    # 3. masukkan golden standard 
    golden_normal = [
      "CATEGORY: NORMAL/NO TUMOR\n"
      "MEDICAL CONTEXT: Clinical Brain MRI Standards.\n"
      "CLINICAL QUESTION: What defines a normal brain MRI?\n"
      "EXPERT ANSWER: A normal brain MRI shows symmetrical structures, intact midline, normal ventricular size, and no evidence of mass, lesion, or edema."
    ]
    
    final_collection = normal_data + tumor_data + golden_normal
      
    # 4. simpan ke file teks di folder data/
    try:
      with open(self.output_path, "w", encoding="utf-8") as f:
        f.write("\n\n---\n\n".join(final_collection))
      
      print(f"Berhasil mendapatkan {len(final_collection)} data.")
      print(f"File knowledge disimpan di: {self.output_path}")
      
    except IOError as e:
      print(f"[Error] Terjadi error dalam menyimpan knowledge base: {e}")
  
if __name__ == "__main__":
  downloader = DownloadMedicaldata()
  downloader.download_and_filter(limit_per_class=60)