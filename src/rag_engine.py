import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig, AutoConfig, StoppingCriteria, StoppingCriteriaList
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

class StopOnTokens(StoppingCriteria):
  """custom stopping criteria untuk mencegah LLM 'halusinasi'"""
  def __init__(self, stop_ids):
      self.stop_ids = stop_ids
  def __call__(self, input_ids: torch.LongTensor,scores: torch.FloatTensor, **kwargs) -> bool:
      # Cek apakah token terakhir yang keluar ada di daftar stop_ids
      for stop_id in self.stop_ids:
          if input_ids[0][-1] == stop_id:
              return True
      return False

class RAGEngine:
  """inisialisasi RAG engine dengan quantization LLM 4-bit dan setup vector DB"""
  def __init__(self, model_id="Qwen/Qwen2.5-3B-Instruct"):
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    
    # cek kompatibilitas scaling RoPE
    if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
      if "type" not in config.rope_scaling:
        config.rope_scaling["type"] = config.rope_scaling.get("rope_type", "default")
    
    # 1. konfigurasi kuantisasi 4-bit agar muat di GPU (RTX 3050)
    bnb_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_compute_dtype=torch.float16,
      bnb_4bit_use_double_quant=True
    )
    
    # 2. Load tokenizer dan quantization LLM
    self.tokenizer = AutoTokenizer.from_pretrained(model_id)
    self.tokenizer.pad_token = self.tokenizer.eos_token
    
    self.model = AutoModelForCausalLM.from_pretrained(
      model_id,
      config=config,
      quantization_config=bnb_config,
      device_map="auto",
      trust_remote_code=True,
    )
    
    # 3. setup pipeline untuk text generation
    stop_tokens = ["<|im_end|>", "<|im_start|>", "Assistant:", "User:"]
    stop_ids = [self.tokenizer.encode(t, add_special_tokens=False)[-1] for t in stop_tokens]
    stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_ids)])
    
    pipe = pipeline(
      "text-generation",
      model=self.model,
      tokenizer=self.tokenizer,
      max_new_tokens=150,
      do_sample=True,
      return_full_text=False,
      repetition_penalty=1.1,
      temperature=0.1,
      top_p=0.9,
      top_k=None,
      stopping_criteria=stopping_criteria
    )
    self.llm = HuggingFacePipeline(pipeline=pipe)
    
    # 4. load knowledge base
    self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    self.vector_db = None
    
  def setup_knowledge_base(self, documents):
    """buat index pencarian dari dokumen teks"""
    self.vector_db = FAISS.from_documents(documents, self.embeddings)
    
  def query_clinical_report(self, diagnosis, confidence):
    """generate report berdasarkan hasil inferensi dan context"""
    if self.vector_db is None:
      raise ValueError("Vector DB belum dimuat. Pastikan index FAISS sudah di-assign sebelumnya.")
    
    # proses retrieve 
    search_query = f"Protokol klinis {diagnosis}"
    docs = self.vector_db.similarity_search(search_query, k=2)
    
    # ekstrak teks dari database dokumen
    context_parts = [doc.page_content for doc in docs if hasattr(doc, "page_content")]
    
    # gabungkan isi dokumen jadi satu string konteks
    context_text = "\n\n".join(context_parts)
    
    # jika hasil prediksi normal, paksa konteksnya agar LLM tidak 'pamer' ilmu
    if diagnosis in ["Normal", "No Tumor"]:
      context_text = "Struktur anatomi otak normal, ventrikel simetris, tidak ada massa abnormal."
    
    # jika dari pencarian FAISS kosong, berikan fallback
    if not context_text.strip():
      context_text = "Informasi medis standar untuk kondisi yang dilaporkan."
    
    # buat prompt template
    template = """<|im_start|>system
    Anda adalah asisten radiologi senior. Tugas Anda memberikan penjelasan medis singkat.
    Jangan mengulang data input. Langsung berikan analisis.<|im_end|>
    <|im_start|>user
    Hasil diagnosa sistem menunjukkan {diagnosis} dengan tingkat kepercayaan {confidence:.2%}.
    Referensi tambahan: {context}

    Berikan penjelasan singkat mengenai hasil tersebut dalam 2 kalimat Bahasa Indonesia.<|im_end|>
    <|im_start|>assistant
    Hasil pemeriksaan menunjukkan bahwa """
    
    prompt = PromptTemplate.from_template(template)
    
    # buat chain menggunakan LCEL
    chain = prompt | self.llm.bind(stop=["<|im_end|>", "DATA PASIEN:"]) | StrOutputParser()
    
    try:
      response = chain.invoke({
        "diagnosis": diagnosis,
        "confidence": confidence,
        "context": context_text[:800]  
      })
      
      full_text = "Hasil pemeriksaan menunjukkan bahwa " + response.split("<|im_end|>")[0].strip()
      
      # Semantic guardrail: blacklist kata pemicu model 'halusinasi'
      blacklist = ["alergi", "bencana", "astrofaktorm", "knight"]
      
      if any(word in full_text.lower() for word in blacklist) or len(full_text) < 20:
        return f"Analisis MRI menunjukkan indikasi {diagnosis} ({confidence:.2%}). Disarankan konsultasi lebih lanjut."

      return full_text
    
    except Exception as e:
      return f"Hasil klasifikasi: {diagnosis} ({confidence:.2%}). (Note: Gagal menghasilkan ringkasan naratif)."