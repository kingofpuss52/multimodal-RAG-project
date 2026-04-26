import mlflow
import time
import os
from typing import Dict

class MedicalTracking:
    """
    MLOps diimplementasikan untuk tracking eksperimen Multimodal.
    Menyimpan log metrik model inferensi, performa RAG, dan artifact report.
    """
    
    def __init__(self, experiment_name: str="Brain-Tumor-Multimodal-RAG"):
        mlflow.set_experiment(experiment_name)
        
        # Tentukan folder penyimpanan lokal
        self.output_dir = "reports"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
    def _get_next_filename(self) -> str:
        """generate urutan nama file berikutnya agar tidak menimpa report sebelumnya"""
        i = 1
        while True:
            filename = f"temp_report_{i}.txt"
            full_path = os.path.join(self.output_dir, filename)
            if not os.path.exists(full_path):
                return full_path
            i += 1

    def log_full_analysis(self, vision_results: Dict, rag_results: Dict, hardware_info: Dict) -> str:
        """catat semua metrik dari model inferensi dan RAG dalam sekali jalan"""
        run_name = f"Run-{vision_results['label']}-{time.strftime('%Y%m%d-%H%M%S')}"
        
        with mlflow.start_run(run_name=run_name):
            # 1. Log Parameter Sistem
            mlflow.log_params({
                "vision_model": "EfficientNetB5",
                "llm_model": "Qwen-2.5-3B-Instruct",
                "quantization": "4-bit-NF4",
                "device": hardware_info['device'],
                "vector_db": "FAISS",
                "embedding_model": "all-MiniLM-L6-v2"
            })

            # 2. Log Metrik Vision (inferensi)
            mlflow.log_metrics({
                "vision_confidence": vision_results['confidence'],
                "vision_inference_time": vision_results['time']
            })

            # 3. Log Metrik RAG (Teks)
            mlflow.log_metrics({
                "rag_inference_time": rag_results['time'],
                "response_char_count": len(rag_results['report']),
                "context_relevance_score": rag_results['context_len'] # Jumlah karakter konteks
            })

            # 4. Log Tags (Metadata)
            mlflow.set_tags({
                "developer": "Nathasya Utami Hakim",
                "diagnosis_label": vision_results['label'],
                "project_phase": "Production-Evaluation"
            })

            # 5. Simpan hasil laporan
            report_path = self._get_next_filename()
            
            try:
                with open(report_path, "w", encoding="utf-8") as f:
                    f.write("--- INTEGRATED CLINICAL ANALYSIS REPORT ---\n")
                    f.write(f"Timestamp   : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Diagnosis   : {vision_results['label']}\n")
                    f.write(f"Confidence  : {vision_results['confidence']:.2%}\n")
                    f.write(f"Hardware    : {hardware_info.get('device', 'N/A')}\n")
                    f.write("-" * 43 + "\n")
                    f.write(f"NARRATIVE ANALYSIS:\n{rag_results['report']}\n")
                
                # upload ke artifact MLflow    
                mlflow.log_artifact(report_path, artifact_path="clinical_reports")
                
            except IOError as e:
                print(f"[Error] Gagal menyimpan report: {e}")
            
            return report_path