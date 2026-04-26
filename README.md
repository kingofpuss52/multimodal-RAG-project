# 🧠 NeuroScan AI: Multimodal Brain Tumor RAG System

[![MLflow](https://img.shields.io/badge/MLflow-Experiment_Tracking-blue?logo=mlflow)](https://mlflow.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-yellow?logo=python)](https://www.python.org/)
[![Vision-Model](https://img.shields.io/badge/Vision-EfficientNetB5-green)](https://keras.io/api/applications/efficientnet/)
[![LLM-Model](https://img.shields.io/badge/LLM-Qwen--2.5--3B-orange)](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)

## 📌 Project Overview
This project integrates **Computer Vision** and **Retrieval-Augmented Generation (RAG)** to detect brain tumors from MRI scans and automatically generate professional clinical reports. The system is optimized for hybrid deployment (CPU for Vision, GPU for LLM) on an **NVIDIA RTX 3050** using 4-bit quantization, ensuring high performance under hardware constraints.

### **Key Features:**
* **High-Fidelity Classification**: Utilizes **EfficientNetB5** to classify MRI scans into: Glioma, Meningioma, Pituitary, and Normal.
* **Clinical RAG Engine**: Generates structured narrative reports in Indonesian using **Qwen-2.5-3B**, grounded in medical knowledge retrieved from PubMed datasets.
* **Automated Optimization**: Employs **Optuna** for hyperparameter tuning of RAG chunking strategies, tracked via **MLflow**.
* **MLOps Integration**: Real-time monitoring of inference latency and model confidence.

---

## 🏗️ System Architecture
The system follows a modular multimodal pipeline:
1.  **Vision Engine**: Extracts diagnostic labels from MRI imagery.
2.  **Knowledge Base**: A FAISS-powered vector store containing clinical protocols.
3.  **Inference Engine**: Synthesizes vision results and retrieved context into a clinical summary.

---

## 🛠️ Tech Stack
* **Deep Learning**: TensorFlow/Keras (Vision), PyTorch (LLM).
* **NLP & RAG**: LangChain, Hugging Face Transformers, Qwen-2.5-3B.
* **Optimization**: Optuna (Tuning), BitsAndBytes (4-bit Quantization), FAISS (Vector DB).
* **Tracking**: MLflow.
* **Interface**: Streamlit.

---

## 📊 Performance Insights
### **Model Evaluation (Vision Engine)**
The EfficientNetB5 model, optimized via Bayesian Search, achieves high discriminative power across all classes.

![Confusion Matrix](/assets/confusion%20matrix%20BO.png)

* **Inference Latency**: 0.7s - 2.5s (Vision) | 3.0s - 5.0s (RAG).
* **Accuracy**: Robust diagonal performance with minor semantic overlap between Glioma and Meningioma.

---

## 🚀 Execution Pipeline (Step-by-Step)

Follow these steps to initialize the knowledge base and launch the system:

### **1. Environment Setup**
```bash
# Create and activate environment
conda create -n neurovision_env python=3.10 -y
conda activate neurovision_env

# Install dependencies
pip install -r requirements.txt
```

### **2. Data Acquisition & Pre-processing**
Download and filter clinical data from the PubMed QA dataset to build the local knowledge base.
```bash
python src/download_medical_data.py
```

### **3. RAG Hyperparameter Tuning (Optional but Recommended**
Run the automated tuner to find the optimal ```chunk_size``` and ```chunk_overlap``` using Optuna.
```bash
python src/tuning_rag.py
```
*Results are saved to ```configs/best_params.json``` and logged to MLflow.*

### **4. Vector Index Construction**
Generate the FAISS vector database using the optimized parameters.
```bash
python src/create_index_faiss.py
```

### **5. Model Placement**
Ensure your trained vision model is placed inside the ```models/``` directory.

## 🖥️ Launching the Application
Run the following commands in separate terminals to start the full multimodal ecosystem:

**Terminal 1: MLflow Tracking Server**
```bash
mlflow ui
```

**Terminal 2: Streamlit Interactive Dashboard**
```bash
streamlit run app.py
```

## 📂 Project Structure

* ```src/```: Core logic for Vision, RAG, Indexing, and Tuning.
* ```data/```: Knowledge base text files and FAISS index
* ```models/```: Pre-trained model weights.
* ```configs/```: Optimized system configurations.
* ```reports/```: Locally persisted clinical analysis outputs.

## ⚖️ Medical Disclaimer
This system is a Decision Support System (DSS) developed for research and portfolio purposes. The generated outputs are not official medical diagnoses. Always consult a certified radiologist or medical professional for clinical assessments.

---

Developed by: Nathasya Utami Hakim - 2026