# 🧠 MediVision-RAG: Multimodal Brain Tumor Classification & Clinical Report Generation

[![MLflow](https://img.shields.io/badge/MLflow-Experiment_Tracking-blue?logo=mlflow)](https://mlflow.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-yellow?logo=python)](https://www.python.org/)
[![Vision-Model](https://img.shields.io/badge/Vision-EfficientNetB5-green)](https://keras.io/api/applications/efficientnet/)
[![LLM-Model](https://img.shields.io/badge/LLM-Qwen--2.5--3B-orange)](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)

## 📌 Project Overview
This project integrates **Computer Vision** and **Retrieval-Augmented Generation (RAG)** to detect brain tumors from MRI scans and automatically generate professional clinical reports. The system is optimized for edge deployment on an **NVIDIA RTX 3050** using 4-bit quantization techniques, ensuring high performance under hardware constraints.

### **Key Features:**
* **High-Fidelity Classification**: Utilizes **EfficientNetB5** to classify MRI scans into four categories: Glioma, Meningioma, Pituitary, and Normal.
* **Clinical RAG Engine**: Generates structured narrative reports in Indonesian using **Qwen-2.5-3B**, grounded in medical knowledge retrieved from PubMed datasets via **FAISS Vector DB**.
* **MLOps Experiment Tracking**: Real-time monitoring of inference latency, model confidence, and semantic consistency using **MLflow**.
* **Interactive Interface**: A user-friendly **Streamlit** dashboard for end-to-end diagnostic workflows.

---

## 🏗️ System Architecture
The system follows a modular multimodal pipeline:
1.  **Vision Engine**: Processes raw MRI imagery to extract diagnostic labels.
2.  **Knowledge Base**: A FAISS-powered vector store containing clinical protocols for context retrieval.
3.  **Inference Engine**: An LLM-based generator that synthesizes vision results and retrieved context into a final clinical summary.

---

## 📊 Performance Insights (via MLflow)
* **Inference Latency**: Achieved a steady-state vision inference time of **0.7s - 2.5s** on local hardware.
* **Model Confidence**: Maintains a >99% confidence interval across primary test sets.
* **Error Analysis**: Systematically identified edge-case misclassifications in some tumor variants, providing data-driven insights for future fine-tuning iterations.

### **Model Evaluation (Vision Engine)**
The EfficientNetB5 model was evaluated on a held-out test set. Below is the confusion matrix showing the model's performance across the four categories (Glioma, Meningioma, Pituitary, and Normal).

![Confusion Matrix](/assets/confusion%20matrix%20BO.png)

*The model exhibits high discriminative power, with minor overlaps identified in some class due to anatomical variations.*

---

## 🛠️ Tech Stack
* **Deep Learning**: TensorFlow, Keras, EfficientNetB5.
* **NLP & RAG**: LangChain, Hugging Face Transformers, Qwen-2.5-3B.
* **Optimization**: 4-bit Quantization (BitsAndBytes), FAISS (Vector Database).
* **Tools**: MLflow, Streamlit, Python.

---

## 🚀 Installation & Setup
Follow these steps to set up the environment and run the application:

### **1. Prerequisites**
* **Python 3.10** (Managed via `.python-version`)
* **NVIDIA GPU** with CUDA support (Recommended for LLM inference)
* **WSL2** (If running on Windows)

### **2. Environment Setup**
It is recommended to use a virtual environment or Conda:
```bash
# Create a new conda environment
conda create -n neurovision_env python=3.10 -y
conda activate neurovision_env

# Clone the repository
git clone [https://github.com/username/MediVision-RAG.git](https://github.com/username/MediVision-RAG.git)
cd MediVision-RAG

# Install dependencies
pip install -r requirements.txt
```

### **3. Model Acquisition**
* Vision Model: The EfficientNetB5 weights will be loaded automatically on the first run.
* LLM: Qwen-2.5-3B-Instruct will be downloaded from Hugging Face. Ensure you have ~5GB of disk space.

### **4. Launching the System**
To get the full experience with experiment tracking, run these in two separate terminals:

**Terminal 1 (Start MLflow server)**
``` bash
mlflow ui
```

**Terminal 2 (Start Streamlit App)**
```bash
streamlit run app.py
```

### **5. Usage**
1. Open the Streamlit dashboard at ```http://localhost:8501``` or the URL can be shown in terminal.
2. Upload a Brain MRI image (JPG/PNG).
3. Wait for the Vision Engine to classify and the RAG Engine to generate the report.
4. Check ```http://localhost:5000``` to see the detailed performance logs in MLflow.