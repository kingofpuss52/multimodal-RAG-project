import os
import tensorflow as tf
import keras
import numpy as np
from PIL import Image
from keras.applications.efficientnet import preprocess_input
from typing import Union, Dict

# paksa TensorFlow menggunakan backend Keras 3
os.environ["KERAS_BACKEND"] = "tensorflow"

class VisionEngine:
  def __init__(self, model_path):
    """
    Inferensi untuk klasifikasi tumor otak menggunakan EfficientNetB5.
    Setup agar model inferensi menggunakan CPU agar LLM menggunakan GPU.
    """
    
    self._configure_hardware()
    self.classes = ['Glioma', 'Meningioma', 'Normal', 'Pituitary']
    self.img_size = (224, 224)
    self.model = self._load_model(model_path)
    
  def _configure_hardware(self):
    """konfigurasi agar tensorflow hanya menggunakan CPU"""
    try:
      tf.config.set_visible_devices([], 'GPU')
      
    except RuntimeError as e:
      print(f"[System Info] TensorFlow Hardware Config: {e}")
      
  def _load_model(self, path:str):
    """load model inferensi"""
    if not os.path.exists(path):
      raise FileNotFoundError(f"File model tidak ditemukan di: {path}")
    
    try:
      model = keras.models.load_model(path, compile=False)
      return model
    
    except Exception as e:
      raise RuntimeError(f"Gagal load model inferensi: {str(e)}")
    
  def preprocess_image(self, pil_image: Image.Image) -> np.ndarray:
    """preprocessing untuk model EfficientNetB5 """
    
    # konversi ke RGB dan resize
    img = pil_image.convert("RGB").resize(self.img_size)
    img_array = np.array(img)
    
    # perluas dimensi
    img_array = np.expand_dims(img_array, axis=0)
    
    return preprocess_input(img_array)
  
  def predict(self, image_input: Union[str, Image.Image]) -> Dict:
    """Prediksi jenis tumor dan skor confidence.
    Menerima path file atau file gambar 
    """
    if self.model is None:
      raise AttributeError("Model belum dimuat.")
    
    # ambil path gambar atau objek PIL
    if isinstance(image_input, str):
      img = Image.open(image_input)
      
    else:
      img = image_input
      
    processed_img = self.preprocess_image(img)
    preds = self.model.predict(processed_img, verbose=0)
    
    idx = np.argmax(preds)
    
    return {
      "label": self.classes[idx],
      "confidence": float(preds[0][idx]),
      "all_predictions": dict(zip(self.classes, map(float, preds[0])))
    }