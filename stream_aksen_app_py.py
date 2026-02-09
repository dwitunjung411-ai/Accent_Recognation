import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tempfile
import os
import tensorflow as tf

# ==========================================================
# 1. DEFINISI CLASS (SESUAIKAN DENGAN PROSES SAVE)
# ==========================================================
@tf.keras.utils.register_keras_serializable(package="Custom")
class PrototypicalNetwork(tf.keras.Model):
    def __init__(self, embedding_model=None, **kwargs):
        super(PrototypicalNetwork, self).__init__(**kwargs)
        self.embedding = embedding_model

    def call(self, inputs):
        return self.embedding(inputs)

    def get_config(self):
        config = super().get_config()
        config.update({"embedding_model": tf.keras.layers.serialize(self.embedding)})
        return config

# ==========================================================
# 2. LOAD MODEL & DATA PENDUKUNG
# ==========================================================
@st.cache_resource
def load_system_resources():
    # 1. Sesuaikan Nama File dengan yang ada di GitHub (image_8401e7.png)
    model_name = "model_embedding_aksen.keras" 
    
    try:
        # Load Model
        custom_objects = {"PrototypicalNetwork": PrototypicalNetwork}
        model = tf.keras.models.load_model(model_name, custom_objects=custom_objects, compile=False)
        
        # 2. Load Support Set (Diperlukan karena ini Prototypical Network)
        # File ini terlihat ada di GitHub Anda
        support_set = np.load("support_set.npy")
        support_labels = np.load("support_labels.npy")
        
        return model, support_set, support_labels
    except Exception as e:
        st.sidebar.error(f"Gagal Load: {e}")
        return None, None, None

# ==========================================================
# 3. LOGIKA PREDIKSI FEW-SHOT
# ==========================================================
def predict_few_shot(audio_path, model, support_set, support_labels):
    # Ekstraksi MFCC dari Query (Audio Upload)
    y, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    query_feat = np.mean(mfcc.T, axis=0).reshape(1, -1)
    
    # Hitung Embedding
    query_embed = model.predict(query_feat, verbose=0)
    support_embed = model.predict(support_set, verbose=0)
    
    # Hitung Prototypes (Rata-rata embedding per kelas)
    unique_labels = np.unique(support_labels)
    prototypes = []
    for label in unique_labels:
        p = np.mean(support_embed[support_labels == label], axis=0)
        prototypes.append(p)
    prototypes = np.array(prototypes)
    
    # Hitung Jarak Euclidean terdekat
    distances = np.linalg.norm(prototypes - query_embed, axis=1)
    class_idx = np.argmin(distances)
    
    aksen_classes = ["Sunda", "Jawa Tengah", "Jawa Timur", "Yogyakarta", "Betawi"]
    return aksen_classes[class_idx]

# ==========================================================
# 4. MAIN UI
# ==========================================================
def main():
    st.set_page_config(page_title="Accent Recognition", layout="wide")
    st.title("🎙️ Accent Recognition (Few-Shot Learning)")

    # Load resources
    model, s_set, s_labs = load_system_resources()

    with st.sidebar:
        if model is not None:
            st.success("Sistem Online")
        else:
            st.error("Sistem Offline")
            st.write("Cek apakah file `.keras` dan `.npy` sudah ada di root folder.")

    col1, col2 = st.columns(2)

    with col1:
        audio_file = st.file_uploader("Upload Audio", type=["wav", "mp3"])
        if audio_file and model is not None:
            st.audio(audio_file)
            if st.button("Deteksi Aksen"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(audio_file.getbuffer())
                    
                hasil = predict_few_shot(tmp.name, model, s_set, s_labs)
                
                with col2:
                    st.info(f"### Hasil: {hasil}")
                
                os.remove(tmp.name)

if __name__ == "__main__":
    main()
