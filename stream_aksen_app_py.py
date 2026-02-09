import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tempfile
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

# ==========================================================
# 1. DEFINISI CLASS PROTOTYPICAL NETWORK
# ==========================================================
@tf.keras.utils.register_keras_serializable(package="Custom")
class PrototypicalNetwork(tf.keras.Model):
    def __init__(self, embedding_model=None, **kwargs):
        super(PrototypicalNetwork, self).__init__(**kwargs)
        self.embedding = embedding_model

    def call(self, support_set, query_set, support_labels, n_way):
        # Memastikan embedding dipanggil dengan query_set
        return self.embedding(query_set)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embedding_model": tf.keras.layers.serialize(self.embedding)
        })
        return config

# ==========================================================
# 2. FUNGSI LOAD DATA
# ==========================================================
@st.cache_resource
def load_accent_model():
    model_name = "model_embedding_aksen.keras"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, model_name)

    if os.path.exists(model_path):
        try:
            custom_objects = {"PrototypicalNetwork": PrototypicalNetwork}
            # Load tanpa compile untuk stabilitas
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
            return model
        except Exception as e:
            return None
    return None

@st.cache_data
def load_metadata_df():
    csv_path = "metadata.csv"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

# ==========================================================
# 3. FUNGSI PREDIKSI (PERBAIKAN ERROR QUERY_SET)
# ==========================================================
def predict_accent(audio_path, model):
    if model is None: return "Model tidak tersedia"
    try:
        # Load & Preprocess
        y, sr = librosa.load(audio_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_scaled = np.mean(mfcc.T, axis=0)

        # Sesuai error: Model Prototypical seringkali butuh input dalam bentuk list
        # atau argumen bernama jika dibungkus class kustom
        input_data = np.expand_dims(mfcc_scaled, axis=0)

        # Mencoba prediksi langsung (seringkali model.predict cukup jika call() sudah benar)
        prediction = model.predict(input_data)

        aksen_classes = ["Sunda", "Jawa Tengah", "Jawa Timur", "Yogyakarta", "Betawi"]
        return aksen_classes[np.argmax(prediction)]
    except Exception as e:
        return f"Error Analisis: {str(e)}"

# ==========================================================
# 4. MAIN UI (WIDE LAYOUT & NEW ICONS)
# ==========================================================
def main():
    # Set layout lebar agar tidak sempit
    st.set_page_config(page_title="Deteksi Aksen Prototypical", page_icon="🎙️", layout="wide")

    model_aksen = load_accent_model()
    df_metadata = load_metadata_df()

    st.title("🎙️ Sistem Deteksi Aksen Prototypical Indonesia")
    st.write("Aplikasi berbasis *Few-Shot Learning* untuk klasifikasi aksen daerah.")
    st.divider()

    with st.sidebar:
        st.header("🛸 Status Sistem")
        if model_aksen:
            st.success("🤖 Model: Terhubung")
        else:
            st.error("🚫 Model: Terputus")

        if df_metadata is not None:
            st.success("📁 Metadata: Siap")
        else:
            st.warning("⚠️ Metadata: Kosong")

        st.divider()
        st.caption("Skripsi Project - 2026")

    # Pembagian kolom agar lebar
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader("📥 Input Audio")
        audio_file = st.file_uploader("Upload file (.wav, .mp3)", type=["wav", "mp3"])

        if audio_file:
            st.audio(audio_file)
            # Tombol diperlebar agar proporsional
            # Tambahkan prioritas/prioritized detection
PRIORITY_ACCENTS = ['jakarta', 'sunda', 'jawa', 'bali', 'minang']

def detect_with_priority(query_features, support_set):
    # Deteksi semua aksen
    all_predictions = prototypical_network(query_features, support_set)
    
    # Jika confidence rendah untuk non-prioritas, cek prioritas dulu
    if all_predictions['confidence'] < 0.7:
        # Filter hanya aksen prioritas
        priority_set = {k: v for k, v in support_set.items() 
                       if k in PRIORITY_ACCENTS}
        priority_pred = prototypical_network(query_features, priority_set)
        
        if priority_pred['confidence'] > 0.6:
            return priority_pred
    
    return all_predictionsmain()
