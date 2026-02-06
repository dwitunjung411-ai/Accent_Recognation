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
        return self.embedding(query_set)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embedding_model": tf.keras.layers.serialize(self.embedding)
        })
        return config

# ==========================================================
# 2. FUNGSI LOAD DATA (MODEL & METADATA)
# ==========================================================
@st.cache_resource
def load_accent_model():
    model_name = "model_detect_aksen.keras"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, model_name)
    
    if os.path.exists(model_path):
        try:
            custom_objects = {"PrototypicalNetwork": PrototypicalNetwork}
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
            return model
        except Exception as e:
            st.error(f"❌ Gagal memuat model: {e}")
            return None
    return None

@st.cache_data
def load_metadata():
    csv_path = "metadata.csv"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

def get_metadata_info(file_name, df_metadata):
    """Mencari info berdasarkan nama file di metadata.csv"""
    if df_metadata is not None:
        # Pastikan kolom 'file_name' sesuai dengan nama kolom di CSV kamu
        match = df_metadata[df_metadata['file_name'] == file_name]
        if not match.empty:
            return match.iloc[0].to_dict()
    return None

# ==========================================================
# 3. FUNGSI PREDIKSI
# ==========================================================
def predict_accent(audio_path, model):
    if model is None: return "Model tidak tersedia"
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_scaled = np.mean(mfcc.T, axis=0)
        input_data = np.expand_dims(mfcc_scaled, axis=0) 
        
        prediction = model.predict(input_data)
        aksen_classes = ["Sunda", "Jawa Tengah", "Jawa Timur", "Yogyakarta", "Betawi"]
        return aksen_classes[np.argmax(prediction)]
    except Exception as e:
        return f"Error Analisis: {str(e)}"

# ==========================================================
# 4. MAIN UI
# ==========================================================
def main():
    st.set_page_config(page_title="Deteksi Aksen Prototypical", page_icon="🎙️")
    
    model_aksen = load_accent_model()
    df_metadata = load_metadata()

    st.title("🎙️ Sistem Deteksi Aksen Indonesia")
    
    with st.sidebar:
        st.header("⚙️ Status Sistem")
        st.success("Model Terhubung") if model_aksen else st.error("Model Terputus")
        st.success("Metadata Terhubung") if df_metadata is not None else st.warning("Metadata.csv Tidak Ada")
        st.divider()
        st.caption("Skripsi Project - Voice Recognition")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Upload Audio")
        audio_file = st.file_uploader("Pilih file audio", type=["wav", "mp3"])
        
        if audio_file:
            st.audio(audio_file)
            if st.button("🚀 Mulai Analisis", type="primary"):
                if model_aksen:
                    with st.spinner("Menganalisis..."):
                        # 1. Proses Prediksi Aksen
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                            tmp.write(audio_file.getbuffer())
                            hasil_aksen = predict_accent(tmp.name, model_aksen)
                        
                        # 2. Cari data di metadata.csv secara otomatis
                        user_info = get_metadata_info(audio_file.name, df_metadata)
                        
                        with col2:
                            st.subheader("📊 Hasil Deteksi")
                            st.info(f"Aksen Terdeteksi: **{hasil_aksen}**")
                            
                            st.subheader("👤 Profil Pembicara")
                            if user_info:
                                # Menampilkan info dari metadata
                                st.write(f"**Usia:** {user_info.get('usia', 'Tidak diketahui')}")
                                st.write(f"**Gender:** {user_info.get('gender', 'Tidak diketahui')}")
                                st.write(f"**Provinsi:** {user_info.get('provinsi', 'Tidak diketahui')}")
                            else:
                                st.warning("Data file ini tidak ditemukan di metadata.csv")
                            
                            st.balloons()
                        os.unlink(tmp.name)
                else:
                    st.error("Model tidak siap.")

if __name__ == "__main__":
    main()
