import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tempfile
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

# ==========================================================
# 1. DEFINISI CLASS PROTOTYPICAL NETWORK (WAJIB ADA)
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
# 2. FUNGSI LOAD MODEL (MENGGUNAKAN NAMA BARU)
# ==========================================================
@st.cache_resource
def load_accent_model():
    # Nama model diperbarui sesuai permintaan kamu
    model_name = "model_detect_aksen.keras"
    
    # Mencari path absolut file di server
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, model_name)
    
    if os.path.exists(model_path):
        try:
            custom_objects = {"PrototypicalNetwork": PrototypicalNetwork}
            # Load model tanpa compile untuk menghindari error optimizer
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
            return model
        except Exception as e:
            st.error(f"❌ Gagal memuat model: {e}")
            return None
    else:
        # Menampilkan pesan error yang membantu jika file tidak ada di GitHub
        st.error(f"⚠️ File '{model_name}' tidak ditemukan!")
        st.info(f"Pastikan file sudah di-upload ke folder yang sama dengan script ini di GitHub.")
        return None

# ==========================================================
# 3. FUNGSI PREDIKSI
# ==========================================================
def predict_accent(audio_path, model):
    if model is None:
        return "Model tidak tersedia"

    try:
        # Load audio (SR 16000 adalah standar umum, sesuaikan jika berbeda)
        y, sr = librosa.load(audio_path, sr=16000)
        
        # --- BAGIAN PENTING: SESUAIKAN DENGAN SKRIPSI KAMU ---
        # Contoh ekstraksi MFCC (Pastikan shape input sama dengan saat training)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_scaled = np.mean(mfcc.T, axis=0)
        input_data = np.expand_dims(mfcc_scaled, axis=0) 
        # ----------------------------------------------------

        prediction = model.predict(input_data)
        
        aksen_classes = ["Sunda", "Jawa Tengah", "Jawa Timur", "Yogyakarta", "Betawi"]
        predicted_idx = np.argmax(prediction)
        
        return aksen_classes[predicted_idx]
    except Exception as e:
        return f"Error Analisis: {str(e)}"

# ==========================================================
# 4. ANTARMUKA UTAMA (UI)
# ==========================================================
def main():
    st.set_page_config(page_title="Deteksi Aksen Prototypical", page_icon="🎙️")
    
    # Load model ke cache
    model_aksen = load_accent_model()

    st.title("🎙️ Sistem Deteksi Aksen Indonesia")
    st.write("Gunakan aplikasi ini untuk mendeteksi aksen daerah dari rekaman suara.")

    with st.sidebar:
        st.header("⚙️ Status Sistem")
        if model_aksen:
            st.success("Model: Terhubung")
        else:
            st.error("Model: Terputus")
        
        st.divider()
        st.caption("Skripsi Project - Prototypical Network")

    # Layout kolom
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Upload Audio")
        audio_file = st.file_uploader("Pilih file .wav atau .mp3", type=["wav", "mp3"])
        
        if audio_file:
            st.audio(audio_file)
            if st.button("🚀 Mulai Deteksi", type="primary"):
                if model_aksen:
                    with st.spinner("Sedang menganalisis..."):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                            tmp.write(audio_file.getbuffer())
                            hasil = predict_accent(tmp.name, model_aksen)
                        
                        with col2:
                            st.subheader("Hasil Analisis")
                            st.info(f"Aksen Terdeteksi: **{hasil}**")
                            st.balloons()
                        
                        os.unlink(tmp.name) # Hapus file temp
                else:
                    st.warning("Model belum siap. Periksa file di GitHub.")

if __name__ == "__main__":
    main()
