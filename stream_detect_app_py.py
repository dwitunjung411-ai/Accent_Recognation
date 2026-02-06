import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import os
import tempfile

# ==========================================================
# 1. DEFINISI CLASS PROTOTYPICAL NETWORK
# ==========================================================
class PrototypicalNetwork(tf.keras.Model):
    def __init__(self, encoder, **kwargs):
        super(PrototypicalNetwork, self).__init__(**kwargs)
        self.encoder = encoder

    # PINDAHKAN @tf.function ke sini (di atas method call)
    @tf.function(reduce_retracing=True)
    def call(self, inputs):
        return self.encoder(inputs)

    def predict(self, x):
        return self.encoder.predict(x)

# ==========================================================
# 2. FUNGSI LOAD MODEL DENGAN CACHE
# ==========================================================
@st.cache_resource
def load_accent_model():
    model_path = "model_aksen.keras"
    if os.path.exists(model_path):
        try:
            # Muat model dasar (encoder)
            base_model = load_model(model_path, compile=False)
            # Bungkus ke dalam class PrototypicalNetwork
            model = PrototypicalNetwork(encoder=base_model)
            return model
        except Exception as e:
            st.error(f"Gagal memuat model: {e}")
            return None
    else:
        st.error(f"File model '{model_path}' tidak ditemukan di repository!")
        return None

# ==========================================================
# 3. INTERFACE UTAMA
# ==========================================================
def main():
    st.set_page_config(page_title="Deteksi Aksen & Karakteristik Suara", layout="wide")
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    mode = st.sidebar.radio("Select Mode:", ["Upload Audio"])
    
    # Load Model
    model = load_accent_model()

    if mode == "Upload Audio":
        st.subheader("🎵 Analisis Audio")
        uploaded_file = st.file_uploader("Upload file audio (wav/mp3)", type=["wav", "mp3"])
        
        if uploaded_file is not None:
            st.audio(uploaded_file)
            
            if st.button("Mulai Deteksi"):
                if model is not None:
                    with st.spinner('Sedang memproses...'):
                        try:
                            # --- PREPROCESSING (Pastikan ini sesuai dengan Librosa kamu) ---
                            # Dummy input sesuai shape di error (1, 13)
                            input_features = np.random.rand(1, 13).astype(np.float32)
                            
                            # Melakukan Prediksi
                            # Gunakan model.predict untuk menghindari masalah symbolic tensor di Streamlit
                            predictions = model.predict(input_features)
                            
                            # --- TAMPILAN HASIL ---
                            st.success("Analisis Berhasil!")
                            
                            # Contoh menampilkan data (sesuaikan dengan output asli model skripsi kamu)
                            res_col1, res_col2, res_col3 = st.columns(3)
                            res_col1.metric("📅 Usia", "70") # Contoh statis sesuai gambar
                            res_col2.metric("👤 Gender", "Perempuan")
                            res_col3.metric("📍 Provinsi", "DKI Jakarta")
                            
                        except Exception as e:
                            st.error(f"Terjadi kesalahan saat inferensi: {e}")
                else:
                    st.error("Model belum dimuat dengan benar.")

if __name__ == "__main__":
    main()
