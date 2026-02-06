import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import os
import tempfile

# ==========================================
# 1. DEFINISI CLASS PROTOTYPICAL NETWORK
# ==========================================
# Pindahkan @tf.function ke dalam method call, bukan di atas class.
class PrototypicalNetwork(tf.keras.Model):
    def __init__(self, encoder, **kwargs):
        super(PrototypicalNetwork, self).__init__(**kwargs)
        self.encoder = encoder

    @tf.function(reduce_retracing=True)
    def call(self, inputs, training=False):
        # Menggunakan self.encoder secara langsung
        return self.encoder(inputs, training=training)

    def predict(self, x):
        # Gunakan logic ini untuk inferensi di Streamlit
        return self.encoder.predict(x)

# ==========================================
# 2. FUNGSI LOAD MODEL DENGAN CACHE
# ==========================================
@st.cache_resource
def load_accent_model():
    model_path = "model_aksen.keras"
    if os.path.exists(model_path):
        try:
            # Memuat base encoder terlebih dahulu
            base_model = load_model(model_path, compile=False)
            # Membungkus ke dalam class PrototypicalNetwork
            model = PrototypicalNetwork(encoder=base_model)
            return model
        except Exception as e:
            st.error(f"Gagal memuat model: {e}")
            return None
    else:
        st.error(f"File model '{model_path}' tidak ditemukan!")
        return None

# ==========================================
# 3. INTERFACE STREAMLIT & PROSES PREDIKSI
# ==========================================
def main():
    st.title("Deteksi Aksen, Gender, dan Usia")
    
    # Load Model
    model = load_accent_model()
    
    # Mode Pilihan (seperti pada screenshot kamu)
    st.sidebar.title("Settings")
    mode = st.sidebar.radio("Select Mode:", ["Upload Audio"])

    if mode == "Upload Audio":
        uploaded_file = st.file_uploader("Pilih file audio...", type=["wav", "mp3", "m4a"])
        
        if uploaded_file is not None:
            st.audio(uploaded_file, format='audio/wav')
            
            if st.button("Proses Audio"):
                with st.spinner('Sedang menganalisis...'):
                    # --- Simulasi Preprocessing ---
                    # Pastikan input_data memiliki shape (1, 13) atau sesuai model kamu
                    # Contoh dummy data berdasarkan error "inputs=tf.Tensor(shape=(1, 13)...)"
                    input_data = np.random.rand(1, 13).astype(np.float32) 
                    
                    try:
                        # GUNAKAN .predict() untuk menghindari error symbolic tensor
                        prediction = model.predict(input_data)
                        
                        # Tampilkan Hasil (Contoh layout sesuai UI kamu)
                        st.success("Analisis Selesai!")
                        col1, col2, col3 = st.columns(3)
                        
                        # Asumsi output model berupa list atau array hasil multitaks
                        # Sesuaikan indeks [0][0] dengan output real skripsi kamu
                        col1.metric("📅 Usia", "30") 
                        col2.metric("👤 Gender", "Laki-laki")
                        col3.metric("📍 Provinsi", "DKI Jakarta")
                        
                    except Exception as e:
                        st.error(f"Error during processing: {e}")

if __name__ == "__main__":
    main()
