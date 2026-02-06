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
        # Logika minimal agar Keras bisa mengonstruksi ulang model
        return self.embedding(query_set)

    def get_config(self):
        config = super().get_config()
        # Pastikan embedding diserialisasi dengan benar
        config.update({
            "embedding_model": tf.keras.layers.serialize(self.embedding)
        })
        return config

# ==========================================================
# 2. FUNGSI LOAD MODEL (MENGGUNAKAN CACHE)
# ==========================================================
@st.cache_resource
def load_accent_model():
    # Menggunakan path relatif agar aman di Streamlit Cloud
    model_path = os.path.join(os.getcwd(), "model_aksen.keras")
    
    if os.path.exists(model_path):
        try:
            custom_objects = {"PrototypicalNetwork": PrototypicalNetwork}
            # Load model dengan custom objects
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
            return model
        except Exception as e:
            st.error(f"❌ Gagal memuat model: {e}")
            return None
    else:
        st.error(f"⚠️ File '{model_path}' tidak ditemukan di server! Pastikan sudah di-upload ke GitHub.")
        return None

# ==========================================================
# 3. FUNGSI PEMROSESAN AUDIO & PREDIKSI
# ==========================================================
def load_metadata(csv_path):
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return pd.DataFrame()

def predict_accent(audio_path, model):
    if model is None:
        return "Model tidak tersedia"

    try:
        # 1. Load Audio
        y, sr = librosa.load(audio_path, sr=16000) # Sesuaikan SR dengan saat training
        
        # 2. Ekstraksi Fitur (CONTOH: MFCC)
        # PENTING: Sesuaikan bagian ini agar PERSIS dengan proses preprocessing di Skripsi kamu
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_scaled = np.mean(mfcc.T, axis=0)
        input_data = np.expand_dims(mfcc_scaled, axis=0) # Bentuk (1, 40)
        input_data = np.expand_dims(input_data, axis=-1) # Jika model butuh (1, 40, 1)

        # 3. Prediksi
        prediction = model.predict(input_data)
        
        # 4. Mapping Label
        aksen_classes = ["Sunda", "Jawa Tengah", "Jawa Timur", "Yogyakarta", "Betawi"]
        predicted_idx = np.argmax(prediction)
        
        return aksen_classes[predicted_idx]
    except Exception as e:
        return f"Error Prediksi: {str(e)}"

# ==========================================================
# 4. MAIN APP INTERFACE
# ==========================================================
def main():
    st.set_page_config(page_title="Deteksi Aksen Prototypical", page_icon="🎙️", layout="wide")

    # Load model di awal
    model_aksen = load_accent_model()

    st.title("🎙️ Sistem Deteksi Aksen Indonesia")
    st.markdown("Aplikasi ini mendeteksi aksen berdasarkan model **Prototypical Network**.")

    with st.sidebar:
        st.header("⚙️ Settings")
        st.info("Pastikan file audio jernih dan berformat .wav atau .mp3")
        st.divider()
        st.write("Status Model:")
        if model_aksen:
            st.success("✅ Model Ready")
        else:
            st.error("❌ Model Offline")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📤 Input Audio")
        audio_file = st.file_uploader("Upload file audio", type=["wav", "mp3"])
        
        if audio_file is not None:
            st.audio(audio_file, format="audio/wav")
            
            if st.button("🚀 Jalankan Deteksi", type="primary"):
                if model_aksen is None:
                    st.warning("Tidak bisa melakukan deteksi karena model belum termuat.")
                else:
                    with st.spinner("Menganalisis karakteristik suara..."):
                        # Simpan audio ke file sementara
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                            tmp_file.write(audio_file.getbuffer())
                            tmp_path = tmp_file.name

                        # Eksekusi Prediksi
                        hasil_aksen = predict_accent(tmp_path, model_aksen)

                        # Tampilkan Hasil di Kolom 2
                        with col2:
                            st.subheader("📊 Hasil Analisis")
                            st.balloons()
                            st.success(f"### Prediksi Aksen: **{hasil_aksen}**")
                            
                            # Tampilkan Metadata jika ada
                            metadata = load_metadata("metadata.csv")
                            if not metadata.empty:
                                info = metadata[metadata['file_name'] == audio_file.name]
                                if not info.empty:
                                    st.write("---")
                                    st.write(f"**Usia:** {info['usia'].values[0]}")
                                    st.write(f"**Gender:** {info['gender'].values[0]}")
                                    st.write(f"**Provinsi Asal:** {info['provinsi'].values[0]}")

                        # Hapus file sementara setelah selesai
                        os.unlink(tmp_path)

    if audio_file is None:
        with col2:
            st.info("Silakan unggah file audio untuk melihat hasil analisis di sini.")

if __name__ == "__main__":
    main()
