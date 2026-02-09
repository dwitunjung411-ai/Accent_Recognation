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

    def call(self, inputs, training=False):
        # Sesuaikan call dengan input inference Anda
        return self.embedding(inputs)

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
    model_name = "model_embedding_aksen.keras"
    # Gunakan path relatif yang lebih aman untuk deployment
    model_path = os.path.join(os.getcwd(), model_name)

    if not os.path.exists(model_path):
        st.error(f"⚠️ File model '{model_name}' tidak ditemukan di direktori root.")
        return None

    try:
        # Tambahkan compile=False agar tidak perlu optimizer saat load
        custom_objects = {"PrototypicalNetwork": PrototypicalNetwork}
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
        return model
    except Exception as e:
        st.error(f"❌ Error Detail: {e}")
        return None

@st.cache_data
def load_metadata_df():
    csv_path = "metadata.csv"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

# ==========================================================
# 3. FUNGSI PREDIKSI (DENGAN RE-SAMPLING)
# ==========================================================
def predict_accent(audio_path, model):
    if model is None: return "Model tidak tersedia"
    try:
        # Load audio (sr=16000 sesuai training)
        y, sr = librosa.load(audio_path, sr=16000)
        
        # Ekstraksi MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_scaled = np.mean(mfcc.T, axis=0)
        
        # Reshape untuk model (batch_size, features)
        input_data = np.expand_dims(mfcc_scaled, axis=0)

        # Prediksi
        prediction = model.predict(input_data, verbose=0)
        aksen_classes = ["Sunda", "Jawa Tengah", "Jawa Timur", "Yogyakarta", "Betawi"]
        
        index = np.argmax(prediction)
        return aksen_classes[index]
    except Exception as e:
        return f"Error Analisis: {str(e)}"

# ==========================================================
# 4. MAIN UI
# ==========================================================
def main():
    st.set_page_config(page_title="Accent Recognition", page_icon="🎙️", layout="wide")

    model_aksen = load_accent_model()
    df_metadata = load_metadata_df()

    st.title("🎙️ Accent Recognition")
    st.divider()

    with st.sidebar:
        st.header("⚙️ Status Sistem")
        if model_aksen:
            st.success("Model: Online")
        else:
            st.error("Model: Offline")
            st.info("Pastikan file 'model_detect_aksen.keras' ada di folder yang sama dengan app.py")

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader("📤 Input Audio")
        audio_file = st.file_uploader("Upload file (.wav, .mp3)", type=["wav", "mp3"])

        if audio_file:
            st.audio(audio_file)
            btn_detect = st.button("🚀 Extract Feature and Detect", type="primary", use_container_width=True)
            
            if btn_detect:
                if model_aksen:
                    with st.spinner("Sedang memproses..."):
                        # Gunakan context manager agar file tertutup otomatis
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                            tmp.write(audio_file.getbuffer())
                            tmp_path = tmp.name

                        try:
                            hasil_aksen = predict_accent(tmp_path, model_aksen)
                            
                            # Cari Metadata
                            user_info = None
                            if df_metadata is not None:
                                match = df_metadata[df_metadata['file_name'] == audio_file.name]
                                if not match.empty:
                                    user_info = match.iloc[0].to_dict()

                            with col2:
                                st.subheader("📊 Hasil Analisis")
                                st.info(f"### Aksen Terdeteksi: **{hasil_aksen}**")
                                st.write("---")
                                st.subheader("🔹 Info Pembicara")
                                if user_info:
                                    st.write(f"📅 **Usia:** {user_info.get('usia', '-')}")
                                    st.write(f"🗣️ **Gender:** {user_info.get('gender', '-')}")
                                    st.write(f"📍 **Provinsi:** {user_info.get('provinsi', '-')}")
                                else:
                                    st.warning("Data file ini tidak ditemukan di metadata.csv")
                        finally:
                            if os.path.exists(tmp_path):
                                os.remove(tmp_path)
                else:
                    st.error("Gagal menjalankan deteksi karena model offline.")

if __name__ == "__main__":
    main()
