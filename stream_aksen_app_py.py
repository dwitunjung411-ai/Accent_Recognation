import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tempfile
import os
import tensorflow as tf

# ==========================================================
# 1. DEFINISI CLASS PROTOTYPICAL (WAJIB ADA)
# ==========================================================
@tf.keras.utils.register_keras_serializable()
class PrototypicalNetwork(tf.keras.Model):
    def __init__(self, embedding_model=None, **kwargs):
        super().__init__(**kwargs)
        self.embedding = embedding_model

    def call(self, support_set, query_set, support_labels=None, n_way=None):
        return self.embedding(query_set)

# ==========================================================
# 2. LOAD EMBEDDING MODEL (AMBIL CNN SAJA)
# ==========================================================
@st.cache_resource
def load_embedding_model():
    proto = tf.keras.models.load_model(
        "model_embedding_aksen.keras",
        compile=False,
        custom_objects={"PrototypicalNetwork": PrototypicalNetwork}
    )
    return proto.embedding   # 🔥 INI KUNCI UTAMA

# ==========================================================
# 3. LOAD CENTROID
# ==========================================================
@st.cache_resource
def load_centroids():
    return np.load("accent_centroids.npy", allow_pickle=True).item()

# ==========================================================
# 4. LOAD METADATA (OPSIONAL)
# ==========================================================
@st.cache_data
def load_metadata():
    if os.path.exists("metadata.csv"):
        return pd.read_csv("metadata.csv")
    return None

# ==========================================================
# 5. EKSTRAKSI MFCC
# ==========================================================
def extract_mfcc(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

# ==========================================================
# 6. PREDIKSI AKSEN (FIX TOTAL)
# ==========================================================
def predict_accent(audio_path, model, centroids):
    mfcc = extract_mfcc(audio_path)
    mfcc = np.expand_dims(mfcc, axis=0)

    embedding = model.predict(mfcc, verbose=0)
    embedding = np.squeeze(embedding)

    distances = {}
    for cls, centroid in centroids.items():
        centroid = np.squeeze(np.array(centroid))
        distances[cls] = np.linalg.norm(embedding - centroid)

    return min(distances, key=distances.get)

# ==========================================================
# 7. STREAMLIT UI
# ==========================================================
def main():
    st.set_page_config(
        page_title="Deteksi Aksen Indonesia",
        page_icon="🎙️",
        layout="wide"
    )

    st.title("🎙️ Deteksi Aksen Bahasa Indonesia")
    st.write("Few-Shot Learning berbasis **Prototypical Network (Embedding CNN)**")
    st.divider()

    model = load_embedding_model()
    centroids = load_centroids()
    metadata = load_metadata()

    with st.sidebar:
        st.header("📌 Status Sistem")
        st.success("Model siap")
        st.success("Centroid siap")
        st.caption("Skripsi Project · 2026")

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader("📥 Upload Audio")
        audio_file = st.file_uploader(
            "Upload file (.wav / .mp3)",
            type=["wav", "mp3"]
        )

        if audio_file:
            st.audio(audio_file)

            if st.button("🚀 Deteksi Aksen", use_container_width=True):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(audio_file.getbuffer())
                    audio_path = tmp.name

                with st.spinner("Menganalisis suara..."):
                    hasil = predict_accent(audio_path, model, centroids)

                os.unlink(audio_path)

                with col2:
                    st.subheader("📊 Hasil Analisis")
                    st.success(f"🎭 **Aksen Terdeteksi:** {hasil}")

                    if metadata is not None:
                        match = metadata[metadata["file_name"] == audio_file.name]
                        if not match.empty:
                            data = match.iloc[0]
                            st.divider()
                            st.subheader("👤 Info Pembicara")
                            st.write(f"🎂 Usia: {data.get('usia','-')}")
                            st.write(f"🚻 Gender: {data.get('gender','-')}")
                            st.write(f"🗺️ Provinsi: {data.get('provinsi','-')}")

# ==========================================================
if __name__ == "__main__":
    main()
