import streamlit as st
import numpy as np
import librosa
import tempfile
import os
import tensorflow as tf

# ==========================================================
# 1. DEFINISI PROTOTYPICAL NETWORK (WAJIB SAMA)
# ==========================================================
@tf.keras.utils.register_keras_serializable()
class PrototypicalNetwork(tf.keras.Model):
    def __init__(self, embedding_model=None, **kwargs):
        super().__init__(**kwargs)
        self.embedding = embedding_model

    def call(self, support_set, query_set, support_labels=None, n_way=None):
        return self.embedding(query_set)

# ==========================================================
# 2. LOAD MODEL → AMBIL EMBEDDING CNN SAJA
# ==========================================================
@st.cache_resource
def load_embedding_model():
    proto = tf.keras.models.load_model(
        "model_embedding_aksen.keras",
        compile=False,
        custom_objects={"PrototypicalNetwork": PrototypicalNetwork}
    )

    # 🔥 INI KUNCI UTAMA
    embedding_model = proto.embedding
    return embedding_model

# ==========================================================
# 3. LOAD CENTROID
# ==========================================================
@st.cache_resource
def load_centroids():
    return np.load("accent_centroids.npy", allow_pickle=True).item()

# ==========================================================
# 4. EKSTRAKSI MFCC
# ==========================================================
def extract_mfcc(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

# ==========================================================
# 5. PREDIKSI AKSEN (TANPA PROTOTYPICAL CALL)
# ==========================================================
def predict_accent(audio_path, embedding_model, centroids):
    mfcc = extract_mfcc(audio_path)
    mfcc = np.expand_dims(mfcc, axis=0)

    embedding = embedding_model.predict(mfcc, verbose=0)
    embedding = np.squeeze(embedding)

    distances = {}
    for cls, centroid in centroids.items():
        centroid = np.squeeze(np.array(centroid))
        distances[cls] = np.linalg.norm(embedding - centroid)

    return min(distances, key=distances.get)

# ==========================================================
# 6. STREAMLIT UI
# ==========================================================
def main():
    st.set_page_config(
        page_title="Deteksi Aksen Indonesia",
        page_icon="🎙️",
        layout="wide"
    )

    st.title("🎙️ Deteksi Aksen Bahasa Indonesia")
    st.divider()

    embedding_model = load_embedding_model()
    centroids = load_centroids()

    st.sidebar.success("Model embedding siap")
    st.sidebar.success("Centroid siap")

    audio_file = st.file_uploader("Upload audio (.wav / .mp3)", type=["wav", "mp3"])

    if audio_file:
        st.audio(audio_file)

        if st.button("🚀 Deteksi Aksen"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_file.getbuffer())
                path = tmp.name

            with st.spinner("Menganalisis suara..."):
                hasil = predict_accent(path, embedding_model, centroids)

            os.unlink(path)

            st.success(f"🎭 Aksen Terdeteksi: **{hasil}**")

# ==========================================================
if __name__ == "__main__":
    main()
