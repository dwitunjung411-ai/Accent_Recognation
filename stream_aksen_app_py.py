import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tempfile
import os
import tensorflow as tf

# ==========================================================
# LOAD MODEL
# ==========================================================
@st.cache(allow_output_mutation=True)
def load_embedding_model():
    return tf.keras.models.load_model("model_embedding_aksen.keras", compile=False)

# ==========================================================
# LOAD CENTROID
# ==========================================================
@st.cache(allow_output_mutation=True)
def load_centroids():
    return np.load("accent_centroids.npy", allow_pickle=True).item()

# ==========================================================
# LOAD METADATA
# ==========================================================
@st.cache(allow_output_mutation=True)
def load_metadata():
    if os.path.exists("metadata.csv"):
        return pd.read_csv("metadata.csv")
    return None

# ==========================================================
# MFCC
# ==========================================================
def extract_mfcc(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

# ==========================================================
# PREDIKSI AKSEN
# ==========================================================
def predict_accent(audio_path, model, centroids):
    mfcc = extract_mfcc(audio_path)
    mfcc = np.expand_dims(mfcc, axis=0)

    embedding = model.predict(mfcc)
    embedding = np.squeeze(embedding)

    distances = {}
    for label, centroid in centroids.items():
        centroid = np.squeeze(np.array(centroid))
        distances[label] = np.linalg.norm(embedding - centroid)

    return min(distances, key=distances.get)

# ==========================================================
# STREAMLIT UI
# ==========================================================
def main():
    st.title("🎙️ Deteksi Aksen Bahasa Indonesia")

    model = load_embedding_model()
    centroids = load_centroids()
    metadata = load_metadata()

    audio_file = st.file_uploader("Upload Audio", type=["wav", "mp3"])

    if audio_file:
        st.audio(audio_file)

        if st.button("🔍 Deteksi Aksen"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_file.getbuffer())
                tmp_path = tmp.name

            hasil = predict_accent(tmp_path, model, centroids)
            os.remove(tmp_path)

            st.success(f"🎯 Aksen Terdeteksi: **{hasil}**")

if __name__ == "__main__":
    main()
