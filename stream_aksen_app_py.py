import streamlit as st
import numpy as np
import librosa
import tempfile
import tensorflow as tf
import os

st.set_page_config(page_title="Accent Test", layout="centered")

# =========================
# LOAD MODEL
# =========================
def load_model():
    return tf.keras.models.load_model("model_embedding_aksen.keras", compile=False)

# =========================
# MFCC
# =========================
def extract_mfcc(path):
    y, sr = librosa.load(path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

# =========================
# PREDICT (TEST SAJA)
# =========================
def predict(audio_path, model):
    mfcc = extract_mfcc(audio_path)
    mfcc = np.expand_dims(mfcc, axis=0)
    emb = model.predict(mfcc)
    return emb.shape

# =========================
# UI
# =========================
st.title("🎙️ Test Model Embedding")

try:
    model = load_model()
    st.success("✅ Model berhasil di-load")
except Exception as e:
    st.error(f"❌ Gagal load model: {e}")
    st.stop()

audio = st.file_uploader("Upload audio", type=["wav", "mp3"])

if audio:
    st.audio(audio)

    if st.button("TEST MODEL"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio.getbuffer())
            path = tmp.name

        try:
            shape = predict(path, model)
            st.success(f"🎯 Model jalan | Embedding shape: {shape}")
        except Exception as e:
            st.error(f"❌ Error predict: {e}")

        os.remove(path)
