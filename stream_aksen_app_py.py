import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import os
import tempfile

# ==============================
# CONFIG
# ==============================
SAMPLE_RATE = 16000
N_MFCC = 40
MAX_LEN = 300

st.set_page_config(
    page_title="Deteksi Aksen Bahasa Indonesia",
    layout="centered"
)

st.title("🎙️ Deteksi Aksen Bahasa Indonesia")
st.caption("Few-Shot Learning (Prototypical Network – Inference)")

# ==============================
# LOAD ENCODER MODEL
# ==============================
@st.cache_resource
def load_encoder():
    return tf.keras.models.load_model("model/encoder.h5")

encoder = load_encoder()

# ==============================
# FEATURE EXTRACTION
# ==============================
def extract_mfcc(audio_path):
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=N_MFCC
    )

    mfcc = mfcc.T

    if mfcc.shape[0] > MAX_LEN:
        mfcc = mfcc[:MAX_LEN]
    else:
        pad_len = MAX_LEN - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad_len), (0, 0)))

    return mfcc.astype(np.float32)

# ==============================
# BUILD PROTOTYPES
# ==============================
@st.cache_data
def build_prototypes():
    df = pd.read_csv("data/metadata.csv")

    embeddings = {}
    counts = {}

    for _, row in df.iterrows():
        audio_path = os.path.join("data/audio", row["filename"])
        label = row["aksen"]

        mfcc = extract_mfcc(audio_path)
        mfcc = np.expand_dims(mfcc, axis=0)

        emb = encoder.predict(mfcc, verbose=0)[0]

        if label not in embeddings:
            embeddings[label] = emb
            counts[label] = 1
        else:
            embeddings[label] += emb
            counts[label] += 1

    for label in embeddings:
        embeddings[label] /= counts[label]

    return embeddings

prototypes = build_prototypes()
labels = list(prototypes.keys())

# ==============================
# PREDICTION
# ==============================
def predict_accent(audio_path):
    mfcc = extract_mfcc(audio_path)
    mfcc = np.expand_dims(mfcc, axis=0)

    query_emb = encoder.predict(mfcc, verbose=0)[0]

    distances = {}
    for label, proto in prototypes.items():
        distances[label] = float(np.linalg.norm(query_emb - proto))

    pred_label = min(distances, key=distances.get)
    return pred_label, distances

# ==============================
# STREAMLIT UI
# ==============================
uploaded_file = st.file_uploader(
    "Upload audio (.wav)",
    type=["wav"]
)

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        temp_audio_path = tmp.name

    st.audio(temp_audio_path)

    with st.spinner("🔍 Menganalisis aksen..."):
        pred, dist = predict_accent(temp_audio_path)

    st.success(f"🗣️ **Aksen terdeteksi: {pred}**")

    st.subheader("📊 Jarak ke Prototype")
    st.json(dist)
