import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import os

# ================================
# CONFIG
# ================================
SAMPLE_RATE = 16000
N_MELS = 40
MAX_LEN = 300

st.set_page_config(page_title="Deteksi Aksen", layout="centered")
st.title("🎙️ Deteksi Aksen Bahasa Indonesia")

# ================================
# PROTOTYPICAL NETWORK (INFERENCE)
# ================================
@tf.keras.utils.register_keras_serializable(package="Custom")
class PrototypicalNetwork(tf.keras.Model):
    def __init__(self, encoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder

    def call(self, x):
        return self.encoder(x)

# ================================
# LOAD MODEL
# ================================
@st.cache_resource
def load_proto_model():
    return tf.keras.models.load_model(
        "model/proto_model.h5",
        custom_objects={"PrototypicalNetwork": PrototypicalNetwork}
    )

model = load_proto_model()

# ================================
# FEATURE EXTRACTION
# ================================
def extract_mfcc(audio_path):
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=N_MELS
    )

    mfcc = mfcc.T
    if mfcc.shape[0] > MAX_LEN:
        mfcc = mfcc[:MAX_LEN]
    else:
        pad = MAX_LEN - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad), (0, 0)))

    return mfcc.astype(np.float32)

# ================================
# LOAD METADATA & PROTOTYPE
# ================================
@st.cache_data
def build_prototypes():
    df = pd.read_csv("data/metadata.csv")

    embeddings = {}
    counts = {}

    for _, row in df.iterrows():
        path = os.path.join("data/audio", row["filename"])
        aksen = row["aksen"]

        mfcc = extract_mfcc(path)
        mfcc = np.expand_dims(mfcc, axis=0)

        emb = model(mfcc).numpy()[0]

        if aksen not in embeddings:
            embeddings[aksen] = emb
            counts[aksen] = 1
        else:
            embeddings[aksen] += emb
            counts[aksen] += 1

    for k in embeddings:
        embeddings[k] /= counts[k]

    return embeddings

prototypes = build_prototypes()
labels = list(prototypes.keys())

# ================================
# PREDICTION
# ================================
def predict_accent(audio_path):
    mfcc = extract_mfcc(audio_path)
    mfcc = np.expand_dims(mfcc, axis=0)

    query_emb = model(mfcc).numpy()[0]

    distances = {}
    for label, proto in prototypes.items():
        distances[label] = np.linalg.norm(query_emb - proto)

    pred = min(distances, key=distances.get)
    return pred, distances

# ================================
# STREAMLIT UI
# ================================
uploaded_file = st.file_uploader(
    "Upload audio (.wav)", type=["wav"]
)

if uploaded_file:
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())

    st.audio("temp.wav")

    with st.spinner("Menganalisis aksen..."):
        pred, dist = predict_accent("temp.wav")

    st.success(f"🗣️ Aksen terdeteksi: **{pred}**")

    st.subheader("📊 Jarak ke Prototype")
    st.json(dist)
