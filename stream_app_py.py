import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import tempfile
import os

from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# ==============================
# CONFIG
# ==============================
st.set_page_config(
    page_title="Accent Recognition (Few-Shot)",
    layout="centered"
)

# ==============================
# LOAD LABEL ENCODER
# ==============================
@st.cache_resource
def load_label_encoder(csv_path):
    metadata = pd.read_csv(csv_path)
    le = LabelEncoder()
    le.fit(metadata["label_aksen"])
    return le

# ==============================
# FEATURE EXTRACTION (MFCC)
# ==============================
def extract_mfcc(file_path, sr=22050, n_mfcc=40, max_len=174):
    y, sr = librosa.load(file_path, sr=sr)
    y = librosa.util.normalize(y)

    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc, n_fft=2048, hop_length=512
    )

    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
    else:
        mfcc = mfcc[:, :max_len]

    # reshape → (H, W, C)
    mfcc = mfcc[..., np.newaxis]
    return mfcc.astype(np.float32)

# ==============================
# PROTOTYPICAL NETWORK
# ==============================
class PrototypicalNetwork(tf.keras.Model):
    def __init__(self, embedding_model, **kwargs):
        super().__init__(**kwargs)
        self.embedding = embedding_model

    def call(self, support_set, query_set, support_labels, n_way):
        support_emb = self.embedding(support_set)
        query_emb = self.embedding(query_set)

        prototypes = []
        for i in range(n_way):
            mask = tf.equal(support_labels, i)
            class_emb = tf.boolean_mask(support_emb, mask)
            proto = tf.reduce_mean(class_emb, axis=0)
            prototypes.append(proto)
        prototypes = tf.stack(prototypes)

        dists = []
        for q in query_emb:
            dists.append(tf.norm(prototypes - q, axis=1))
        dists = tf.stack(dists)

        return -dists

# ==============================
# LOAD MODEL
# ==============================
@st.cache_resource
def load_fewshot_model():
    custom_objects = {"PrototypicalNetwork": PrototypicalNetwork}
    return load_model("model_aksen.keras", custom_objects=custom_objects)

# ==============================
# CREATE SUPPORT SET
# ==============================
@st.cache_resource
def build_support_set(metadata_csv, audio_dir, n_way=5, k_shot=3):
    df = pd.read_csv(metadata_csv)
    le = LabelEncoder()
    df["label_id"] = le.fit_transform(df["label_aksen"])

    support_x = []
    support_y = []

    for label in df["label_id"].unique()[:n_way]:
        samples = df[df["label_id"] == label].head(k_shot)
        for _, row in samples.iterrows():
            path = os.path.join(audio_dir, row["file_name"])
            mfcc = extract_mfcc(path)
            support_x.append(mfcc)
            support_y.append(label)

    return np.array(support_x), np.array(support_y), le

# ==============================
# DETECT FUNCTION (INI KUNCI)
# ==============================
def detect_accent(audio_path, model, support_x, support_y, label_encoder):
    mfcc = extract_mfcc(audio_path)
    query = mfcc[np.newaxis, ...]

    logits = model(
        tf.convert_to_tensor(support_x),
        tf.convert_to_tensor(query),
        tf.convert_to_tensor(support_y),
        n_way=len(np.unique(support_y))
    )

    pred_id = tf.argmax(logits, axis=1).numpy()[0]
    return label_encoder.inverse_transform([pred_id])[0]

# ==============================
# MAIN APP
# ==============================
def main():
    st.title("🎤 Accent Recognition (Few-Shot Learning)")

    audio_file = st.file_uploader(
        "Upload Audio (.wav / .mp3)",
        type=["wav", "mp3"]
    )

    if audio_file:
        st.audio(audio_file)

        if st.button("🚀 Extract Features and Detect"):
            with st.spinner("Processing..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(audio_file.getbuffer())
                    audio_path = tmp.name

                model = load_fewshot_model()
                support_x, support_y, le = build_support_set(
                    "metadata.csv",
                    "Voice_Skripsi_fix"
                )

                pred = detect_accent(
                    audio_path,
                    model,
                    support_x,
                    support_y,
                    le
                )

                st.success(f"🎯 Prediksi Aksen: **{pred}**")

# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    main()
