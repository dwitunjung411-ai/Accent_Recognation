import streamlit as st
import numpy as np
import tensorflow as tf
import librosa
import os
import random

from tensorflow.keras.models import load_model

# =========================================================
# STREAMLIT CONFIG
# =========================================================
st.set_page_config(
    page_title="Deteksi Aksen Bahasa (Few-Shot Learning)",
    layout="wide"
)

# =========================================================
# FEATURE EXTRACTION
# =========================================================
def extract_mfcc_3channel(file, sr=22050, n_mfcc=40, max_len=174):
    y, sr = librosa.load(file, sr=sr)
    y = librosa.util.normalize(y)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    if mfcc.shape[1] < max_len:
        pad = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad)))
        delta = np.pad(delta, ((0, 0), (0, pad)))
        delta2 = np.pad(delta2, ((0, 0), (0, pad)))
    else:
        mfcc = mfcc[:, :max_len]
        delta = delta[:, :max_len]
        delta2 = delta2[:, :max_len]

    return np.stack([mfcc, delta, delta2], axis=-1).astype(np.float32)

# =========================================================
# PROTOTYPICAL NETWORK (INFERENSI ONLY)
# =========================================================
@tf.keras.utils.register_keras_serializable(package="Custom")
class PrototypicalNetwork(tf.keras.Model):
    def __init__(self, embedding_model=None, **kwargs):
        super().__init__(**kwargs)
        self.embedding = embedding_model

    def call(self, inputs, training=False):
        return self.embedding(inputs, training=training)

    def forward_few_shot(self, support_set, query_set, support_labels, n_way):
        support_emb = self.embedding(support_set, training=False)
        query_emb = self.embedding(query_set, training=False)

        support_emb = tf.math.l2_normalize(support_emb, axis=1)
        query_emb = tf.math.l2_normalize(query_emb, axis=1)

        prototypes = []
        for i in range(n_way):
            mask = tf.equal(support_labels, i)
            proto = tf.reduce_mean(tf.boolean_mask(support_emb, mask), axis=0)
            prototypes.append(proto)

        prototypes = tf.stack(prototypes)

        distances = tf.reduce_sum(
            tf.square(
                tf.expand_dims(query_emb, 1) -
                tf.expand_dims(prototypes, 0)
            ),
            axis=2
        )

        return -distances

    def get_config(self):
        config = super().get_config()
        config.update({
            "embedding_model": tf.keras.layers.serialize(self.embedding)
        })
        return config

    @classmethod
    def from_config(cls, config):
        embedding_cfg = config.pop("embedding_model")
        embedding_model = tf.keras.layers.deserialize(embedding_cfg)
        return cls(embedding_model=embedding_model, **config)

# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_trained_model():
    return load_model(
        "model_accent.keras",
        custom_objects={"PrototypicalNetwork": PrototypicalNetwork},
        compile=False
    )

model = load_trained_model()

# =========================================================
# LOAD SUPPORT SET (DIKUNCI SESUAI DATA)
# =========================================================
@st.cache_resource
def load_support_set():
    DATA_DIR = "data"

    classes = [
        "Betawi",
        "Sunda",
        "Jawa_Tengah",
        "Jawa_Timur",
        "YogyaKarta"
    ]

    k_shot = 5

    X, y = [], []
    label_map = {}

    for idx, cls in enumerate(classes):
        label_map[idx] = cls
        cls_path = os.path.join(DATA_DIR, cls)

        files = os.listdir(cls_path)
        selected = random.sample(files, k_shot)

        for f in selected:
            feat = extract_mfcc_3channel(os.path.join(cls_path, f))
            X.append(feat)
            y.append(idx)

    return np.array(X), np.array(y), label_map

support_set, support_labels, label_map = load_support_set()

# =========================================================
# UI
# =========================================================
st.title("🎧 Deteksi Aksen Bahasa (Few-Shot Learning)")
st.write("Dataset aksen: Betawi, Sunda, Jawa Tengah, Jawa Timur, Yogyakarta")

uploaded_file = st.file_uploader(
    "Upload audio (.wav / .mp3)",
    type=["wav", "mp3"]
)

# =========================================================
# PREDICTION
# =========================================================
if uploaded_file is not None:
    st.audio(uploaded_file)

    if st.button("🚀 Extract Feature and Detect"):
        with st.spinner("Menganalisis aksen..."):
            query_feat = extract_mfcc_3channel(uploaded_file)
            query_feat = np.expand_dims(query_feat, axis=0)

            logits = model.forward_few_shot(
                tf.convert_to_tensor(support_set),
                tf.convert_to_tensor(query_feat),
                tf.convert_to_tensor(support_labels),
                n_way=5
            )

            probs = tf.nn.softmax(logits, axis=1).numpy()[0]
            pred_idx = np.argmax(probs)
            pred_label = label_map[pred_idx]

        st.subheader("🎭 Aksen Terdeteksi")
        st.success(pred_label)

        st.subheader("📊 Confidence")
        for i, cls in label_map.items():
            st.write(f"{cls}: {probs[i] * 100:.2f}%")
