import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import tempfile
import os

# ===============================
# CONFIG
# ===============================
N_WAY = 5
SR = 22050

st.set_page_config(
    page_title="Deteksi Aksen - Few Shot",
    layout="centered"
)

# ===============================
# MODEL DEFINITION (HARUS SAMA)
# ===============================
@tf.keras.utils.register_keras_serializable()
class PrototypicalNetwork(tf.keras.Model):
    def __init__(self, embedding_model, **kwargs):
        super().__init__(**kwargs)
        self.embedding = embedding_model

    def call(self, inputs):
        """
        inputs = (support_set, query_set, support_labels, n_way)
        """
        support_set, query_set, support_labels, n_way = inputs

        support_emb = self.embedding(support_set)
        query_emb = self.embedding(query_set)

        prototypes = []
        for i in range(n_way):
            mask = tf.equal(support_labels, i)
            class_emb = tf.boolean_mask(support_emb, mask)
            proto = tf.reduce_mean(class_emb, axis=0)
            prototypes.append(proto)

        prototypes = tf.stack(prototypes)

        distances = tf.norm(
            tf.expand_dims(query_emb, 1) - tf.expand_dims(prototypes, 0),
            axis=2
        )

        return -distances

# ===============================
# LOAD MODEL & SUPPORT SET
# ===============================
@st.cache_resource
def load_all():
    model = tf.keras.models.load_model(
        "model_aksen.keras",
        custom_objects={"PrototypicalNetwork": PrototypicalNetwork}
    )

    support_set = np.load("support_set.npy")
    support_labels = np.load("support_labels.npy")

    return model, support_set, support_labels

model, support_set, support_labels = load_all()

# ===============================
# FEATURE EXTRACTION (SAMA DENGAN TRAINING)
# ===============================
def extract_mfcc(audio_path, sr=22050, n_mfcc=40, max_len=174):
    y, sr = librosa.load(audio_path, sr=sr)
    y = librosa.util.normalize(y)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=2048,
        hop_length=512
    )

    if mfcc.shape[1] < max_len:
        pad = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0,0),(0,pad)), mode="constant")
    else:
        mfcc = mfcc[:, :max_len]

    mfcc = mfcc[..., np.newaxis]  # (40,174,1)
    return mfcc.astype(np.float32)

# ===============================
# STREAMLIT UI
# ===============================
st.title("🎙️ Deteksi Aksen Bahasa (Few-Shot Learning)")
st.write("Model Prototypical Network")

audio_file = st.file_uploader(
    "Upload audio (.wav / .mp3)",
    type=["wav", "mp3"]
)

if audio_file:
    st.audio(audio_file)

    if st.button("🚀 Deteksi Aksen"):
        with st.spinner("Memproses audio..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_file.read())
                audio_path = tmp.name

            # QUERY FEATURE
            query_feat = extract_mfcc(audio_path)
            query_feat = query_feat[np.newaxis, ...]  # (1,H,W,C)

            # CONVERT TO TENSOR
            support_tensor = tf.convert_to_tensor(support_set, tf.float32)
            query_tensor = tf.convert_to_tensor(query_feat, tf.float32)
            support_labels_tensor = tf.convert_to_tensor(support_labels, tf.int32)

            # MODEL CALL (INI KUNCI)
            logits = model.call(
                support_tensor,
                query_tensor,
                support_labels_tensor,
                N_WAY
            )

            probs = tf.nn.softmax(logits, axis=1).numpy()[0]
            pred_idx = np.argmax(probs)

            aksen_classes = [
                "Sunda",
                "Jawa Tengah",
                "Jawa Timur",
                "Yogyakarta",
                "Betawi"
            ]

            st.success(f"🎭 **Aksen Terdeteksi: {aksen_classes[pred_idx]}**")

            st.write("Probabilitas:")
            for i, cls in enumerate(aksen_classes):
                st.write(f"{cls}: {probs[i]*100:.2f}%")

            os.unlink(audio_path)
