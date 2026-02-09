import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tempfile
import os
import tensorflow as tf

# ==========================================================
# 1. CUSTOM CLASS (TANPA REGISTER SERIALIZABLE)
# ==========================================================
class PrototypicalNetwork(tf.keras.Model):
    def __init__(self, embedding_model=None, **kwargs):
        super().__init__(**kwargs)
        self.embedding = embedding_model

    def call(self, support_set, query_set, support_labels, n_way):
        return self.embedding(query_set)

# ==========================================================
# 2. LOAD MODEL & METADATA
# ==========================================================
@st.cache_resource
def load_accent_model():
    model_path = "model_ditek.keras"
    if not os.path.exists(model_path):
        return None

    return tf.keras.models.load_model(
        model_path,
        custom_objects={"PrototypicalNetwork": PrototypicalNetwork},
        compile=False
    )

@st.cache_data
def load_metadata():
    if os.path.exists("metadata.csv"):
        return pd.read_csv("metadata.csv")
    return None

# ==========================================================
# 3. FEATURE EXTRACTION
# ==========================================================
def extract_mfcc(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

# ==========================================================
# 4. PREDIKSI AKSEN
# ==========================================================
def predict_accent(audio_path, model):
    features = extract_mfcc(audio_path)
    X = np.expand_dims(features, axis=0)
    preds = model.predict(X)

    aksen_labels = ["Sunda", "Jawa Tengah", "Jawa Timur", "Yogyakarta", "Betawi"]
    return aksen_labels[np.argmax(preds)]

# ==========================================================
# 5. STREAMLIT UI
# ==========================================================
def main():
    st.set_page_config(
        page_title="Accent Recognition",
        page_icon="🎙️",
        layout="wide"
    )

    model = load_accent_model()
    metadata = load_metadata()

    st.title("🎙️ Accent Recognition System")
    st.divider()

    with st.sidebar:
        st.header("⚙️ System Status")
        st.success("Model Loaded") if model else st.error("Model Not Found")

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader("📤 Upload Audio")
        audio_file = st.file_uploader(
            "Upload file audio (.wav)",
            type=["wav"]
        )

        if audio_file:
            st.audio(audio_file)

            if st.button("🚀 Detect Accent", use_container_width=True):
                if model is None:
                    st.error("Model gagal dimuat")
                    return

                with st.spinner("Menganalisis suara..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(audio_file.getbuffer())
                        tmp_path = tmp.name

                    hasil_aksen = predict_accent(tmp_path, model)

                    user_info = None
                    if metadata is not None:
                        row = metadata[metadata["file_name"] == audio_file.name]
                        if not row.empty:
                            user_info = row.iloc[0]

                    with col2:
                        st.subheader("📊 Hasil Analisis")
                        st.success(f"### 🗣️ Aksen: **{hasil_aksen}**")

                        st.divider()
                        st.subheader("👤 Metadata Pembicara")

                        if user_info is not None:
                            st.write(f"📅 **Usia** : {user_info['usia']}")
                            st.write(f"🧑 **Gender** : {user_info['gender']}")
                            st.write(f"📍 **Provinsi** : {user_info['provinsi']}")
                        else:
                            st.warning("Metadata tidak ditemukan")

                    os.remove(tmp_path)

if __name__ == "__main__":
    main()
