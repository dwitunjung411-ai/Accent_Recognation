import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tempfile
import os
import tensorflow as tf

# ==========================================================
# 1. CUSTOM MODEL CLASS (JANGAN PAKAI DECORATOR)
# ==========================================================
class PrototypicalNetwork(tf.keras.Model):
    def __init__(self, embedding_model=None, **kwargs):
        super().__init__(**kwargs)
        self.embedding = embedding_model

    def call(self, support_set, query_set, support_labels=None, n_way=None):
        # inference only
        return self.embedding(query_set)

# ==========================================================
# 2. LOAD MODEL
# ==========================================================
def load_accent_model():
    model_path = "model_ditek.keras"

    # DEBUG (boleh dihapus setelah normal)
    st.write("📂 File di direktori:", os.listdir("."))
    st.write("📦 Model path:", model_path)
    st.write("✅ Exists:", os.path.exists(model_path))

    if not os.path.exists(model_path):
        st.error("❌ File model_ditek.keras tidak ditemukan")
        return None

    try:
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={"PrototypicalNetwork": PrototypicalNetwork},
            compile=False
        )
        return model
    except Exception as e:
        st.error(f"❌ Gagal load model: {e}")
        return None

# ==========================================================
# 3. LOAD METADATA
# ==========================================================
@st.cache_data
def load_metadata():
    if os.path.exists("metadata.csv"):
        return pd.read_csv("metadata.csv")
    return None

# ==========================================================
# 4. FEATURE EXTRACTION
# ==========================================================
def extract_mfcc(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc_mean

# ==========================================================
# 5. PREDIKSI AKSEN
# ==========================================================
def predict_accent(audio_path, model):
    features = extract_mfcc(audio_path)
    X = np.expand_dims(features, axis=0)

    preds = model.predict(X)

    aksen_labels = [
        "Sunda",
        "Jawa Tengah",
        "Jawa Timur",
        "Yogyakarta",
        "Betawi"
    ]

    return aksen_labels[np.argmax(preds)]

# ==========================================================
# 6. STREAMLIT UI
# ==========================================================
def main():
    st.set_page_config(
        page_title="Accent Recognition",
        page_icon="🎙️",
        layout="wide"
    )

    st.title("🎙️ Accent Recognition System")
    st.divider()

    model = load_accent_model()
    metadata = load_metadata()

    with st.sidebar:
        st.header("⚙️ Status Sistem")
        if model:
            st.success("Model: Loaded")
        else:
            st.error("Model: Not Loaded")

    col1, col2 = st.columns([1, 1.2])

    # ===================== INPUT =====================
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
                    st.error("Model tidak tersedia")
                    return

                with st.spinner("Menganalisis suara..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(audio_file.getbuffer())
                        tmp_path = tmp.name

                    # Prediksi
                    hasil_aksen = predict_accent(tmp_path, model)

                    # Ambil metadata
                    user_info = None
                    if metadata is not None:
                        row = metadata[metadata["file_name"] == audio_file.name]
                        if not row.empty:
                            user_info = row.iloc[0]

                    # ===================== OUTPUT =====================
                    with col2:
                        st.subheader("📊 Hasil Analisis")
                        st.success(f"### 🗣️ Aksen: **{hasil_aksen}**")

                        st.divider()
                        st.subheader("👤 Informasi Pembicara")

                        if user_info is not None:
                            st.write(f"📅 **Usia** : {user_info['usia']}")
                            st.write(f"🧑 **Gender** : {user_info['gender']}")
                            st.write(f"📍 **Provinsi** : {user_info['provinsi']}")
                        else:
                            st.warning("Metadata tidak ditemukan")

                    os.remove(tmp_path)

# ==========================================================
if __name__ == "__main__":
    main()
