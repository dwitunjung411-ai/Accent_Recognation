# ==========================================================
# 0. FORCE CPU (WAJIB UNTUK STREAMLIT CLOUD)
# ==========================================================
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ==========================================================
# 1. IMPORT LIBRARY
# ==========================================================
import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tempfile
import tensorflow as tf
from tensorflow.keras.models import load_model

# Pastikan TensorFlow tidak mencari GPU
try:
    tf.config.set_visible_devices([], 'GPU')
except:
    pass

# ==========================================================
# 2. DEFINISI CLASS PROTOTYPICAL NETWORK
# ==========================================================
@tf.keras.utils.register_keras_serializable(package="Custom")
class PrototypicalNetwork(tf.keras.Model):
    def __init__(self, embedding_model=None, **kwargs):
        super().__init__(**kwargs)
        self.embedding = embedding_model

    def call(self, inputs, training=False):
        # Model embedding langsung dipakai
        return self.embedding(inputs)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embedding_model": tf.keras.layers.serialize(self.embedding)
        })
        return config

# ==========================================================
# 3. LOAD MODEL & METADATA
# ==========================================================
@st.cache_resource
def load_accent_model():
    model_name = "model_embedding_aksen.keras"
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, model_name)

    if not os.path.exists(model_path):
        return None

    try:
        custom_objects = {"PrototypicalNetwork": PrototypicalNetwork}
        model = load_model(
            model_path,
            custom_objects=custom_objects,
            compile=False
        )
        return model
    except Exception as e:
        st.error(f"Gagal load model: {e}")
        return None


@st.cache_data
def load_metadata_df():
    csv_path = "metadata.csv"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

# ==========================================================
# 4. FUNGSI PREDIKSI AKSEN (CPU SAFE)
# ==========================================================
def predict_accent(audio_path, model):
    if model is None:
        return "Model tidak tersedia"

    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=16000)

        # Ekstraksi MFCC
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=40
        )

        mfcc_mean = np.mean(mfcc.T, axis=0)
        input_data = np.expand_dims(mfcc_mean, axis=0)

        # Prediksi
        preds = model.predict(input_data, verbose=0)

        aksen_classes = [
            "Sunda",
            "Jawa Tengah",
            "Jawa Timur",
            "Yogyakarta",
            "Betawi"
        ]

        return aksen_classes[int(np.argmax(preds))]

    except Exception as e:
        return f"Error Analisis: {str(e)}"

# ==========================================================
# 5. STREAMLIT UI
# ==========================================================
def main():
    st.set_page_config(
        page_title="Deteksi Aksen Prototypical",
        page_icon="🎙️",
        layout="wide"
    )

    st.title("🎙️ Sistem Deteksi Aksen Prototypical Indonesia")
    st.write("Aplikasi berbasis **Few-Shot Learning** untuk klasifikasi aksen daerah.")
    st.divider()

    model_aksen = load_accent_model()
    df_metadata = load_metadata_df()

    # ================= SIDEBAR =================
    with st.sidebar:
        st.header("🛸 Status Sistem")

        if model_aksen:
            st.success("🤖 Model: Terhubung")
        else:
            st.error("🚫 Model: Tidak ditemukan")

        if df_metadata is not None:
            st.success("📁 Metadata: Siap")
        else:
            st.warning("⚠️ Metadata: Tidak tersedia")

        st.divider()
        st.caption("Skripsi Informatika • 2026")

    # ================= MAIN =================
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader("📥 Input Audio")
        audio_file = st.file_uploader(
            "Upload audio (.wav / .mp3)",
            type=["wav", "mp3"]
        )

        if audio_file:
            st.audio(audio_file)

            if st.button(
                "🚀 Extract Feature and Detect",
                type="primary",
                use_container_width=True
            ):
                if model_aksen:
                    with st.spinner("Menganalisis suara..."):
                        with tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=".wav"
                        ) as tmp:
                            tmp.write(audio_file.getbuffer())
                            tmp_path = tmp.name

                        hasil_aksen = predict_accent(tmp_path, model_aksen)

                        # Cari metadata
                        user_info = None
                        if df_metadata is not None:
                            match = df_metadata[
                                df_metadata["file_name"] == audio_file.name
                            ]
                            if not match.empty:
                                user_info = match.iloc[0].to_dict()

                        with col2:
                            st.subheader("📊 Hasil Analisis")
                            with st.container(border=True):
                                st.markdown("#### 🎭 Aksen Terdeteksi")
                                st.success(hasil_aksen)

                            st.divider()
                            st.subheader("💎 Info Pembicara")
                            if user_info:
                                st.markdown(f"🎂 **Usia:** {user_info.get('usia', '-')}")
                                st.markdown(f"🚻 **Gender:** {user_info.get('gender', '-')}")
                                st.markdown(f"🗺️ **Provinsi:** {user_info.get('provinsi', '-')}")
                            else:
                                st.warning("Metadata tidak ditemukan")

                        os.unlink(tmp_path)
                else:
                    st.error("Model tidak tersedia.")

if __name__ == "__main__":
    main()
