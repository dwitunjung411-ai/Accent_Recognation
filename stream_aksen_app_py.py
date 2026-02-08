import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tempfile
import os
import tensorflow as tf

# ==========================================================
# 1. LOAD EMBEDDING MODEL (BUKAN PROTOTYPICAL)
# ==========================================================
@st.cache_resource
def load_embedding_model():
    model_name = "model_embedding_aksen.keras"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, model_name)

    if os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            return model
        except Exception as e:
            st.error(f"Gagal load model: {e}")
            return None
    return None

# ==========================================================
# 2. LOAD METADATA
# ==========================================================
@st.cache_data
def load_metadata_df():
    csv_path = "metadata.csv"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

# ==========================================================
# 3. EKSTRAKSI MFCC
# ==========================================================
def extract_mfcc(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc = np.mean(mfcc.T, axis=0)
    return mfcc

# ==========================================================
# 4. PREDIKSI AKSEN (VERSI BENAR)
# ==========================================================
def predict_accent(audio_path, model):
    try:
        mfcc = extract_mfcc(audio_path)
        mfcc = np.expand_dims(mfcc, axis=0)

        # Ambil embedding
        embedding = model.predict(mfcc)

        # === SIMPLE CLASSIFIER (CENTROID BASED) ===
        
        embedding = embedding[0]  # (embedding_dim,)

distances = {}
for cls, centroid in centroids.items():
    centroid = np.array(centroid)
    distances[cls] = np.linalg.norm(embedding - centroid)

predicted_class = min(distances, key=distances.get)




        distances = {}
        for cls, centroid in class_centroids.items():
            distances[cls] = np.linalg.norm(embedding - centroid)

        predicted_class = min(distances, key=distances.get)
        return predicted_class

    except Exception as e:
        return f"Error Analisis: {e}"

# ==========================================================
# 5. STREAMLIT UI
# ==========================================================
def main():
    st.set_page_config(
        page_title="Deteksi Aksen Indonesia",
        page_icon="🎙️",
        layout="wide"
    )

    model = load_embedding_model()
    metadata = load_metadata_df()

    st.title("🎙️ Sistem Deteksi Aksen Bahasa Indonesia")
    st.write("Aplikasi berbasis *Few-Shot Learning* menggunakan **embedding CNN**.")
    st.divider()

    with st.sidebar:
        st.header("📌 Status Sistem")
        st.success("Model siap") if model else st.error("Model gagal")
        st.success("Metadata siap") if metadata is not None else st.warning("Metadata kosong")
        st.caption("Skripsi Project · 2026")

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader("📥 Input Audio")
        audio_file = st.file_uploader("Upload audio (.wav / .mp3)", type=["wav", "mp3"])

        if audio_file:
            st.audio(audio_file)

            if st.button("🚀 Extract Feature and Detect", use_container_width=True):
                if model:
                    with st.spinner("Menganalisis suara..."):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                            tmp.write(audio_file.getbuffer())
                            tmp_path = tmp.name

                        hasil = predict_accent(tmp_path, model)
                        os.unlink(tmp_path)

                        with col2:
                            st.subheader("📊 Hasil Analisis")
                            st.success(f"🎭 **Aksen Terdeteksi:** {hasil}")

                            if metadata is not None:
                                match = metadata[metadata['file_name'] == audio_file.name]
                                if not match.empty:
                                    data = match.iloc[0]
                                    st.divider()
                                    st.subheader("👤 Info Pembicara")
                                    st.write(f"🎂 Usia: {data.get('usia', '-')}")
                                    st.write(f"🚻 Gender: {data.get('gender', '-')}")
                                    st.write(f"🗺️ Provinsi: {data.get('provinsi', '-')}")
                                else:
                                    st.info("Metadata tidak ditemukan.")
                else:
                    st.error("Model belum tersedia.")

if __name__ == "__main__":
    main()
