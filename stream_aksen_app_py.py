import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tempfile
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

@st.cache_resource
def load_model_from_url(model_url):
    """Load model langsung dari URL"""
    try:
        # Download model
        response = requests.get(model_url)
        response.raise_for_status()
        
        # Simpan sementara
        with tempfile.NamedTemporaryFile(delete=False, suffix='.keras') as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name
        
        # Load model
        custom_objects = {"PrototypicalNetwork": PrototypicalNetwork}
        model = tf.keras.models.load_model(tmp_path, custom_objects=custom_objects)
        
        # Hapus file temporary
        os.unlink(tmp_path)
        
        return model
    except Exception as e:
        st.error(f"Failed to load model from URL: {str(e)}")
        return None

@st.cache_resource
def load_numpy_from_url(url):
    """Load numpy array dari URL"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # Simpan sementara
        with tempfile.NamedTemporaryFile(delete=False, suffix='.npy') as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name
        
        # Load numpy
        data = np.load(tmp_path, allow_pickle=True)
        
        # Hapus file temporary
        os.unlink(tmp_path)
        
        return data
    except Exception as e:
        st.error(f"Failed to load data from URL: {str(e)}")
        return None

# Di load_resources():
def load_resources():
    # GitHub URLs (GANTI DENGAN URL ANDA)
    GITHUB_BASE = "https://github.com/dwitunjung411-ai/Accent_Recognation/"
    
    MODEL_URL = GITHUB_BASE + "model_detect_aksen.keras"
    SUPPORT_SET_URL = GITHUB_BASE + "support_set.npy"
    SUPPORT_LABELS_URL = GITHUB_BASE + "support_labels.npy"
    
    try:
        with st.spinner("Loading model from GitHub..."):
            model = load_model_from_url(MODEL_URL)
        
        with st.spinner("Loading support set from GitHub..."):
            support_set = load_numpy_from_url(SUPPORT_SET_URL)
        
        with st.spinner("Loading support labels from GitHub..."):
            support_labels = load_numpy_from_url(SUPPORT_LABELS_URL)
        
        if model and support_set is not None and support_labels is not None:
            st.success("✅ All resources loaded successfully from GitHub!")
            return model, support_set, support_labels
        else:
            st.error("Failed to load one or more resources.")
            return None, None, None
            
    except Exception as e:
        st.error(f"Error loading from GitHub: {str(e)}")
        return None, None, None

# ==========================================================
# 3. FUNGSI PREDIKSI (PERBAIKAN ERROR QUERY_SET)
# ==========================================================
def predict_accent(audio_path, model):
    if model is None: return "Model tidak tersedia"
    try:
        # Load & Preprocess
        y, sr = librosa.load(audio_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_scaled = np.mean(mfcc.T, axis=0)

        # Sesuai error: Model Prototypical seringkali butuh input dalam bentuk list
        # atau argumen bernama jika dibungkus class kustom
        input_data = np.expand_dims(mfcc_scaled, axis=0)

        # Mencoba prediksi langsung (seringkali model.predict cukup jika call() sudah benar)
        prediction = model.predict(input_data)

        aksen_classes = ["Sunda", "Jawa Tengah", "Jawa Timur", "Yogyakarta", "Betawi"]
        return aksen_classes[np.argmax(prediction)]
    except Exception as e:
        return f"Error Analisis: {str(e)}"

# ==========================================================
# 4. MAIN UI (WIDE LAYOUT & NEW ICONS)
# ==========================================================
def main():
    # Set layout lebar agar tidak sempit
    st.set_page_config(page_title="Deteksi Aksen Prototypical", page_icon="🎙️", layout="wide")

    model_aksen = load_accent_model()
    df_metadata = load_metadata_df()

    st.title("🎙️ Sistem Deteksi Aksen Prototypical Indonesia")
    st.write("Aplikasi berbasis *Few-Shot Learning* untuk klasifikasi aksen daerah.")
    st.divider()

    with st.sidebar:
        st.header("🛸 Status Sistem")
        if model_aksen:
            st.success("🤖 Model: Terhubung")
        else:
            st.error("🚫 Model: Terputus")

        if df_metadata is not None:
            st.success("📁 Metadata: Siap")
        else:
            st.warning("⚠️ Metadata: Kosong")

        st.divider()
        st.caption("Skripsi Project - 2026")

    # Pembagian kolom agar lebar
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader("📥 Input Audio")
        audio_file = st.file_uploader("Upload file (.wav, .mp3)", type=["wav", "mp3"])

        if audio_file:
            st.audio(audio_file)
            # Tombol diperlebar agar proporsional
            if st.button("🚀 Extract Feature and Detect", type="primary", use_container_width=True):
                if model_aksen:
                    with st.spinner("Menganalisis karakteristik suara..."):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                            tmp.write(audio_file.getbuffer())
                            tmp_path = tmp.name

                        hasil_aksen = predict_accent(tmp_path, model_aksen)

                        # Pencarian metadata
                        user_info = None
                        if df_metadata is not None:
                            match = df_metadata[df_metadata['file_name'] == audio_file.name]
                            if not match.empty:
                                user_info = match.iloc[0].to_dict()

                        with col2:
                            st.subheader("📊 Hasil Analisis")
                            # Gunakan container agar lebih rapi
                            with st.container(border=True):
                                st.markdown(f"#### 🎭 Aksen Terdeteksi:")
                                st.info(f"**{hasil_aksen}**")

                            st.divider()
                            st.subheader("💎 Info Pembicara")
                            if user_info:
                                # Variasi emoticon baru
                                st.markdown(f"🎂 **Usia:** {user_info.get('usia', '-')} Tahun")
                                st.markdown(f"🚻 **Gender:** {user_info.get('gender', '-')}")
                                st.markdown(f"🗺️ **Provinsi:** {user_info.get('provinsi', '-')}")
                            else:
                                st.warning("🕵️ Data file tidak terdaftar di metadata.csv")

                        os.unlink(tmp_path)
                else:
                    st.error("Gagal memproses: Model tidak ditemukan.")

if __name__ == "__main__":
    main()
