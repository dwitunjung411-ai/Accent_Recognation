import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tempfile
import os
import tensorflow as tf

# ==========================================================
# 1. DEFINISI CLASS (Disesuaikan agar tidak error 'str' object)
# ==========================================================
@tf.keras.utils.register_keras_serializable(package="Custom")
class PrototypicalNetwork(tf.keras.Model):
    def __init__(self, embedding_model=None, **kwargs):
        super(PrototypicalNetwork, self).__init__(**kwargs)
        self.embedding = embedding_model

    def call(self, inputs):
        return self.embedding(inputs)

    def get_config(self):
        config = super().get_config()
        # Menggunakan serialize yang lebih aman untuk versi Keras terbaru
        config.update({"embedding_model": tf.keras.layers.serialize(self.embedding)})
        return config

# ==========================================================
# 2. LOAD RESOURCES (MODEL & SUPPORT SET)
# ==========================================================
@st.cache_resource
def load_system_resources():
    # Nama file disesuaikan dengan yang ada di GitHub Anda (image_8401e7.png)
    model_file = "model_embedding_aksen.keras" 
    support_set_file = "support_set.npy"
    support_labels_file = "support_labels.npy"
    
    try:
        # Load Model dengan custom_objects
        custom_objects = {"PrototypicalNetwork": PrototypicalNetwork}
        model = tf.keras.models.load_model(model_file, custom_objects=custom_objects, compile=False)
        
        # Load Support Set untuk klasifikasi Few-Shot
        s_set = np.load(support_set_file)
        s_labels = np.load(support_labels_file)
        
        return model, s_set, s_labels
    except Exception as e:
        # Menampilkan pesan error spesifik di sidebar jika gagal
        st.sidebar.error(f"Error Loading: {str(e)}")
        return None, None, None

# ==========================================================
# 3. FUNGSI PREDIKSI (Few-Shot Logic)
# ==========================================================
def predict_accent_few_shot(audio_path, model, s_set, s_labels):
    try:
        # 1. Load dan Preprocessing Audio Query
        y, sr = librosa.load(audio_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        query_feat = np.mean(mfcc.T, axis=0).reshape(1, -1)
        
        # 2. Ekstraksi Embedding (vektor fitur)
        query_embed = model.predict(query_feat, verbose=0)
        support_embed = model.predict(s_set, verbose=0)
        
        # 3. Hitung Prototype (Rata-rata embedding per kelas)
        # Daftar kelas sesuai urutan di metadata Anda
        aksen_classes = ["Sunda", "Jawa Tengah", "Jawa Timur", "Yogyakarta", "Betawi"]
        prototypes = []
        for i in range(len(aksen_classes)):
            # Hitung rata-rata vektor untuk setiap label kelas
            p = np.mean(support_embed[s_labels == i], axis=0)
            prototypes.append(p)
        prototypes = np.array(prototypes)
        
        # 4. Cari Jarak Terdekat (Euclidean Distance)
        distances = np.linalg.norm(prototypes - query_embed, axis=1)
        result_idx = np.argmin(distances)
        
        return aksen_classes[result_idx]
    except Exception as e:
        return f"Error Analisis: {str(e)}"

# ==========================================================
# 4. MAIN UI
# ==========================================================
def main():
    st.set_page_config(page_title="Accent Recognition", layout="wide")
    
    # Load semua file sekaligus
    model_aksen, s_set, s_labels = load_system_resources()
    df_metadata = pd.read_csv("metadata.csv") if os.path.exists("metadata.csv") else None

    st.title("🎙️ Accent Recognition")
    st.divider()

    with st.sidebar:
        st.header("⚙️ Status Sistem")
        if model_aksen is not None:
            st.success("Model & Support Set: Online")
        else:
            st.error("Sistem: Offline")
            st.info("Pastikan file .keras dan .npy ada di folder root.")

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader("📤 Input Audio")
        audio_file = st.file_uploader("Upload file (.wav, .mp3)", type=["wav", "mp3"])

        if audio_file:
            st.audio(audio_file)
            if st.button("🚀 Detect Accent", type="primary", use_container_width=True):
                if model_aksen is not None:
                    with st.spinner("Menganalisis aksen..."):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                            tmp.write(audio_file.getbuffer())
                            tmp_path = tmp.name

                        hasil = predict_accent_few_shot(tmp_path, model_aksen, s_set, s_labels)

                        with col2:
                            st.subheader("📊 Hasil Analisis")
                            st.info(f"### Aksen Terdeteksi: **{hasil}**")
                            
                            # Cek Metadata
                            if df_metadata is not None:
                                match = df_metadata[df_metadata['file_name'] == audio_file.name]
                                if not match.empty:
                                    info = match.iloc[0]
                                    st.write("---")
                                    st.subheader("🔹 Info Pembicara")
                                    st.write(f"📅 Usia: {info.get('usia', '-')}")
                                    st.write(f"🗣️ Gender: {info.get('gender', '-')}")
                                    st.write(f"📍 Provinsi: {info.get('provinsi', '-')}")

                        os.unlink(tmp_path)
                else:
                    st.error("Sistem gagal dimuat. Periksa log server.")

if __name__ == "__main__":
    main()
