import streamlit as st
import tensorflow as tf
import keras
import librosa
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder

# ==========================================
# 1. DEFINISI KELAS KUSTOM (Wajib Sama dengan Colab)
# ==========================================
@keras.saving.register_keras_serializable(package="Custom")
class PrototypicalNetwork(tf.keras.Model):
    def __init__(self, embedding_model=None, **kwargs):
        super(PrototypicalNetwork, self).__init__(**kwargs)
        self.embedding = embedding_model

    def call(self, support_set, query_set, support_labels, n_way):
        # Proses embedding untuk support dan query
        support_embeddings = self.embedding(support_set)
        query_embeddings = self.embedding(query_set)

        # Hitung prototype (rata-rata embedding per kelas)
        prototypes = []
        for i in range(n_way):
            mask = tf.equal(support_labels, i)
            class_embeddings = tf.boolean_mask(support_embeddings, mask)
            prototype = tf.reduce_mean(class_embeddings, axis=0)
            prototypes.append(prototype)
        prototypes = tf.stack(prototypes)

        # Hitung jarak (Logits)
        distances = []
        for q in query_embeddings:
            dist = tf.norm(prototypes - q, axis=1)
            distances.append(dist)
        return -tf.stack(distances)

# ==========================================
# 2. FUNGSI EKSTRAKSI FITUR (Sesuai Skripsi Anda)
# ==========================================
def extract_mfcc(file_path, sr=22050, n_mfcc=40, max_len=174):
    try:
        y, sr = librosa.load(file_path, sr=sr)
        y = librosa.util.normalize(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=2048, hop_length=512)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)

        # Padding/Truncating
        if mfcc.shape[1] < max_len:
            pad_width = max_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
            delta = np.pad(delta, ((0, 0), (0, pad_width)), mode='constant')
            delta2 = np.pad(delta2, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_len]
            delta = delta[:, :max_len]
            delta2 = delta2[:, :max_len]

        return np.stack([mfcc, delta, delta2], axis=-1)
    except Exception as e:
        return None

# ==========================================
# 3. LOAD MODEL & DATA
# ==========================================
@st.cache_resource
def load_all_resources():
    # Pastikan file-file ini ada di folder GitHub yang sama
    model_path = "model_aksen.keras"
    csv_path = "metadata.csv"
    
    if not os.path.exists(model_path) or not os.path.exists(csv_path):
        return None, None, "File Model atau Metadata.csv tidak ditemukan di GitHub!"

    try:
        custom_objects = {"PrototypicalNetwork": PrototypicalNetwork}
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        metadata = pd.read_csv(csv_path)
        return model, metadata, "Berhasil"
    except Exception as e:
        return None, None, f"Gagal memuat model: {str(e)}"

# ==========================================
# 4. ANTARMUKA STREAMLIT
# ==========================================
st.set_page_config(page_title="Deteksi Aksen Skripsi", layout="wide")
st.title("🎤 Sistem Deteksi Aksen (Few-Shot Learning)")

model_aksen, df_meta, msg = load_all_resources()

if model_aksen is None:
    st.error(msg)
    st.info("Pastikan file 'model_aksen.keras' dan 'metadata.csv' sudah di-upload ke GitHub.")
else:
    # Inisialisasi Encoder (Wajib sesuai training)
    le_y = LabelEncoder().fit(df_meta['label_aksen'])
    scaler_usia = StandardScaler().fit(df_meta['usia'].values.reshape(-1, 1))
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    ohe.fit(df_meta[['gender', 'provinsi']])

    st.success("Model dan Metadata siap!")

    # Layout Input
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Data Pembicara")
        usia = st.number_input("Masukkan Usia", 5, 100, 20)
        gender = st.selectbox("Pilih Gender", df_meta['gender'].unique())
        provinsi = st.selectbox("Pilih Provinsi Asal", df_meta['provinsi'].unique())
        uploaded_audio = st.file_uploader("Upload Suara Anda (.wav)", type=["wav"])

    if uploaded_audio:
        st.audio(uploaded_audio)
        
        if st.button("🚀 Deteksi Aksen"):
            try:
                # 1. Simpan dan Ekstrak Fitur Query (Audio yang diupload)
                with open("query_temp.wav", "wb") as f:
                    f.write(uploaded_audio.getbuffer())
                
                query_mfcc = extract_mfcc("query_temp.wav")
                
                # 2. Siapkan Metadata Query
                meta_query = np.hstack([scaler_usia.transform([[usia]]), ohe.transform([[gender, provinsi]])])
                meta_broadcast = np.repeat(meta_query[:, np.newaxis, np.newaxis, :], 40, axis=1)
                meta_broadcast = np.repeat(meta_broadcast, 174, axis=2)
                
                # Gabungkan Fitur Audio + Metadata (11 Channel)
                query_final = np.concatenate([query_mfcc[np.newaxis, ...], meta_broadcast], axis=-1).astype(np.float32)

                # 3. SIAPKAN SUPPORT SET (REFERENSI)
                # Model FSL butuh contoh audio nyata untuk dibandingkan
                # Di sini kita ambil secara acak dari metadata (Support Set)
                n_way = 5
                k_shot = 3 # Ambil 3 contoh per kelas
                
                support_features = []
                support_labels = []
                
                selected_classes = le_y.classes_[:n_way] # Ambil 5 kelas pertama
                
                # CATATAN: Folder audio asli skripsi harus ada di GitHub agar ini jalan!
                # Jika folder audio tidak ada, bagian ini akan error.
                st.info("Sedang menyiapkan data referensi (Support Set)...")
                
                # (Proses Support Set di sini harus mirip dengan logika create_episode di Colab)
                # Untuk demo ini, kita gunakan placeholder jika file audio tidak tersedia di server
                
                st.warning("Error 'query_set' hilang karena model Prototypical butuh data referensi tambahan.")
                st.write("Hubungkan script ini dengan folder Voice_Skripsi_fix di GitHub Anda.")

            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")
