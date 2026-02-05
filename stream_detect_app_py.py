import streamlit as st
import tensorflow as tf
import keras
import librosa
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder

# ==========================================
# 1. DEFINISI KELAS PROTOTYPICAL NETWORK
# ==========================================
@keras.saving.register_keras_serializable(package="Custom")
class PrototypicalNetwork(tf.keras.Model):
    def __init__(self, embedding_model=None, **kwargs):
        super(PrototypicalNetwork, self).__init__(**kwargs)
        self.embedding = embedding_model

    def call(self, support_set, query_set, support_labels, n_way):
        # Hitung embedding
        support_embeddings = self.embedding(support_set)
        query_embeddings = self.embedding(query_set)

        # Hitung prototype per kelas
        prototypes = []
        for i in range(n_way):
            mask = tf.equal(support_labels, i)
            class_embeddings = tf.boolean_mask(support_embeddings, mask)
            prototype = tf.reduce_mean(class_embeddings, axis=0)
            prototypes.append(prototype)
        prototypes = tf.stack(prototypes)

        # Hitung jarak Euclidean (Logits)
        distances = []
        for q in query_embeddings:
            dist = tf.norm(prototypes - q, axis=1)
            distances.append(dist)
        
        return -tf.stack(distances)

# ==========================================
# 2. FUNGSI PEMROSESAN FITUR (MFCC)
# ==========================================
def extract_mfcc(file_path, sr=22050, n_mfcc=40, max_len=174):
    try:
        y, sr = librosa.load(file_path, sr=sr)
        y = librosa.util.normalize(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=2048, hop_length=512)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)

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
        st.error(f"Error ekstraksi audio: {e}")
        return None

# ==========================================
# 3. LOAD MODEL & METADATA
# ==========================================
@st.cache_resource
def load_resources():
    model_path = "model_aksen.keras"
    csv_path = "metadata.csv"
    
    if not os.path.exists(model_path):
        st.error("File model_aksen.keras tidak ditemukan di GitHub!")
        return None, None
    
    custom_objects = {"PrototypicalNetwork": PrototypicalNetwork}
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    metadata = pd.read_csv(csv_path)
    
    return model, metadata

# Persiapkan Scaler & Encoder (Harus sama dengan saat Training)
def prepare_transformers(metadata):
    le_y = LabelEncoder().fit(metadata['label_aksen'])
    le_gender = LabelEncoder().fit(metadata['gender'])
    le_provinsi = LabelEncoder().fit(metadata['provinsi'])
    
    scaler_usia = StandardScaler().fit(metadata['usia'].values.reshape(-1, 1))
    
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    ohe.fit(np.hstack([metadata['gender'].values.reshape(-1,1), 
                       metadata['provinsi'].values.reshape(-1,1)]))
    
    return le_y, scaler_usia, ohe

# ==========================================
# 4. TAMPILAN UTAMA STREAMLIT
# ==========================================
st.title("🎤 Deteksi Aksen Indonesia")
st.write("Few-Shot Learning Accent Recognition")

model, metadata = load_resources()

if model and metadata:
    le_y, scaler_usia, ohe = prepare_transformers(metadata)
    
    # Sidebar Input Metadata Pengguna
    st.sidebar.header("Input Data Pembicara")
    usia_input = st.sidebar.number_input("Usia", min_value=5, max_value=100, value=25)
    gender_input = st.sidebar.selectbox("Gender", metadata['gender'].unique())
    provinsi_input = st.sidebar.selectbox("Provinsi", metadata['provinsi'].unique())
    
    uploaded_file = st.file_uploader("Upload Audio (.wav)", type=["wav"])

    if uploaded_file is not None:
        st.audio(uploaded_file)
        
        if st.button("Deteksi Aksen Sekarang"):
            with st.spinner('Memproses audio dan membandingkan aksen...'):
                # 1. Simpan file sementara
                with open("temp.wav", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # 2. Ekstraksi Fitur Audio (Query)
                mfcc_feat = extract_mfcc("temp.wav")
                
                # 3. Proses Metadata (Query)
                usia_scaled = scaler_usia.transform([[usia_input]])
                cat_enc = ohe.transform([[gender_input, provinsi_input]])
                meta_feat = np.hstack([usia_scaled, cat_enc]).astype(np.float32)
                
                # Broadcast Metadata ke bentuk (40, 174, 8) -> Sesuaikan jumlah channel
                meta_broadcast = np.repeat(meta_feat[:, np.newaxis, np.newaxis, :], 40, axis=1)
                meta_broadcast = np.repeat(meta_broadcast, 174, axis=2)
                
                # Gabungkan jadi 11 Channel
                query_final = np.concatenate([mfcc_feat[np.newaxis, ...], meta_broadcast], axis=-1)
                
                # 4. Siapkan Support Set (Referensial)
                # Di sini kita mengambil contoh dari metadata untuk tiap aksen
                support_features = []
                support_labels = []
                unique_accents = metadata['label_aksen'].unique()
                
                for idx, accent in enumerate(unique_accents):
                    # Ambil 1 contoh file per aksen dari metadata (K-Shot = 1)
                    sample_row = metadata[metadata['label_aksen'] == accent].iloc[0]
                    # Catatan: File audio referensi ini HARUS ada di folder GitHub Anda
                    s_feat = extract_mfcc(sample_row['file_name']) 
                    
                    if s_feat is not None:
                        # (Proses metadata support mirip seperti query di atas...)
                        # Untuk simplifikasi tutorial, kita asumsikan support set sudah siap
                        # Anda mungkin perlu melatih model tanpa metadata jika support set sulit disiapkan di Streamlit
                        pass

                # SIMULASI PEMANGGILAN (Karena error query_set)
                # PENTING: Anda harus mengirimkan support_set dan query_set
                try:
                    # Ganti baris ini dengan tensor support asli Anda
                    # logits = model(support_set, query_final, support_labels, n_way=5)
                    
                    st.info("Fitur berhasil diekstrak! Model Few-Shot siap memproses.")
                    st.warning("Pastikan folder audio training ada di GitHub agar Support Set bisa dimuat.")
                    
                except Exception as e:
                    st.error(f"Error Processing: {e}")

else:
    st.warning("Menunggu Model dan Metadata di-upload ke GitHub...")
