import streamlit as st
import tensorflow as tf
import keras
import librosa
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder

# 1. DEFINISI KELAS PROTOTYPICAL NETWORK (Wajib Sama dengan Colab)
@keras.saving.register_keras_serializable(package="Custom")
class PrototypicalNetwork(tf.keras.Model):
    def __init__(self, embedding_model=None, **kwargs):
        super(PrototypicalNetwork, self).__init__(**kwargs)
        self.embedding = embedding_model

    def call(self, support_set, query_set, support_labels, n_way):
        support_embeddings = self.embedding(support_set)
        query_embeddings = self.embedding(query_set)

        prototypes = []
        for i in range(n_way):
            mask = tf.equal(support_labels, i)
            class_embeddings = tf.boolean_mask(support_embeddings, mask)
            prototype = tf.reduce_mean(class_embeddings, axis=0)
            prototypes.append(prototype)
        prototypes = tf.stack(prototypes)

        distances = []
        for q in query_embeddings:
            dist = tf.norm(prototypes - q, axis=1)
            distances.append(dist)
        return -tf.stack(distances)

# 2. FUNGSI EKSTRAKSI FITUR MFCC (11 Channel)
def extract_mfcc_combined(file_path, usia, gender, provinsi, scaler_usia, ohe, sr=22050, n_mfcc=40, max_len=174):
    y, sr = librosa.load(file_path, sr=sr)
    y = librosa.util.normalize(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=2048, hop_length=512)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    # Padding audio
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
        delta = np.pad(delta, ((0, 0), (0, pad_width)), mode='constant')
        delta2 = np.pad(delta2, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc, delta, delta2 = mfcc[:, :max_len], delta[:, :max_len], delta2[:, :max_len]
    
    audio_feat = np.stack([mfcc, delta, delta2], axis=-1)

    # Proses Metadata
    usia_scaled = scaler_usia.transform([[usia]])
    cat_enc = ohe.transform([[gender, provinsi]])
    meta_feat = np.hstack([usia_scaled, cat_enc]).astype(np.float32)
    
    # Broadcast Metadata ke 11 Channel
    meta_broadcast = np.repeat(meta_feat[:, np.newaxis, np.newaxis, :], 40, axis=1)
    meta_broadcast = np.repeat(meta_broadcast, 174, axis=2)
    
    return np.concatenate([audio_feat[np.newaxis, ...], meta_broadcast], axis=-1).astype(np.float32)

# 3. LOAD SUMBER DAYA
@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model("model_aksen.keras", custom_objects={"PrototypicalNetwork": PrototypicalNetwork})
    df = pd.read_csv("metadata.csv")
    return model, df

# 4. ANTARMUKA UTAMA
st.title("🎤 Deteksi Aksen Otomatis (Skripsi)")
model, df_meta = load_resources()

if model is not None:
    # Persiapkan Encoder & Scaler dari Metadata
    le_y = LabelEncoder().fit(df_meta['label_aksen'])
    scaler_usia = StandardScaler().fit(df_meta['usia'].values.reshape(-1, 1))
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(df_meta[['gender', 'provinsi']])

    uploaded_file = st.file_uploader("Upload Audio", type=["wav"])

    if uploaded_file:
        filename = uploaded_file.name
        st.audio(uploaded_file)
        
        # Cari data otomatis di CSV
        user_data = df_meta[df_meta['file_name'] == filename]

        if not user_data.empty:
            u, g, p = user_data.iloc[0]['usia'], user_data.iloc[0]['gender'], user_data.iloc[0]['provinsi']
            st.success(f"✅ Data Terdeteksi: {g} | {u} Tahun | {p}")
            
            if st.button("🚀 Deteksi Sekarang"):
                # Simpan audio sementara
                with open("temp.wav", "wb") as f: f.write(uploaded_file.getbuffer())
                
                # Ekstrak Query
                query_final = extract_mfcc_combined("temp.wav", u, g, p, scaler_usia, ohe)
                
                # SIAPKAN SUPPORT SET (REFERENSI)
                # Di sini kita ambil 1 contoh acak dari tiap aksen di CSV untuk jadi patokan model
                support_set, support_labels = [], []
                unique_accents = df_meta['label_aksen'].unique()
                
                for idx, accent in enumerate(unique_accents):
                    sample = df_meta[df_meta['label_aksen'] == accent].iloc[0]
                    # Penting: File audio referensi ini HARUS ada di folder GitHub yang sama
                    if os.path.exists(sample['file_name']):
                        s_feat = extract_mfcc_combined(sample['file_name'], sample['usia'], sample['gender'], sample['provinsi'], scaler_usia, ohe)
                        support_set.append(s_feat[0])
                        support_labels.append(idx)
                
                if len(support_set) > 0:
                    logits = model.call(np.array(support_set), query_final, np.array(support_labels), len(unique_accents))
                    pred_idx = np.argmax(logits[0])
                    st.metric("Hasil Deteksi Aksen:", le_y.inverse_transform([pred_idx])[0])
                else:
                    st.error("Folder audio referensi tidak ditemukan di GitHub!")
