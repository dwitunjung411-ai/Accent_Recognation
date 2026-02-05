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
        # Hitung embedding untuk referensi dan input baru
        support_embeddings = self.embedding(support_set)
        query_embeddings = self.embedding(query_set)

        # Logika Prototype 
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

# ==========================================
# 2. FUNGSI EKSTRAKSI FITUR (Sesuai Skripsi)
# ==========================================
def extract_combined_features(file_path, usia, gender, provinsi, scaler_usia, ohe):
    # Ekstraksi MFCC [cite: 1]
    y, sr = librosa.load(file_path, sr=22050)
    y = librosa.util.normalize(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, n_fft=2048, hop_length=512)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    # Padding panjang 174 [cite: 1]
    max_len = 174
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
        delta = np.pad(delta, ((0, 0), (0, pad_width)), mode='constant')
        delta2 = np.pad(delta2, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc, delta, delta2 = mfcc[:, :max_len], delta[:, :max_len], delta2[:, :max_len]
    
    audio_feat = np.stack([mfcc, delta, delta2], axis=-1)

    # Proses Metadata [cite: 1]
    u_scaled = scaler_usia.transform([[usia]])
    c_enc = ohe.transform([[gender, provinsi]])
    meta = np.hstack([u_scaled, c_enc]).astype(np.float32)
    
    # Broadcast meta ke 11 channel [cite: 1]
    meta_b = np.repeat(meta[:, np.newaxis, np.newaxis, :], 40, axis=1)
    meta_b = np.repeat(meta_b, 174, axis=2)
    
    return np.concatenate([audio_feat[np.newaxis, ...], meta_b], axis=-1).astype(np.float32)

# ==========================================
# 3. LOAD RESOURCE
# ==========================================
@st.cache_resource
def load_all():
    model = tf.keras.models.load_model("model_aksen.keras", 
                                       custom_objects={"PrototypicalNetwork": PrototypicalNetwork})
    df = pd.read_csv("metadata.csv")
    return model, df

# ==========================================
# 4. TAMPILAN UI
# ==========================================
st.set_page_config(page_title="Deteksi Aksen Prototypical", layout="wide")

model, df_meta = load_all()

if model and df_meta:
    # Persiapkan Encoder & Scaler [cite: 1]
    le_y = LabelEncoder().fit(df_meta['label_aksen'])
    scaler_u = StandardScaler().fit(df_meta['usia'].values.reshape(-1, 1))
    ohe_m = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(df_meta[['gender', 'provinsi']])

    st.title("🎤 Sistem Deteksi Aksen Otomatis")
    
    with st.sidebar:
        st.header("Settings")
        st.radio("Select Mode:", ["Upload Audio"], index=0)

    uploaded_file = st.file_uploader("Upload file audio (.wav)", type=["wav"])

    if uploaded_file:
        st.audio(uploaded_file)
        
        # Auto-lookup Metadata berdasarkan nama file
        user_row = df_meta[df_meta['file_name'] == uploaded_file.name]

        if not user_row.empty:
            u_val, g_val, p_val = user_row.iloc[0]['usia'], user_row.iloc[0]['gender'], user_row.iloc[0]['provinsi']

            st.success(f"✅ Data ditemukan: {uploaded_file.name}")
            st.subheader("Informasi Pembicara:")
            col1, col2, col3 = st.columns(3)
            col1.write(f"📅 **Usia:** {u_val}")
            col2.write(f"👤 **Gender:** {g_val}")
            col3.write(f"📍 **Provinsi:** {p_val}")

            if st.button("🚀 Extract Features and Detect"):
                with st.spinner("Sedang memproses..."):
                    with open("query.wav", "wb") as f: f.write(uploaded_file.getbuffer())
                    
                    # 1. Ekstrak Query
                    query_final = extract_combined_features("query.wav", u_val, g_val, p_val, scaler_u, ohe_m)
                    
                    # 2. Siapkan Support Set (Referensial) 
                    support_set, support_labels = [], []
                    accents = df_meta['label_aksen'].unique()
                    
                    for idx, acc in enumerate(accents):
                        s_row = df_meta[df_meta['label_aksen'] == acc].iloc[0]
                        # Audio referensi harus ada di folder GitHub
                        if os.path.exists(s_row['file_name']):
                            s_feat = extract_combined_features(s_row['file_name'], s_row['usia'], 
                                                               s_row['gender'], s_row['provinsi'], 
                                                               scaler_u, ohe_m)
                            support_set.append(s_feat[0])
                            support_labels.append(idx)
                    
                    # 3. Jalankan Prediksi FSL 
                    if len(support_set) > 0:
                        logits = model.call(np.array(support_set), query_final, 
                                            np.array(support_labels), len(accents))
                        pred_idx = np.argmax(logits[0])
                        hasil = le_y.inverse_transform([pred_idx])[0]
                        st.markdown(f"### Hasil Deteksi Aksen: **{hasil}**")
                    else:
                        st.error("Gagal menyiapkan Support Set. Pastikan folder audio ada di GitHub!")
        else:
            st.error(f"File '{uploaded_file.name}' tidak ditemukan di metadata.csv!")
