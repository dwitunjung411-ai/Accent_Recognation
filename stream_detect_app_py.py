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
        # Hitung embedding untuk referensi (support) dan input baru (query)
        support_embeddings = self.embedding(support_set)
        query_embeddings = self.embedding(query_set)

        # Hitung rata-rata tiap kelas (prototype)
        prototypes = []
        for i in range(n_way):
            mask = tf.equal(support_labels, i)
            class_embeddings = tf.boolean_mask(support_embeddings, mask)
            prototype = tf.reduce_mean(class_embeddings, axis=0)
            prototypes.append(prototype)
        prototypes = tf.stack(prototypes)

        # Hitung jarak Euclidean
        distances = []
        for q in query_embeddings:
            dist = tf.norm(prototypes - q, axis=1)
            distances.append(dist)
        
        return -tf.stack(distances) # Mengembalikan logits 

# ==========================================
# 2. FUNGSI EKSTRAKSI FITUR (Sesuai Colab)
# ==========================================
def extract_mfcc(file_path, sr=22050, n_mfcc=40, max_len=174):
    try:
        y, sr = librosa.load(file_path, sr=sr)
        y = librosa.util.normalize(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=2048, hop_length=512)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)

        # Padding 
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
# 3. LOAD SUMBER DAYA
# ==========================================
@st.cache_resource
def load_all():
    model_path = "model_aksen.keras"
    csv_path = "metadata.csv"
    
    if not os.path.exists(model_path) or not os.path.exists(csv_path):
        return None, None, "File 'model_aksen.keras' atau 'metadata.csv' tidak ada di GitHub!"

    try:
        custom_objects = {"PrototypicalNetwork": PrototypicalNetwork}
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        df = pd.read_csv(csv_path)
        return model, df, "Success"
    except Exception as e:
        return None, None, f"Error: {str(e)}"

# ==========================================
# 4. HALAMAN UTAMA
# ==========================================
st.set_page_config(page_title="Deteksi Aksen Otomatis", layout="centered")
st.title("🎤 Deteksi Aksen Otomatis")
st.write("Aplikasi akan mendeteksi metadata secara otomatis berdasarkan file audio.")

model, df_meta, status = load_all()

if model is not None:
    # Siapkan encoder & scaler 
    le_y = LabelEncoder().fit(df_meta['label_aksen'])
    scaler_usia = StandardScaler().fit(df_meta['usia'].values.reshape(-1, 1))
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    ohe.fit(df_meta[['gender', 'provinsi']])

    uploaded_file = st.file_uploader("Upload File Audio Skripsi", type=["wav"])

    if uploaded_file is not None:
        filename = uploaded_file.name
        st.audio(uploaded_file)
        
        # --- LOGIKA AUTO-LOOKUP METADATA ---
        # Mencari baris di CSV yang kolom 'file_name' nya sama dengan nama file yang diupload
        user_data = df_meta[df_meta['file_name'] == filename]

        if not user_data.empty:
            usia_val = user_data.iloc[0]['usia']
            gender_val = user_data.iloc[0]['gender']
            prov_val = user_data.iloc[0]['provinsi']
            label_asli = user_data.iloc[0]['label_aksen']

            st.success(f"✅ Data ditemukan untuk file: **{filename}**")
            col1, col2, col3 = st.columns(3)
            col1.metric("Usia", usia_val)
            col2.metric("Gender", gender_val)
            col3.metric("Provinsi", prov_val)
            
            if st.button("🚀 Jalankan Deteksi Aksen"):
                with st.spinner("Mengekstrak fitur dan memproses..."):
                    # 1. Ekstrak Fitur MFCC Query
                    with open("temp_audio.wav", "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    mfcc_query = extract_mfcc("temp_audio.wav")
                    
                    # 2. Proses Metadata Query 
                    usia_scaled = scaler_usia.transform([[usia_val]])
                    cat_enc = ohe.transform([[gender_val, prov_val]])
                    meta_combined = np.hstack([usia_scaled, cat_enc]).astype(np.float32)
                    
                    # Broadcast Metadata ke channel tambahan (untuk jadi 11 channel)
                    meta_broadcast = np.repeat(meta_combined[:, np.newaxis, np.newaxis, :], 40, axis=1)
                    meta_broadcast = np.repeat(meta_broadcast, 174, axis=2)
                    query_final = np.concatenate([mfcc_query[np.newaxis, ...], meta_broadcast], axis=-1)

                    # --- PERSIAPAN SUPPORT SET (REFERENSI) ---
                    # Karena model Anda Prototypical, kita butuh Support Set sebagai referensi
                    # Di sini kita perlu menyiapkan support_set, query_set, support_labels, n_way
                    st.info("Fitur berhasil diekstrak. Menghubungkan ke Support Set...")
                    
                    # Catatan: Di sini Anda perlu memuat tensor Support Set yang 
                    # sudah diproses sebelumnya agar tidak error 'query_set missing'
                    st.warning("Pastikan Anda memiliki fungsi untuk memuat Support Set dari folder suara di GitHub.")

        else:
            st.error(f"❌ File '{filename}' tidak ditemukan di metadata.csv!")
            st.info("Pastikan nama file yang diupload sama persis dengan yang ada di kolom 'file_name' pada CSV.")

else:
    st.error(status)
