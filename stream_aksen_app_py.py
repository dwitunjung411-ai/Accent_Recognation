import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tempfile
import os
import tensorflow as tf

# ==========================================================
# 1. DEFINISI CLASS MODEL (STABLE VERSION)
# ==========================================================
@tf.keras.utils.register_keras_serializable(package="Custom")
class PrototypicalNetwork(tf.keras.Model):
    def __init__(self, embedding_model=None, **kwargs):
        super(PrototypicalNetwork, self).__init__(**kwargs)
        self.embedding = embedding_model

    # Perbaikan: Tambahkan argumen query_set secara eksplisit di sini
    def call(self, support_set, query_set, support_labels, n_way, training=False):
        # Pastikan input adalah tensor agar tidak error saat masuk ke self.embedding
        support_set = tf.convert_to_tensor(support_set, dtype=tf.float32)
        query_set = tf.convert_to_tensor(query_set, dtype=tf.float32)

        # Mendapatkan embedding fitur
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

        # Jarak Euclidean
        distances = tf.norm(
            tf.expand_dims(query_embeddings, 1) - tf.expand_dims(prototypes, 0),
            axis=2
        )
        return -distances

    def get_config(self):
        config = super().get_config()
        config.update({"embedding_model": tf.keras.layers.serialize(self.embedding)})
        return config

# ==========================================================
# 2. FUNGSI EKSTRAKSI MFCC 3-CHANNEL
# ==========================================================
def extract_mfcc_3channel(file_path, sr=22050, n_mfcc=40, max_len=174):
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
            mfcc, delta, delta2 = mfcc[:, :max_len], delta[:, :max_len], delta2[:, :max_len]

        return np.stack([mfcc, delta, delta2], axis=-1)
    except Exception as e:
        st.error(f"Gagal memproses audio: {e}")
        return None

# ==========================================================
# 3. LOADING MODEL & DATA REFERENSI
# ==========================================================
@st.cache_resource
def load_resources():
    model_path = "model_detect_aksen.keras" 
    if not os.path.exists(model_path):
        return None, None, None
    
    try:
        custom_objects = {"PrototypicalNetwork": PrototypicalNetwork}
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
        
        # Mengatasi masalah TrackedDict pada embedding
        if isinstance(model.embedding, dict):
            model.embedding = tf.keras.layers.deserialize(model.embedding)

        # Support set dummy (Ganti dengan file .npy asli untuk akurasi nyata)
        support_set = np.random.randn(5, 40, 174, 3).astype(np.float32) 
        support_labels = np.array([0, 1, 2, 3, 4], dtype=np.int32)
        
        return model, support_set, support_labels
    except Exception as e:
        st.error(f"Error Loading Model: {e}")
        return None, None, None

# ==========================================================
# 4. ANTARMUKA UTAMA
# ==========================================================
def main():
    st.set_page_config(page_title="Analisis Aksen Prototypical", layout="centered")
    st.title("🚀 Sistem Deteksi Aksen Otomatis")
    st.markdown("---")

    model, support_set, support_labels = load_resources()
    aksen_list = ["Sunda", "Jawa Tengah", "Jawa Timur", "Yogyakarta", "Betawi"]

    if model is None:
        st.error("Model tidak ditemukan! Pastikan 'model_detect_aksen.keras' ada di folder aplikasi.")
        return

    audio_file = st.file_uploader("Upload Rekaman Suara (.wav)", type=["wav"])

    if audio_file:
        st.audio(audio_file)
        
       # Cari bagian ini di fungsi main() kamu:
if st.button("🚀 Analisis Aksen Sekarang", type="primary"):
    # ... (proses ekstraksi mfcc) ...
    
    if query_feat is not None:
        # Masukkan fitur audio ke dalam query_tensor
        query_tensor = np.expand_dims(query_feat, axis=0).astype(np.float32) 
        
        # SOLUSI UTAMA: Panggil .call secara manual dengan nama argumen yang jelas
        logits = model.call(
            support_set=support_set,    # Data referensi
            query_set=query_tensor,     # DATA AUDIO YANG DIUNGGAH (Query Set)
            support_labels=support_labels, 
            n_way=5
        )
        
        # Ambil hasil prediksi
        pred_idx = np.argmax(logits.numpy()[0])
        st.success(f"### Aksen Terdeteksi: **{aksen_list[pred_idx]}**")
                        except Exception as e:
                            st.error(f"Kesalahan Analisis: {e}")
                    
                    os.unlink(tmp.name)

if __name__ == "__main__":
    main()
