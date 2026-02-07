import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tempfile
import os
import tensorflow as tf

# ==========================================================
# 1. DEFINISI CLASS PROTOTYPICAL NETWORK (WAJIB SAMA DENGAN TRAINING)
# ==========================================================
@tf.keras.utils.register_keras_serializable(package="Custom")
class PrototypicalNetwork(tf.keras.Model):
    def __init__(self, embedding_model=None, **kwargs):
        super(PrototypicalNetwork, self).__init__(**kwargs)
        self.embedding = embedding_model

    def call(self, support_set, query_set, support_labels, n_way, training=False):
        # Forward pass untuk mendapatkan embedding
        support_embeddings = self.embedding(support_set)
        query_embeddings = self.embedding(query_set)

        # Hitung Prototypes per kelas
        prototypes = []
        for i in range(n_way):
            mask = tf.equal(support_labels, i)
            class_embeddings = tf.boolean_mask(support_embeddings, mask)
            prototype = tf.reduce_mean(class_embeddings, axis=0)
            prototypes.append(prototype)

        prototypes = tf.stack(prototypes)

        # Hitung Jarak Euclidean (Logits adalah negatif jarak)
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
# 2. FUNGSI EKSTRAKSI FITUR (SAMA DENGAN COLAB)
# ==========================================================
def extract_mfcc_3channel(file_path, sr=22050, n_mfcc=40, max_len=174):
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
            mfcc, delta, delta2 = mfcc[:, :max_len], delta[:, :max_len], delta2[:, :max_len]

        return np.stack([mfcc, delta, delta2], axis=-1)
    except Exception as e:
        st.error(f"Gagal ekstrak fitur: {e}")
        return None

# ==========================================================
# 3. HELPER LOAD MODEL & DATA
# ==========================================================
@st.cache_resource
def load_all_assets():
    model_path = "model_aksen.keras" # Pastikan file ini ada di folder yang sama
    custom_objects = {"PrototypicalNetwork": PrototypicalNetwork}
    
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        # Dummy Support Set (Idealnya kamu muat dari file .npy hasil training)
        # Di sini kita asumsikan n_way=5, k_shot=1, feature_shape=(40, 174, 3)
        # UNTUK AKURASI TERBAIK: Muat support_set asli dari X_train kamu
        support_set = np.random.randn(5, 40, 174, 3).astype(np.float32) 
        support_labels = np.array([0, 1, 2, 3, 4], dtype=np.int32)
        return model, support_set, support_labels
    return None, None, None

# ==========================================================
# 4. MAIN INTERFACE
# ==========================================================
def main():
    st.set_page_config(page_title="Deteksi Aksen Prototypical", layout="wide")
    st.title("🎭 Sistem Deteksi Aksen (Few-Shot Learning)")
    
    model, support_set, support_labels = load_all_assets()
    aksen_classes = ["Sunda", "Jawa Tengah", "Jawa Timur", "Yogyakarta", "Betawi"]

    if model is None:
        st.error("Model 'model_aksen.keras' tidak ditemukan!")
        return

    audio_file = st.file_uploader("Unggah Audio (.wav)", type=["wav"])

    if audio_file:
        st.audio(audio_file)
        
        if st.button("Analisis Aksen", type="primary"):
            with st.spinner("Menganalisis..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(audio_file.getbuffer())
                    
                    # 1. Ekstrak Fitur Query
                    query_feat = extract_mfcc_3channel(tmp.name)
                    if query_feat is not None:
                        # Tambah batch dimension
                        query_tensor = np.expand_dims(query_feat, axis=0) 
                        
                        # 2. Prediksi via Call (Few-Shot Inference)
                        logits = model.call(
                            support_set=tf.convert_to_tensor(support_set),
                            query_set=tf.convert_to_tensor(query_tensor),
                            support_labels=tf.convert_to_tensor(support_labels),
                            n_way=5
                        )
                        
                        pred_idx = np.argmax(logits.numpy())
                        hasil = aksen_classes[pred_idx]
                        
                        st.success(f"### Aksen Terdeteksi: {hasil}")
                    
                    os.unlink(tmp.name)

if __name__ == "__main__":
    main()
