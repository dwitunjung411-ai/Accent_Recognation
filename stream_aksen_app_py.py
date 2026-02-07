import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tempfile
import os
import tensorflow as tf

# ==========================================================
# 1. DEFINISI CLASS MODEL (PERBAIKAN INDENTASI & LOGIKA)
# ==========================================================
@tf.keras.utils.register_keras_serializable(package="Custom")
class PrototypicalNetwork(tf.keras.Model):
    def __init__(self, embedding_model=None, **kwargs):
        super(PrototypicalNetwork, self).__init__(**kwargs)
        if embedding_model is None:
            # Buat default embedding model jika tidak ada
            self.embedding = self._create_default_embedding()
        else:
            self.embedding = embedding_model

    def _create_default_embedding(self):
        """Membuat embedding model default jika tidak tersedia"""
        return tf.keras.Sequential([
            tf.keras.layers.Input(shape=(40, 174, 3)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.3)
        ])

    def call(self, support_set, query_set, support_labels, n_way, training=False):
        # Memastikan input dalam bentuk tensor
        support_set = tf.convert_to_tensor(support_set, dtype=tf.float32)
        query_set = tf.convert_to_tensor(query_set, dtype=tf.float32)
        support_labels = tf.convert_to_tensor(support_labels, dtype=tf.int32)

        # Mendapatkan embedding fitur
        support_embeddings = self.embedding(support_set, training=training)
        query_embeddings = self.embedding(query_set, training=training)

        # Kalkulasi Prototype per kelas aksen
        prototypes = []
        for i in range(n_way):
            mask = tf.equal(support_labels, i)
            class_embeddings = tf.boolean_mask(support_embeddings, mask)
            
            # Jika kelas kosong, beri nilai nol (cegah error pembagian)
            if tf.shape(class_embeddings)[0] == 0:
                prototype = tf.zeros_like(support_embeddings[0])
            else:
                prototype = tf.reduce_mean(class_embeddings, axis=0)
            prototypes.append(prototype)

        prototypes = tf.stack(prototypes)

        # Kalkulasi jarak Euclidean
        distances = tf.norm(
            tf.expand_dims(query_embeddings, 1) - tf.expand_dims(prototypes, 0),
            axis=2
        )
        return -distances

    def get_config(self):
        config = super().get_config()
        if self.embedding is not None:
            config.update({"embedding_model": tf.keras.layers.serialize(self.embedding)})
        return config

    @classmethod
    def from_config(cls, config):
        # Deserialize embedding model jika ada
        if "embedding_model" in config:
            embedding_model = tf.keras.layers.deserialize(config["embedding_model"])
            config["embedding_model"] = embedding_model
        return cls(**config)

# ==========================================================
# 2. FUNGSI EKSTRAKSI MFCC 3-CHANNEL (SESUAI SKRIPSI)
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
    
    # Cek jika model tidak ada, buat model baru
    if not os.path.exists(model_path):
        st.warning(f"Model file '{model_path}' tidak ditemukan. Membuat model baru...")
        return create_new_model()
    
    try:
        custom_objects = {
            "PrototypicalNetwork": PrototypicalNetwork,
            "Custom>PrototypicalNetwork": PrototypicalNetwork
        }
        
        # Coba load model
        model = tf.keras.models.load_model(
            model_path, 
            custom_objects=custom_objects,
            compile=False
        )
        
        # Cek jika embedding model ada
        if hasattr(model, 'embedding') and model.embedding is not None:
            st.success("Model berhasil di-load dengan embedding model")
        else:
            st.warning("Embedding model tidak ditemukan. Menggunakan default...")
            model.embedding = PrototypicalNetwork().embedding
        
        # Load support set yang sebenarnya (jika ada)
        support_data_path = "support_set.npy"
        support_labels_path = "support_labels.npy"
        
        if os.path.exists(support_data_path) and os.path.exists(support_labels_path):
            support_set = np.load(support_data_path)
            support_labels = np.load(support_labels_path)
            st.success("Support set berhasil di-load dari file")
        else:
            # Buat support set dummy berdasarkan aksen yang diketahui
            st.warning("File support set tidak ditemukan. Menggunakan data dummy...")
            support_set = create_dummy_support_set()
            support_labels = np.array([0, 1, 2, 3, 4], dtype=np.int32)
        
        return model, support_set, support_labels
        
    except Exception as e:
        st.error(f"Error Loading Model: {e}")
        st.info("Membuat model baru sebagai fallback...")
        return create_new_model()

def create_new_model():
    """Membuat model baru jika model lama tidak bisa di-load"""
    model = PrototypicalNetwork()
    
    # Buat support set dummy
    support_set = create_dummy_support_set()
    support_labels = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    
    return model, support_set, support_labels

def create_dummy_support_set():
    """Membuat support set dummy yang lebih representatif"""
    n_classes = 5
    n_samples_per_class = 3
    
    support_set = []
    for class_idx in range(n_classes):
        for sample_idx in range(n_samples_per_class):
            # Buat data dummy dengan sedikit variasi per kelas
            base_mfcc = np.random.randn(40, 174, 3).astype(np.float32)
            # Tambahkan bias berbeda per kelas untuk simulasi
            base_mfcc += class_idx * 0.1
            support_set.append(base_mfcc)
    
    return np.array(support_set)

# ==========================================================
# 4. ANTARMUKA UTAMA (MAIN APP) - DIPERBAIKI
# ==========================================================
def main():
    st.set_page_config(page_title="Analisis Aksen Prototypical", layout="centered")
    st.title("🎭 Sistem Deteksi Aksen Otomatis")
    st.markdown("---")

    # Load resources
    with st.spinner("Memuat model dan data..."):
        model, support_set, support_labels = load_resources()
    
    aksen_list = ["Sunda", "Jawa Tengah", "Jawa Timur", "Yogyakarta", "Betawi"]
    
    # Tampilkan info model
    with st.expander("ℹ️ Informasi Model"):
        st.write(f"Jumlah kelas aksen: {len(aksen_list)}")
        st.write(f"Ukuran support set: {support_set.shape}")
        st.write(f"Label support set: {support_labels}")
        if hasattr(model, 'embedding'):
            st.write("✅ Embedding model tersedia")
        else:
            st.write("⚠️ Embedding model tidak tersedia")

    audio_file = st.file_uploader("Upload Rekaman Suara (.wav)", type=["wav"])

    if audio_file:
        st.audio(audio_file)
        
        if st.button("🚀 Analisis Aksen Sekarang", type="primary"):
            with st.spinner("Mengekstrak fitur dan membandingkan..."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(audio_file.getbuffer())
                        
                        # Langkah 1: Ekstraksi fitur audio yang diunggah
                        query_feat = extract_mfcc_3channel(tmp.name)
                        
                        if query_feat is not None:
                            # Langkah 2: Siapkan input (batch dimension)
                            query_tensor = np.expand_dims(query_feat, axis=0) 
                            
                            # Langkah 3: Pastikan model siap
                            if model.embedding is None:
                                st.error("Embedding model tidak tersedia!")
                                return
                            
                            # Langkah 4: Jalankan Forward Pass Model dengan error handling
                            try:
                                logits = model.call(
                                    support_set=support_set,
                                    query_set=query_tensor,
                                    support_labels=support_labels,
                                    n_way=5,
                                    training=False
                                )
                                
                                # Langkah 5: Tampilkan Hasil
                                pred_idx = np.argmax(logits.numpy())
                                confidence = tf.nn.softmax(logits).numpy()[0][pred_idx]
                                
                                st.balloons()
                                st.success(f"### Aksen Terdeteksi: **{aksen_list[pred_idx]}**")
                                st.metric("Tingkat Kepercayaan", f"{confidence:.2%}")
                                
                                # Tampilkan semua probabilitas
                                st.subheader("Probabilitas Semua Aksen:")
                                probs = tf.nn.softmax(logits).numpy()[0]
                                for i, (aksen, prob) in enumerate(zip(aksen_list, probs)):
                                    st.progress(float(prob), text=f"{aksen}: {prob:.2%}")
                                    
                            except Exception as e:
                                st.error(f"Error saat prediksi: {str(e)}")
                                st.info("Coba upload file audio dengan format WAV 16-bit mono 22.05kHz")
                        
                        os.unlink(tmp.name)
                        
                except Exception as e:
                    st.error(f"Terjadi kesalahan: {str(e)}")
                    st.info("Pastikan file audio valid dan tidak rusak")

if __name__ == "__main__":
    main()
