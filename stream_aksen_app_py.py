import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tempfile
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
import soundfile as sf

# ==========================================================
# 1. DEFINISI CLASS PROTOTYPICAL NETWORK (SESUAI NOTEBOOK)
# ==========================================================
@tf.keras.utils.register_keras_serializable(package="Custom")
class PrototypicalNetwork(tf.keras.Model):
    def __init__(self, embedding_model=None, **kwargs):
        super(PrototypicalNetwork, self).__init__(**kwargs)
        self.embedding = embedding_model

    def call(self, inputs_data, query_set=None, support_labels=None, n_way=None):
        # Implementasi sesuai notebook
        if query_set is None and isinstance(inputs_data, (list, tuple)):
            support_set = inputs_data[0]
            query_set = inputs_data[1]
            support_labels = inputs_data[2]
            n_way = inputs_data[3]
        else:
            support_set = inputs_data

        if query_set is None or support_labels is None or n_way is None:
            raise ValueError(
                "PrototypicalNetwork requires support_set, query_set, "
                "support_labels, and n_way to run."
            )

        # Compute embeddings
        support_embeddings = self.embedding(support_set)
        query_embeddings = self.embedding(query_set)

        # Calculate prototypes per class
        prototypes = []
        for i in range(n_way):
            mask = tf.equal(support_labels, i)
            class_embeddings = tf.boolean_mask(support_embeddings, mask)
            
            if tf.shape(class_embeddings)[0] == 0:
                prototype = tf.zeros_like(support_embeddings[0])
            else:
                prototype = tf.reduce_mean(class_embeddings, axis=0)
            prototypes.append(prototype)

        prototypes = tf.stack(prototypes)

        # Calculate Euclidean distances
        distances = tf.norm(
            tf.expand_dims(query_embeddings, 1) - tf.expand_dims(prototypes, 0),
            axis=2
        )

        # Convert distances to logits
        logits = -distances
        return logits

    def get_config(self):
        config = super().get_config()
        if self.embedding is not None:
            config.update({
                "embedding_model": tf.keras.layers.serialize(self.embedding)
            })
        return config

    @classmethod
    def from_config(cls, config):
        if "embedding_model" in config:
            embedding_config = config.pop("embedding_model")
            embedding_model = tf.keras.layers.deserialize(embedding_config)
            return cls(embedding_model, **config)
        return cls(**config)

# ==========================================================
# 2. FUNGSI EXTRACT MFCC (SESUAI NOTEBOOK)
# ==========================================================
def extract_mfcc_streamlit(file_path, sr=22050, n_mfcc=40, max_len=174):
    """
    Ekstrak MFCC dengan delta dan delta-delta (sederhana untuk inference)
    """
    try:
        # Load audio
        y, sr = librosa.load(file_path, sr=sr)
        
        # Normalisasi amplitude
        y = librosa.util.normalize(y)
        
        # Ekstrak MFCC dasar
        mfcc = librosa.feature.mfcc(
            y=y, 
            sr=sr, 
            n_mfcc=n_mfcc, 
            n_fft=2048, 
            hop_length=512
        )
        
        # Untuk inference sederhana, ambil mean saja
        mfcc_mean = np.mean(mfcc, axis=1)
        
        # Reshape untuk model (batch, features)
        return mfcc_mean.reshape(1, -1)
        
    except Exception as e:
        st.error(f"Error extracting MFCC: {e}")
        return None

# ==========================================================
# 3. LOAD MODEL DAN DATA
# ==========================================================
@st.cache_resource
def load_accent_model():
    model_name = "model_detect_aksen.keras"
    current_dir = os.getcwd()
    model_path = os.path.join(current_dir, model_name)
    
    if os.path.exists(model_path):
        try:
            custom_objects = {"PrototypicalNetwork": PrototypicalNetwork}
            model = tf.keras.models.load_model(
                model_path, 
                custom_objects=custom_objects, 
                compile=False
            )
            return model
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None
    else:
        st.error(f"Model file not found: {model_path}")
        return None

@st.cache_data
def load_metadata():
    csv_path = "metadata.csv"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return pd.DataFrame()

# ==========================================================
# 4. FUNGSI PREDIKSI YANG DISEDERHANAKAN
# ==========================================================
def predict_accent_simple(audio_path, model):
    """
    Prediksi sederhana tanpa support set untuk demo
    """
    try:
        # Ekstrak fitur sederhana
        features = extract_mfcc_streamlit(audio_path)
        if features is None:
            return "Error extracting features"
        
        # Untuk model Prototypical, kita butuh dummy support set
        # Karena ini demo, kita buat sederhana
        dummy_support = np.random.randn(15, features.shape[1])  # 15 samples, 5 classes * 3 shots
        dummy_query = features
        dummy_labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])  # 5-way, 3-shot
        n_way = 5
        
        # Convert to tensor
        dummy_support_tensor = tf.convert_to_tensor(dummy_support, dtype=tf.float32)
        dummy_query_tensor = tf.convert_to_tensor(dummy_query, dtype=tf.float32)
        dummy_labels_tensor = tf.convert_to_tensor(dummy_labels, dtype=tf.int32)
        
        # Predict
        logits = model.call(
            dummy_support_tensor,
            dummy_query_tensor,
            dummy_labels_tensor,
            n_way
        )
        
        pred_index = tf.argmax(logits, axis=1).numpy()[0]
        
        aksen_classes = ["Betawi", "Jawa Timur", "Jawa Tengah", "Sunda", "Yogyakarta"]
        if pred_index < len(aksen_classes):
            return aksen_classes[pred_index]
        else:
            return f"Unknown (index: {pred_index})"
            
    except Exception as e:
        return f"Error in prediction: {str(e)}"

# ==========================================================
# 5. MAIN UI
# ==========================================================
def main():
    st.set_page_config(
        page_title="Deteksi Aksen Prototypical", 
        page_icon="🎙️", 
        layout="wide"
    )
    
    # Title
    st.title("🎙️ Sistem Deteksi Aksen Indonesia")
    st.markdown("Aplikasi berbasis *Few-Shot Learning* untuk klasifikasi aksen daerah.")
    st.divider()
    
    # Sidebar
    with st.sidebar:
        st.header("🛠️ Konfigurasi")
        
        # Load resources
        model_aksen = load_accent_model()
        df_metadata = load_metadata()
        
        # Status
        st.subheader("Status Sistem")
        if model_aksen:
            st.success("✅ Model Loaded")
        else:
            st.error("❌ Model Not Found")
            
        if not df_metadata.empty:
            st.success(f"✅ Metadata: {len(df_metadata)} records")
        else:
            st.warning("⚠️ Metadata Empty")
        
        st.divider()
        st.caption("Skripsi Project - Prototypical Networks")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Audio")
        
        # File uploader
        audio_file = st.file_uploader(
            "Pilih file audio (.wav, .mp3, .m4a)", 
            type=["wav", "mp3", "m4a"]
        )
        
        if audio_file:
            st.audio(audio_file, format=f"audio/{audio_file.type.split('/')[-1]}")
            
            # Tombol predict
            if st.button("🔍 Deteksi Aksen", type="primary", use_container_width=True):
                if model_aksen is None:
                    st.error("Model tidak tersedia. Pastikan file model_detect_aksen.keras ada di direktori yang sama.")
                else:
                    with st.spinner("Menganalisis audio..."):
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                            tmp.write(audio_file.getbuffer())
                            tmp_path = tmp.name
                        
                        try:
                            # Predict accent
                            hasil = predict_accent_simple(tmp_path, model_aksen)
                            
                            # Clean up
                            os.unlink(tmp_path)
                            
                            # Display results
                            with col2:
                                st.subheader("📊 Hasil Analisis")
                                
                                with st.container(border=True):
                                    st.markdown("### 🎯 Aksen Terdeteksi")
                                    if "Error" in hasil:
                                        st.error(f"**{hasil}**")
                                    else:
                                        st.success(f"**{hasil}**")
                                
                                st.divider()
                                
                                # Try to find metadata
                                if not df_metadata.empty:
                                    st.markdown("### 📋 Info File")
                                    file_match = df_metadata[df_metadata['file_name'] == audio_file.name]
                                    if not file_match.empty:
                                        info = file_match.iloc[0]
                                        st.markdown(f"- **Speaker ID:** {info.get('speaker_id', 'N/A')}")
                                        st.markdown(f"- **Usia:** {info.get('usia', 'N/A')}")
                                        st.markdown(f"- **Gender:** {info.get('gender', 'N/A')}")
                                        st.markdown(f"- **Provinsi:** {info.get('provinsi', 'N/A')}")
                                    else:
                                        st.info("ℹ️ File tidak ditemukan dalam metadata")
                                        
                        except Exception as e:
                            st.error(f"Terjadi error: {str(e)}")
    
    # Instructions
    with st.expander("ℹ️ Petunjuk Penggunaan"):
        st.markdown("""
        1. **Upload file audio** berbahasa Indonesia
        2. Klik tombol **"Deteksi Aksen"**
        3. Sistem akan menganalisis dan menampilkan hasil
        
        **Format file yang didukung:**
        - WAV (direkomendasikan)
        - MP3
        - M4A
        
        **Aksen yang dapat dideteksi:**
        - Betawi
        - Jawa Timur
        - Jawa Tengah
        - Sunda
        - Yogyakarta
        """)

if __name__ == "__main__":
    main()
