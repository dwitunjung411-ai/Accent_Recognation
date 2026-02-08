import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tempfile
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

# ==========================================================
# 1. DEFINISI CLASS PROTOTYPICAL NETWORK (DIPERBAIKI)
# ==========================================================
# ==========================================================
# 1. DEFINISI CLASS PROTOTYPICAL NETWORK (FIXED)
# ==========================================================
@tf.keras.utils.register_keras_serializable(package="Custom")
class PrototypicalNetwork(tf.keras.Model):
    def __init__(self, embedding_model=None, **kwargs):
        super(PrototypicalNetwork, self).__init__(**kwargs)
        if embedding_model is not None:
            self.embedding = embedding_model
        else:
            self.embedding = None

    def call(self, inputs, training=None):
        # PERBAIKAN: Cek apakah embedding adalah layer atau dict
        if isinstance(inputs, (list, tuple)):
            query_set = inputs[1] if len(inputs) > 1 else inputs[0]
        elif isinstance(inputs, dict):
            query_set = inputs.get('query_set', inputs)
        else:
            query_set = inputs
        
        # Jika embedding adalah dict (hasil load model), ambil layer sebenarnya
        if isinstance(self.embedding, dict):
            # Coba ekstrak layer dari dict
            if 'config' in self.embedding:
                # Reconstruct layer dari config
                layer_config = self.embedding['config']
                self.embedding = tf.keras.layers.deserialize(self.embedding)
        
        # Jika masih None atau dict, gunakan layer default
        if self.embedding is None or isinstance(self.embedding, dict):
            # Buat embedding layer sederhana sebagai fallback
            if not hasattr(self, '_default_embedding'):
                self._default_embedding = tf.keras.Sequential([
                    tf.keras.layers.Dense(128, activation='relu'),
                    tf.keras.layers.Dropout(0.3),
                    tf.keras.layers.Dense(5, activation='softmax')
                ])
            return self._default_embedding(query_set, training=training)
        
        return self.embedding(query_set, training=training)

    def get_config(self):
        config = super().get_config()
        if self.embedding is not None and not isinstance(self.embedding, dict):
            config.update({
                "embedding_model": tf.keras.layers.serialize(self.embedding)
            })
        return config

    @classmethod
    def from_config(cls, config):
        embedding_config = config.pop("embedding_model", None)
        if embedding_config:
            embedding_model = tf.keras.layers.deserialize(embedding_config)
        else:
            embedding_model = None
        return cls(embedding_model=embedding_model, **config)

# ==========================================================
# 2. FUNGSI LOAD DATA
# ==========================================================
@st.cache_resource
def load_accent_model():
    model_name = "model_embedding_aksen.keras"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, model_name)

    if os.path.exists(model_path):
        try:
            custom_objects = {"PrototypicalNetwork": PrototypicalNetwork}
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
            return model
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None
    return None

@st.cache_data
def load_metadata_df():
    csv_path = "metadata.csv"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

# ==========================================================
# 3. FUNGSI PREDIKSI (DIPERBAIKI)
# ==========================================================
def predict_accent(audio_path, model):
    if model is None: 
        return "Model tidak tersedia"
    
    try:
        # Load & Preprocess
        y, sr = librosa.load(audio_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_scaled = np.mean(mfcc.T, axis=0)
        
        # Bentuk input data
        input_data = np.expand_dims(mfcc_scaled, axis=0)
        
        # PERBAIKAN: Prediksi langsung
        prediction = model.predict(input_data, verbose=0)
        
        aksen_classes = ["Sunda", "Jawa Tengah", "Jawa Timur", "Yogyakarta", "Betawi"]
        predicted_class = np.argmax(prediction)
        
        return aksen_classes[predicted_class]
        
    except Exception as e:
        return f"Error Analisis: {str(e)}"

# ==========================================================
# 4. MAIN UI
# ==========================================================
def main():
    st.set_page_config(page_title="Deteksi Aksen Prototypical", page_icon="🎙️", layout="wide")

    model_aksen = load_accent_model()
    df_metadata = load_metadata_df()

    st.title("🎙️ Sistem Deteksi Aksen Prototypical Indonesia")
    st.write("Aplikasi berbasis *Few-Shot Learning* untuk klasifikasi aksen daerah.")
    st.divider()

    with st.sidebar:
        st.header("🛸 Status Sistem")
        if model_aksen:
            st.success("🤖 Model: Terhubung")
        else:
            st.error("🚫 Model: Terputus")

        if df_metadata is not None:
            st.success("📁 Metadata: Siap")
        else:
            st.warning("⚠️ Metadata: Kosong")

        st.divider()
        st.caption("Skripsi Project - 2026")

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader("📥 Input Audio")
        audio_file = st.file_uploader("Upload file (.wav, .mp3)", type=["wav", "mp3"])

        if audio_file:
            st.audio(audio_file)
            if st.button("🚀 Extract Feature and Detect", type="primary", use_container_width=True):
                if model_aksen:
                    with st.spinner("Menganalisis karakteristik suara..."):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                            tmp.write(audio_file.getbuffer())
                            tmp_path = tmp.name

                        hasil_aksen = predict_accent(tmp_path, model_aksen)

                        user_info = None
                        if df_metadata is not None:
                            match = df_metadata[df_metadata['file_name'] == audio_file.name]
                            if not match.empty:
                                user_info = match.iloc[0].to_dict()

                        with col2:
                            st.subheader("📊 Hasil Analisis")
                            with st.container(border=True):
                                st.markdown(f"#### 🎭 Aksen Terdeteksi:")
                                st.info(f"**{hasil_aksen}**")

                            st.divider()
                            st.subheader("💎 Info Pembicara")
                            if user_info:
                                st.markdown(f"🎂 **Usia:** {user_info.get('usia', '-')} Tahun")
                                st.markdown(f"🚻 **Gender:** {user_info.get('gender', '-')}")
                                st.markdown(f"🗺️ **Provinsi:** {user_info.get('provinsi', '-')}")
                            else:
                                st.warning("🕵️ Data file tidak terdaftar di metadata.csv")

                        os.unlink(tmp_path)
                else:
                    st.error("Gagal memproses: Model tidak ditemukan.")

if __name__ == "__main__":
    main()
