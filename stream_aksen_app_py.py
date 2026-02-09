import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tempfile
import os
import tensorflow as tf

# ==========================================================
# 1. DEFINISI CLASS PROTOTYPICAL NETWORK (VERSI STABIL)
# ==========================================================
@tf.keras.utils.register_keras_serializable(package="Custom")
class PrototypicalNetwork(tf.keras.Model):
    def __init__(self, embedding_model=None, **kwargs):
        # Hindari error serialisasi dengan memastikan embedding_model ditangani dengan benar
        super(PrototypicalNetwork, self).__init__(**kwargs)
        self.embedding = embedding_model

    def call(self, inputs):
        return self.embedding(inputs)

    def get_config(self):
        config = super().get_config()
        # Menggunakan serialize layer yang lebih kompatibel
        config.update({
            "embedding_model": tf.keras.utils.serialize_keras_object(self.embedding)
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Proses rekonstruksi object embedding_model dari config
        embedding_config = config.pop("embedding_model")
        embedding_model = tf.keras.utils.deserialize_keras_object(embedding_config)
        return cls(embedding_model=embedding_model, **config)

# ==========================================================
# 2. FUNGSI LOAD DATA (MODEL & METADATA)
# ==========================================================
@st.cache_resource
def load_accent_model():
    model_name = "model_embedding_aksen.keras"
    # Menggunakan path absolut agar lebih pasti
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, model_name)

    if not os.path.exists(model_path):
        return None

    try:
        # Menambahkan custom_objects yang diperlukan
        custom_objects = {
            "PrototypicalNetwork": PrototypicalNetwork,
            "Functional": tf.keras.Model # Seringkali diperlukan untuk model embedding
        }
        
        # Load model tanpa kompilasi untuk menghindari error optimizer kustom
        model = tf.keras.models.load_model(
            model_path, 
            custom_objects=custom_objects, 
            compile=False
        )
        return model
    except Exception as e:
        # Menampilkan error di log atau sidebar untuk debugging
        st.sidebar.error(f"Pesan Error: {str(e)}")
        return None

@st.cache_data
def load_metadata_df():
    csv_path = "metadata.csv"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

# ==========================================================
# 3. FUNGSI PREDIKSI
# ==========================================================
def predict_accent(audio_path, model):
    if model is None: return "Model tidak tersedia"
    try:
        # Load audio dengan durasi tetap (misal 3 detik) agar feature konsisten
        y, sr = librosa.load(audio_path, sr=16000, duration=3.0)
        
        # Ekstraksi MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_scaled = np.mean(mfcc.T, axis=0)
        
        # Tambahkan dimensi batch
        input_data = np.expand_dims(mfcc_scaled, axis=0)

        # Melakukan prediksi
        # Jika model adalah embedding model, hasilnya mungkin berupa vector. 
        # Pastikan output layer model Anda adalah Dense(5, activation='softmax')
        prediction = model.predict(input_data, verbose=0)
        
        aksen_classes = ["Sunda", "Jawa Tengah", "Jawa Timur", "Yogyakarta", "Betawi"]
        return aksen_classes[np.argmax(prediction)]
    except Exception as e:
        return f"Error Analisis: {str(e)}"

# ==========================================================
# 4. MAIN UI
# ==========================================================
def main():
    st.set_page_config(page_title="Accent Recognition", page_icon="🎙️", layout="wide")

    model_aksen = load_accent_model()
    df_metadata = load_metadata_df()

    st.title("🎙️ Accent Recognition")
    st.divider()

    # SIDEBAR STATUS
    with st.sidebar:
        st.header("⚙️ Status Sistem")
        if model_aksen:
            st.success("Model: Online")
        else:
            st.error("Model: Offline")
            st.info("Pastikan file 'model_detect_aksen.keras' ada di folder yang sama.")

    # LAYOUT KOLOM
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader("📤 Input Audio")
        audio_file = st.file_uploader("Upload file (.wav, .mp3)", type=["wav", "mp3"])

        if audio_file:
            st.audio(audio_file)
            if st.button("🚀 Extract Feature and Detect", type="primary", use_container_width=True):
                if model_aksen:
                    with st.spinner("Sedang memproses..."):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                            tmp.write(audio_file.getbuffer())
                            tmp_path = tmp.name

                        try:
                            hasil_aksen = predict_accent(tmp_path, model_aksen)
                            
                            # Pencarian Metadata
                            user_info = None
                            if df_metadata is not None:
                                match = df_metadata[df_metadata['file_name'] == audio_file.name]
                                if not match.empty:
                                    user_info = match.iloc[0].to_dict()

                            with col2:
                                st.subheader("📊 Hasil Analisis")
                                st.info(f"### Aksen Terdeteksi: **{hasil_aksen}**")
                                st.divider()
                                st.subheader("🔹 Info Pembicara")
                                if user_info:
                                    st.write(f"📅 **Usia:** {user_info.get('usia', '-')}")
                                    st.write(f"🗣️ **Gender:** {user_info.get('gender', '-')}")
                                    st.write(f"📍 **Provinsi:** {user_info.get('provinsi', '-')}")
                                else:
                                    st.warning("Data file tidak terdaftar di metadata.csv")
                        finally:
                            if os.path.exists(tmp_path):
                                os.remove(tmp_path)
                else:
                    st.error("Model gagal dimuat. Hubungi Admin.")

if __name__ == "__main__":
    main()
