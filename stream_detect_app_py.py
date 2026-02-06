import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tempfile
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

# ==========================================================
# 1. DEFINISI CLASS PROTOTYPICAL NETWORK (WAJIB ADA)
# Pastikan kode di dalam class ini persis sama dengan di Colab kamu
# ==========================================================
@tf.keras.utils.register_keras_serializable(package="Custom")
class PrototypicalNetwork(tf.keras.Model):
 def __init__(self, embedding_model=None, **kwargs):
        super(PrototypicalNetwork, self).__init__(**kwargs)
	self.embedding = embedding_model

  def call(self, support_set, query_set, support_labels, n_way):


@st.cache_resource
def load_accent_model():
model_path= "model_aksen.keras"
if os.path.exists(model_path):

try:
            # Gunakan penamaan yang sesuai dengan metadata model Anda
            custom_objects = {"PrototypicalNetwork": PrototypicalNetwork}
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
            return model
        except Exception as e:
            st.error(f"Error saat loading: {e}")
            return None
    else:

 st.error(f"File {model_path} tidak ditemukan di server!")
        return None


model_aksen = load_accent_model()

# ==========================================================
# 3. FUNGSI PEMROSESAN AUDIO
# ==========================================================
def load_metadata(csv_path):
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return pd.DataFrame()

def predict_accent(audio_path, model):
    if model is None:
        return "Model tidak terload"

    # Ekstraksi Fitur (Sesuaikan dengan durasi/shape saat training)
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # Preprocessing: Sesuaikan shape mfcc agar sesuai input model (n_samples, n_mfcc, time)
    # Ini hanya contoh, sesuaikan dengan bentuk input model skripsi kamu
    mfcc_resized = np.mean(mfcc.T, axis=0)
    input_data = np.expand_dims(mfcc_resized, axis=0)

    # Prediksi menggunakan model (Bukan string lagi)
    aksen_probs = model.predict(input_data)

    # Contoh mapping label (Sesuaikan dengan urutan label skripsi kamu)
    aksen_classes = ["Sunda", "Jawa Tengah", "Jawa Timur", "Yogyakarta", "Betawi"]
    predicted_idx = np.argmax(aksen_probs)
    return aksen_classes[predicted_idx]

# ==========================================================
# 4. MAIN APP
# ==========================================================
def main():
    st.set_page_config(page_title="Deteksi Aksen Prototypical", layout="wide")

    if 'prediction_made' not in st.session_state:
        st.session_state.prediction_made = False

    with st.sidebar:
        st.header("⚙️ Settings")
        demo_mode = st.radio("Select Mode:", ["Upload Audio"])

    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("🎵 Audio Input")
        audio_file = st.file_uploader("Upload file audio (.wav, .mp3)", type=["wav", "mp3"])

    if audio_file is not None:
        st.divider()
        st.audio(audio_file, format="audio/wav")

        if st.button("🚀 Extract Features and Detect", type="primary"):
            with st.spinner("Processing audio..."):
                try:
                    # Simpan audio sementara
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                        tmp_file.write(audio_file.getbuffer())
                        tmp_path = tmp_file.name

                    # Metadata handling
                    metadata = load_metadata("metadata.csv")
                    file_name = audio_file.name
                    metadata_info = metadata[metadata['file_name'] == file_name] if not metadata.empty else pd.DataFrame()

                    if not metadata_info.empty:
                        usia = metadata_info['usia'].values[0]
                        gender = metadata_info['gender'].values[0]
                        provinsi = metadata_info['provinsi'].values[0]

                        st.subheader("Informasi Pembicara:")
                        st.write(f"📅Usia: {usia}")
                        st.write(f"🗣️Gender: {gender}")
                        st.write(f"📍Provinsi: {provinsi}")

                    # PROSES PREDIKSI
                    # Melewatkan objek model_aksen (bukan string) ke fungsi
                    hasil_aksen = predict_accent(tmp_path, model_aksen)

                    st.success(f"### 🎭 Deteksi Aksen: {hasil_aksen}")

                    # Hapus file sementara
                    os.unlink(tmp_path)

                except Exception as e:
                    st.error(f"Error during processing: {str(e)}")

if __name__ == "__main__":
    main()
