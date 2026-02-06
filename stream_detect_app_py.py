import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import os
from tensorflow.keras.models import load_model

# ==========================================================
# 1. FIX SYMBOLIC TENSOR ERROR
# ==========================================================
tf.config.run_functions_eagerly(True)

@tf.keras.utils.register_keras_serializable(package="Custom")
class PrototypicalNetwork(tf.keras.Model):
    def __init__(self, embedding_model=None, **kwargs):
        super(PrototypicalNetwork, self).__init__(**kwargs)
        self.embedding = embedding_model

    @tf.function
    def call(self, support_set=None, query_set=None, support_labels=None, n_way=None, training=False):
        # Logika fleksibel: Jika hanya query_set yang dikirim (saat prediksi)
        # atau jika data dikirim sebagai argumen pertama (default Keras)
        if query_set is not None:
            return self.embedding(query_set)
        return self.embedding(support_set)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embedding_model": tf.keras.layers.serialize(self.embedding)
        })
        return config

# ==========================================================
# 2. FIX FILENOTFOUND ERROR & LOAD DATA
# ==========================================================
@st.cache_resource
def load_all():
    model_path = "model_aksen.h5" # Sesuaikan nama file modelmu
    support_path = "support_set.npy"
    label_path = "support_labels.npy"

    # Cek apakah file ada sebelum di-load
    if not os.path.exists(support_path):
        st.error(f"⚠️ File {support_path} tidak ditemukan! Pastikan sudah di-upload ke GitHub.")
        return None, None, None

    model = load_model(model_path, custom_objects={'PrototypicalNetwork': PrototypicalNetwork}, compile=False)
    support_set = np.load(support_path)
    support_labels = np.load(label_path)
    
    return model, support_set, support_labels

# ==========================================================
# 3. RUN APP
# ==========================================================
st.title("Deteksi Aksen Suara")

# Panggil fungsi load_all dengan aman
model, support_set, support_labels = load_all()

if model is not None:
    uploaded_file = st.file_uploader("Pilih file audio...", type=["wav", "mp3"])
    
    if uploaded_file is not None:
        # Tambahkan bagian Feature Extraction kamu di sini (MFCC)
        # ...
        
        # Saat melakukan prediksi:
        # prediction = model(input_mfcc) 
        st.success("Model dan Support Set berhasil dimuat!")

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
