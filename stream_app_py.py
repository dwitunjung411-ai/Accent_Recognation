import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tempfile
import os
import tensorflow as tf

# Fungsi untuk memuat metadata CSV
def load_metadata(csv_path):
    return pd.read_csv(csv_path)

# Fungsi untuk memproses file audio dan memprediksi aksen
def predict_accent(audio_path):
    # Memuat model aksen
    model = load_accent_model()
    
    # Ekstraksi fitur audio (MFCC)
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    # Menyesuaikan bentuk data untuk model (jika perlu)
    mfcc = np.expand_dims(mfcc, axis=-1)  # Menambahkan dimensi saluran jika model membutuhkan

    # Melakukan prediksi aksen
    aksen_probs = model.predict(np.expand_dims(mfcc, axis=0))  # Tambahkan dimensi batch
    
    # Menampilkan probabilitas untuk setiap aksen
    aksen_classes = ["Sunda", "Jawa_Tengah", "Jawa_Timur", "YogyaKarta", "Betawi"]  # Sesuaikan dengan kelas aksen Anda
    predicted_accent = aksen_classes[np.argmax(aksen_probs)]
    
    # Menghitung probabilitas (kepercayaan model)
    accuracy = np.max(aksen_probs)  # Akurasi = probabilitas tertinggi

    return predicted_accent, accuracy

# Fungsi untuk memuat model aksen yang sudah terlatih
def load_accent_model():
    # Memuat model yang sudah terlatih dalam format .keras
    model = ("model_aksen.keras")  # Path ke model aksen Anda
    return model

# Main app
def main():
    # Inisialisasi session state
    if 'prediction_made' not in st.session_state:
        st.session_state.prediction_made = False
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Demo mode selection
        demo_mode = st.radio(
            "Select Mode:",
            ["Upload Audio"]
        )
        
        # Metadata inputs
        st.subheader("📋 Metadata")
        usia = st.number_input("Usia", 0, 100, 25)
        gender = st.selectbox("Gender", ["Male", "Female"])
        provinsi = st.selectbox("Provinsi", [
            "Jawa Barat", "Jawa Tengah", "Jawa Timur", 
            "Yogyakarta", "Jakarta"
        ])
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🎵 Audio Input")
        
        audio_file = None
        
        if demo_mode == "Upload Audio":
            audio_file = st.file_uploader(
                "Upload file audio (.wav, .mp3)",
                type=["wav", "mp3"],
                help="Upload audio untuk deteksi aksen"
            )
        else:
            # Create a sample audio option
            st.info("Using sample audio for demonstration")
            # You can add actual sample audio files here
    
    with col2:
        st.header("Metadata Summary")
        st.metric("Usia", usia)
        st.metric("Gender", gender)
        st.metric("Provinsi", provinsi)
    
    # Process audio if available
    if audio_file is not None or demo_mode == "Use Sample Audio":
        st.divider()
        
        # Display audio player
        if audio_file is not None:
            st.audio(audio_file, format="audio/wav")
        
        # Feature extraction button
        if st.button("🚀 Extract Features and Predict", type="primary"):
            with st.spinner("Processing audio..."):
                try:
                    # Save uploaded file temporarily
                    if audio_file is not None:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                            tmp_file.write(audio_file.getbuffer())
                            tmp_path = tmp_file.name
                    else:
                        # Use a dummy path for sample mode
                        tmp_path = "sample.wav"
                    
                    # Memuat metadata CSV
                    metadata = load_metadata("metadata.csv")
                    
                    # Mencari metadata berdasarkan nama file audio yang di-upload
                    file_name = audio_file.name
                    metadata_info = metadata[metadata['file_name'] == file_name]
                    
                    if not metadata_info.empty:
                        # Ambil informasi metadata
                        usia = metadata_info['usia'].values[0]
                        gender = metadata_info['gender'].values[0]
                        provinsi = metadata_info['provinsi'].values[0]
                        
                        # Tampilkan metadata yang terkait dengan audio
                        st.markdown(f"<h2 style='color:#FF6347;'><i class='fas fa-calendar'></i> **Usia:** {usia}</h2>", unsafe_allow_html=True)
                        st.markdown(f"<h2 style='color:#FF6347;'><i class='fas fa-venus-mars'></i> **Gender:** {gender}</h2>", unsafe_allow_html=True)
                        st.markdown(f"<h2 style='color:#FF6347;'><i class='fas fa-map-marker-alt'></i> **Provinsi:** {provinsi}</h2>", unsafe_allow_html=True)
                        
                        # Prediksi aksen dari audio yang di-upload
                        aksen, accuracy = predict_accent(tmp_path)
                        st.markdown(f"<h2 style='color:#FF6347;'><i class='fas fa-volume-up'></i> 🎭 **Prediksi Aksen:** {aksen} - {accuracy*100:.2f}%</h2>", unsafe_allow_html=True)
                    else:
                        st.write("Metadata tidak ditemukan untuk audio ini.")
                
                except Exception as e:
                    st.error(f"Error during processing: {str(e)}")

# Run the app
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        st.info("Please check the terminal for detailed error messages.")
