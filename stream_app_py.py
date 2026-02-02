import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import librosa.display
import joblib
import os
from sklearn.preprocessing import StandardScaler
import tempfile

st.title("🎵 Few-Shot Audio Classification")
st.write("Menggunakan Embedding Network dari Skripsi")

# --- Fungsi untuk Load Model ---
@st.cache_resource
def load_embedding_model():
    """Load embedding model dengan arsitektur dari skripsi"""
    
    model_paths = [
        'embedding_model.keras',
        'embedding_weights.h5',
        'model/embedding_model.keras',
        'embeddings/embedding_weights.h5'
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            try:
                if path.endswith('.keras'):
                    model = tf.keras.models.load_model(path)
                    st.sidebar.success(f"✅ Model loaded from {path}")
                    return model
                elif path.endswith('.h5'):
                    # Jika hanya weights, perlu bangun arsitektur dulu
                    from tensorflow.keras import layers
                    
                    # Bangun model dengan arsitektur sama
                    input_shape = (40, 100, 1)  # SESUAIKAN
                    model = tf.keras.Sequential([
                        layers.Conv2D(32, (3, 3), activation='relu', 
                                     input_shape=input_shape),
                        layers.MaxPooling2D((2, 2)),
                        layers.Conv2D(64, (3, 3), activation='relu'),
                        layers.MaxPooling2D((2, 2)),
                        layers.Conv2D(128, (3, 3), activation='relu'),
                        layers.MaxPooling2D((2, 2)),
                        layers.GlobalAveragePooling2D(),
                        layers.Dense(256, activation='relu'),
                        layers.Dropout(0.3),
                        layers.Dense(128, activation='relu')
                    ])
                    model.load_weights(path)
                    st.sidebar.success(f"✅ Weights loaded from {path}")
                    return model
            except Exception as e:
                st.sidebar.error(f"❌ Error loading {path}: {str(e)}")
    
    # Jika model tidak ditemukan
    st.sidebar.error("""
    ❌ Embedding model tidak ditemukan!
    
    Solusi:
    1. Jalankan `python create_dummy_model.py` untuk membuat model dummy
    2. Atau tempatkan model hasil training di root folder
    3. Model harus bernama: `embedding_model.keras` atau `embedding_weights.h5`
    """)
    
    return None

# --- Fungsi Preprocessing Audio ---
def extract_features(audio_path, sr=22050, n_mfcc=40, fixed_length=100):
    """Ekstrak fitur MFCC dari audio"""
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=sr)
        
        # Ekstrak MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        
        # Normalisasi panjang
        if mfcc.shape[1] > fixed_length:
            mfcc = mfcc[:, :fixed_length]
        else:
            pad_width = fixed_length - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
        
        # Reshape untuk model (height, width, channels)
        mfcc = np.expand_dims(mfcc.T, axis=-1)  # Shape: (100, 40, 1)
        mfcc = np.transpose(mfcc, (1, 0, 2))    # Shape: (40, 100, 1)
        
        return mfcc
        
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        return None

# --- Load Model ---
embedding_model = load_embedding_model()

# --- Interface Utama ---
if embedding_model is not None:
    st.sidebar.header("⚙️ Pengaturan Episode")
    
    n_way = st.sidebar.number_input("Jumlah kelas (n_way)", 
                                    min_value=2, max_value=10, value=5)
    k_shot = st.sidebar.number_input("Jumlah contoh per kelas (k_shot)", 
                                     min_value=1, max_value=5, value=3)
    n_query = st.sidebar.number_input("Jumlah query audio", 
                                      min_value=1, max_value=5, value=1)
    
    st.sidebar.divider()
    st.sidebar.header("📂 Upload Audio")
    
    uploaded_files = st.sidebar.file_uploader(
        "Upload file audio (WAV, MP3)",
        type=['wav', 'mp3'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.write(f"📁 Uploaded {len(uploaded_files)} files")
        
        # Proses file
        for i, uploaded_file in enumerate(uploaded_files):
            # Simpan file sementara
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            # Ekstrak fitur
            features = extract_features(tmp_path)
            
            if features is not None:
                st.write(f"File {i+1}: {uploaded_file.name}")
                
                # Visualisasi MFCC
                fig, ax = plt.subplots(figsize=(10, 4))
                img = librosa.display.specshow(
                    features[:, :, 0], 
                    x_axis='time',
                    ax=ax
                )
                plt.colorbar(img, ax=ax)
                ax.set(title=f'MFCC - {uploaded_file.name}')
                st.pyplot(fig)
                
                # Generate embedding
                input_tensor = np.expand_dims(features, axis=0)
                embedding = embedding_model.predict(input_tensor, verbose=0)
                
                st.write(f"Embedding shape: {embedding.shape}")
                st.write(f"Embedding sample: {embedding[0, :5]}")  # 5 nilai pertama
                
                os.unlink(tmp_path)  # Hapus file temp
        
        # Tombol prediksi
        if st.button("🚀 Jalankan Prediksi Few-Shot", type="primary"):
            st.success("Prediksi berhasil dijalankan!")
            st.write(f"Settings: {n_way}-way, {k_shot}-shot")
            
            # Di sini tambahkan logika few-shot learning Anda
            # Contoh: prototypical networks, matching networks, dll.
            
else:
    st.warning("""
    ⚠️ Model embedding belum tersedia.
    
    Silakan:
    1. **Untuk testing cepat**: Jalankan `python create_dummy_model.py`
    2. **Untuk model asli**: Train model embedding dengan dataset Anda, lalu simpan sebagai `embedding_model.keras`
    
    Setelah model tersedia, refresh halaman ini.
    """)
    
    # Tombol untuk membuat model dummy
    if st.button("🛠️ Buat Model Dummy untuk Testing"):
        import subprocess
        with st.spinner("Membuat model dummy..."):
            result = subprocess.run(["python", "create_dummy_model.py"], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                st.success("✅ Model dummy berhasil dibuat!")
                st.info("Silakan refresh halaman (F5)")
            else:
                st.error("❌ Gagal membuat model dummy")
                st.code(result.stderr)

# --- Footer ---
st.divider()
st.caption("Few-Shot Audio Classification System - Skripsi Implementation")
