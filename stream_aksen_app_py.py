import streamlit as st
import numpy as np
import tensorflow as tf
import librosa
import soundfile as sf
import tempfile
import os
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Deteksi Aksen Bahasa Indonesia",
    page_icon="🗣️",
    layout="wide"
)

# Title and description
st.title("🗣️ Deteksi Aksen Bahasa Indonesia")
st.markdown("""
Aplikasi ini mendeteksi aksen bahasa Indonesia dari 5 daerah:
1. **Betawi** - DKI Jakarta
2. **Jawa Timur** - Jawa Timur  
3. **Jawa Tengah** - Jawa Tengah
4. **Sunda** - Jawa Barat
5. **Yogyakarta** - D.I. Yogyakarta
""")

# Sidebar for information
with st.sidebar:
    st.header("ℹ️ Informasi Aplikasi")
    
    st.subheader("Aksen yang didukung:")
    aksen_list = [
        "Betawi - DKI Jakarta",
        "Jawa Timur - Jawa Timur",
        "Jawa Tengah - Jawa Timur",
        "Sunda - Jawa Barat",
        "Yogyakarta - D.I. Yogyakarta"
    ]
    for aksen in aksen_list:
        st.write(f"• {aksen}")
    
    st.subheader("Cara penggunaan:")
    st.write("1. Unggah file audio (.wav)")
    st.write("2. Audio akan diproses otomatis")
    st.write("3. Lihat hasil deteksi aksen")
    
    st.subheader("Persyaratan audio:")
    st.write("- Format: WAV")
    st.write("- Durasi: 1-5 detik")
    st.write("- Sample rate: 22050 Hz")

# Function to create a simple CNN model
def create_cnn_model(input_shape=(128, 128, 1), num_classes=5):
    """Membuat model CNN sederhana untuk klasifikasi"""
    model = tf.keras.Sequential([
        # First Conv Block
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', 
                              input_shape=input_shape, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # Second Conv Block
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # Third Conv Block
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # Dense layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Initialize or load model
@st.cache_resource
def load_model():
    """Membuat atau memuat model"""
    try:
        # Coba muat model yang sudah ada
        model = tf.keras.models.load_model('accent_model.h5')
        st.success("✅ Model berhasil dimuat dari file")
    except:
        # Buat model baru jika file tidak ada
        st.info("🔄 Membuat model baru...")
        model = create_cnn_model()
        
        # Generate dummy weights untuk demo
        # Dalam aplikasi nyata, Anda akan melatih model terlebih dahulu
        dummy_input = np.random.randn(1, 128, 128, 1)
        model.predict(dummy_input)  # Inisialisasi weights
        
        # Simpan model untuk penggunaan berikutnya
        model.save('accent_model.h5')
        st.success("✅ Model baru berhasil dibuat")
    
    return model

# Function to extract MFCC features
def extract_features(audio_path, n_mfcc=13, max_pad_len=128):
    """Ekstrak fitur MFCC dari audio"""
    try:
        # Load audio file
        audio, sample_rate = librosa.load(audio_path, sr=22050)
        
        # Normalize audio
        audio = audio / np.max(np.abs(audio))
        
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        
        # Pad or truncate to max_pad_len
        if mfccs.shape[1] > max_pad_len:
            mfccs = mfccs[:, :max_pad_len]
        else:
            pad_width = max_pad_len - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        
        # Add channel dimension for CNN
        mfccs = np.expand_dims(mfccs, axis=-1)
        
        return mfccs
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None

# Function to preprocess audio
def preprocess_audio(uploaded_file):
    """Preprocessing file audio yang diupload"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_path = tmp_file.name
        
        # Check audio duration
        audio, sr = librosa.load(temp_path, sr=None)
        duration = len(audio) / sr
        
        if duration < 1 or duration > 10:
            st.warning(f"⚠️ Durasi audio: {duration:.2f} detik. Sebaiknya antara 1-5 detik.")
        
        # Resample if necessary
        if sr != 22050:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=22050)
        
        # Save resampled audio
        resampled_path = temp_path.replace('.wav', '_resampled.wav')
        sf.write(resampled_path, audio, 22050)
        
        # Clean up
        os.unlink(temp_path)
        
        return resampled_path, duration, sr
        
    except Exception as e:
        st.error(f"Error preprocessing audio: {e}")
        return None, None, None

# Main app
def main():
    # Load model
    model = load_model()
    
    # Define accent labels
    accent_labels = ['Betawi', 'Jawa Timur', 'Jawa Tengah', 'Sunda', 'Yogyakarta']
    label_encoder = LabelEncoder()
    label_encoder.fit(accent_labels)
    
    # File upload section
    st.header("📁 Unggah File Audio")
    
    uploaded_file = st.file_uploader(
        "Pilih file audio (.wav)",
        type=['wav'],
        help="Unggah file audio dalam format WAV dengan sample rate 22050 Hz"
    )
    
    if uploaded_file is not None:
        # Display audio info
        col1, col2 = st.columns(2)
        
        with col1:
            st.audio(uploaded_file, format='audio/wav')
        
        with col2:
            # Preprocess audio
            with st.spinner("🔄 Memproses audio..."):
                audio_path, duration, original_sr = preprocess_audio(uploaded_file)
            
            if audio_path:
                st.info(f"📊 Informasi Audio:")
                st.write(f"- Durasi: {duration:.2f} detik")
                st.write(f"- Sample rate asli: {original_sr} Hz")
                st.write(f"- Format: WAV")
        
        # Process and predict
        if st.button("🔍 Deteksi Aksen", type="primary"):
            with st.spinner("🔄 Mengekstrak fitur dan menganalisis..."):
                # Extract features
                features = extract_features(audio_path)
                
                if features is not None:
                    # Make prediction
                    prediction = model.predict(np.expand_dims(features, axis=0), verbose=0)
                    
                    # Get predicted class and confidence
                    predicted_class_idx = np.argmax(prediction[0])
                    confidence = prediction[0][predicted_class_idx] * 100
                    predicted_accent = label_encoder.inverse_transform([predicted_class_idx])[0]
                    
                    # Display results
                    st.success("✅ Analisis selesai!")
                    
                    # Results in columns
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Aksen Terdeteksi", predicted_accent)
                    
                    with col2:
                        st.metric("Tingkat Kepercayaan", f"{confidence:.1f}%")
                    
                    
                    # Show detailed probabilities
                    st.subheader("📈 Probabilitas per Aksen:")
                    
                    # Create progress bars for each accent
                    for i, (accent, prob) in enumerate(zip(accent_labels, prediction[0] * 100)):
                        col1, col2, col3 = st.columns([2, 5, 1])
                        with col1:
                            st.write(f"{accent}")
                        with col2:
                            st.progress(int(prob))
                        with col3:
                            st.write(f"{prob:.1f}%")
                    
                    # Show feature visualization
                    with st.expander("📊 Visualisasi Fitur MFCC"):
                        import matplotlib.pyplot as plt
                        
                        fig, ax = plt.subplots(figsize=(10, 4))
                        mfcc_features = features[:, :, 0].T
                        im = ax.imshow(mfcc_features, aspect='auto', origin='lower', cmap='viridis')
                        ax.set_title('MFCC Features dari Audio')
                        ax.set_xlabel('MFCC Coefficients')
                        ax.set_ylabel('Frames')
                        plt.colorbar(im, ax=ax)
                        st.pyplot(fig)
                
                else:
                    st.error("❌ Gagal mengekstrak fitur dari audio")
            
            # Clean up temporary files
            if audio_path and os.path.exists(audio_path):
                os.unlink(audio_path)



# Run the app
if __name__ == "__main__":
    main()
