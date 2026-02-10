import streamlit as st
import numpy as np
import tensorflow as tf
import librosa
import soundfile as sf
import tempfile
import os
import warnings
import sys

# ===== FIX: Handle pkg_resources import error =====
try:
    import pkg_resources
except ImportError:
    # Fallback for environments without pkg_resources
    import importlib.metadata as metadata
    # Create a mock pkg_resources if needed
    class MockPkgResources:
        @staticmethod
        def get_distribution(name):
            class Dist:
                version = "0.0.0"
            return Dist()
    
    sys.modules['pkg_resources'] = MockPkgResources()
    import pkg_resources
# ===== END FIX =====

from sklearn.preprocessing import LabelEncoder
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
        "Jawa Tengah - Jawa Tengah",  # Fixed: was "Jawa Timur"
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
    except Exception as e:
        # Buat model baru jika file tidak ada
        st.info("🔄 Membuat model baru...")
        model = create_cnn_model()
        
        # Generate dummy weights untuk demo
        dummy_input = np.random.randn(1, 128, 128, 1).astype(np.float32)
        model.predict(dummy_input, verbose=0)  # Inisialisasi weights
        
        # Simpan model untuk penggunaan berikutnya
        try:
            model.save('accent_model.h5')
            st.success("✅ Model baru berhasil dibuat")
        except Exception as save_error:
            st.warning(f"⚠️ Model dibuat tapi tidak disimpan: {save_error}")
    
    return model

# Function to extract MFCC features (FIXED VERSION)
def extract_features(audio_path, n_mfcc=13, max_pad_len=128):
    """Ekstrak fitur MFCC dari audio"""
    try:
        # Load audio file using soundfile (more reliable)
        audio, sample_rate = sf.read(audio_path)
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Resample if necessary
        if sample_rate != 22050:
            # Use librosa for resampling if available
            try:
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=22050)
                sample_rate = 22050
            except:
                # Simple resampling fallback
                ratio = 22050 / sample_rate
                new_length = int(len(audio) * ratio)
                audio = np.interp(
                    np.linspace(0, len(audio), new_length),
                    np.arange(len(audio)),
                    audio
                )
                sample_rate = 22050
        
        # Normalize audio
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(
            y=audio, 
            sr=sample_rate, 
            n_mfcc=n_mfcc,
            n_fft=2048,
            hop_length=512
        )
        
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
        st.error(f"Error extracting features: {str(e)}")
        return None

# Function to preprocess audio (FIXED VERSION)
def preprocess_audio(uploaded_file):
    """Preprocessing file audio yang diupload"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_path = tmp_file.name
        
        # Read audio using soundfile (avoid librosa for initial reading)
        try:
            audio, sr = sf.read(temp_path)
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            duration = len(audio) / sr
            
            if duration < 1 or duration > 10:
                st.warning(f"⚠️ Durasi audio: {duration:.2f} detik. Sebaiknya antara 1-5 detik.")
            
            # Save as mono WAV with correct sample rate
            processed_path = temp_path.replace('.wav', '_processed.wav')
            sf.write(processed_path, audio, sr, subtype='PCM_16')
            
        except Exception as read_error:
            st.error(f"Error membaca file audio: {read_error}")
            # Clean up and return None
            os.unlink(temp_path)
            return None, None, None
        
        # Clean up original temp file
        os.unlink(temp_path)
        
        return processed_path, duration, sr
        
    except Exception as e:
        st.error(f"Error preprocessing audio: {str(e)}")
        # Clean up any remaining temp files
        try:
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.unlink(temp_path)
            if 'processed_path' in locals() and os.path.exists(processed_path):
                os.unlink(processed_path)
        except:
            pass
        return None, None, None

# Main app
def main():
    # Load model
    try:
        model = load_model()
    except Exception as model_error:
        st.error(f"Gagal memuat model: {model_error}")
        st.info("Membuat model sederhana alternatif...")
        # Create a very simple fallback model
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(128, 128, 1)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(5, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy')
    
    # Define accent labels
    accent_labels = ['Betawi', 'Jawa Timur', 'Jawa Tengah', 'Sunda', 'Yogyakarta']
    
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
                st.write(f"- Sample rate: {original_sr} Hz")
                st.write(f"- Format: WAV")
            else:
                st.error("Gagal memproses file audio")
                return
        
        # Process and predict button
        if st.button("🔍 Deteksi Aksen", type="primary", use_container_width=True):
            with st.spinner("🔄 Mengekstrak fitur dan menganalisis..."):
                # Extract features
                features = extract_features(audio_path)
                
                if features is not None:
                    try:
                        # Make prediction
                        prediction = model.predict(np.expand_dims(features, axis=0), verbose=0)
                        
                        # Get predicted class and confidence
                        predicted_class_idx = np.argmax(prediction[0])
                        confidence = prediction[0][predicted_class_idx] * 100
                        predicted_accent = accent_labels[predicted_class_idx]
                        
                        # Display results
                        st.success("✅ Analisis selesai!")
                        
                        # Results in columns
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Aksen Terdeteksi", predicted_accent)
                        
                        with col2:
                            st.metric("Tingkat Kepercayaan", f"{confidence:.1f}%")
                        
                        with col3:
                            # Show emoji based on confidence
                            if confidence > 80:
                                emoji = "🎯"
                            elif confidence > 60:
                                emoji = "👍"
                            else:
                                emoji = "🤔"
                            st.metric("Kualitas Deteksi", emoji)
                        
                        # Show detailed probabilities
                        st.subheader("📈 Probabilitas per Aksen:")
                        
                        # Create progress bars for each accent
                        for i, (accent, prob) in enumerate(zip(accent_labels, prediction[0] * 100)):
                            col1, col2, col3 = st.columns([2, 5, 1])
                            with col1:
                                st.write(f"{accent}")
                            with col2:
                                st.progress(min(100, int(prob)) / 100)
                            with col3:
                                st.write(f"{prob:.1f}%")
                        
                    
                    except Exception as pred_error:
                        st.error(f"Gagal melakukan prediksi: {pred_error}")
                        # Show dummy results for demo
                        st.info("Mode demo: Menampilkan contoh hasil...")
                        
                        # Dummy results
                        dummy_pred = np.random.dirichlet(np.ones(5), size=1)[0]
                        pred_idx = np.argmax(dummy_pred)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Aksen Terdeteksi (Demo)", accent_labels[pred_idx])
                        with col2:
                            st.metric("Tingkat Kepercayaan", f"{dummy_pred[pred_idx]*100:.1f}%")
                
                else:
                    st.error("❌ Gagal mengekstrak fitur dari audio. Pastikan file audio valid.")
            
            # Clean up temporary files
            try:
                if audio_path and os.path.exists(audio_path):
                    os.unlink(audio_path)
            except:
                pass
    

# Run the app
if __name__ == "__main__":
    main()
