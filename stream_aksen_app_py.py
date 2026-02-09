import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tempfile
import os

# Force TensorFlow to use CPU only
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings

import tensorflow as tf

# Configure TensorFlow for CPU
tf.config.set_visible_devices([], 'GPU')

# ==========================================================
# 1. DEFINISI CLASS PROTOTYPICAL NETWORK (CPU-OPTIMIZED)
# ==========================================================
@tf.keras.utils.register_keras_serializable(package="Custom")
class PrototypicalNetwork(tf.keras.Model):
    """
    Prototypical Network untuk Few-Shot Learning
    Optimized untuk CPU-only environment (Streamlit Cloud)
    """
    def __init__(self, embedding_model=None, **kwargs):
        super(PrototypicalNetwork, self).__init__(**kwargs)
        self.embedding = embedding_model

    def call(self, inputs, training=None):
        """
        Method call yang sudah diperbaiki
        Menerima inputs standar Keras (bukan 4 parameter seperti sebelumnya)
        """
        if self.embedding is not None:
            return self.embedding(inputs, training=training)
        return inputs

    def get_config(self):
        """Serialize konfigurasi model"""
        config = super().get_config()
        config.update({
            "embedding_model": tf.keras.layers.serialize(self.embedding) if self.embedding else None
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        """Deserialize model dari konfigurasi"""
        embedding_model = None
        if config.get("embedding_model"):
            embedding_model = tf.keras.layers.deserialize(config["embedding_model"])
        return cls(embedding_model=embedding_model)

# ==========================================================
# 2. FUNGSI LOAD DATA (CPU-OPTIMIZED)
# ==========================================================
@st.cache_resource
def load_accent_model():
    """
    Load model Prototypical Network untuk deteksi aksen
    Optimized untuk CPU-only environment
    """
    model_name = "model_embedding_aksen.keras"
    
    # Cari model di berbagai lokasi
    possible_paths = [
        model_name,  # Current directory
        os.path.join(os.getcwd(), model_name),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), model_name),
    ]
    
    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        st.sidebar.warning(f"⚠️ Model file '{model_name}' tidak ditemukan")
        st.sidebar.caption("Pastikan file model ada di repository")
        return None

    try:
        # Load model dengan CPU
        with tf.device('/CPU:0'):
            custom_objects = {"PrototypicalNetwork": PrototypicalNetwork}
            model = tf.keras.models.load_model(
                model_path, 
                custom_objects=custom_objects, 
                compile=False
            )
        
        st.sidebar.success(f"✅ Model loaded (CPU mode)")
        return model
        
    except Exception as e:
        st.sidebar.error(f"❌ Error loading model: {str(e)}")
        return None

@st.cache_data
def load_metadata_df():
    """
    Load metadata pembicara dari CSV
    """
    csv_name = "metadata.csv"
    
    # Cari CSV di berbagai lokasi
    possible_paths = [
        csv_name,
        os.path.join(os.getcwd(), csv_name),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), csv_name),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                st.sidebar.success(f"✅ Metadata: {len(df)} records")
                return df
            except Exception as e:
                st.sidebar.error(f"❌ Error loading metadata: {str(e)}")
                return None
    
    st.sidebar.info("ℹ️ Metadata file tidak tersedia (opsional)")
    return None

# ==========================================================
# 3. FUNGSI EKSTRAKSI FITUR AUDIO (CPU-OPTIMIZED)
# ==========================================================
def extract_mfcc_features(audio_path, sr=16000, n_mfcc=40):
    """
    Ekstraksi MFCC features dari audio file
    Optimized untuk CPU
    
    Parameters:
    - audio_path: path ke file audio
    - sr: sample rate (default 16000 Hz)
    - n_mfcc: jumlah koefisien MFCC (default 40)
    
    Returns:
    - mfcc_scaled: array MFCC yang sudah dinormalisasi
    """
    try:
        # Load audio file dengan librosa
        y, sr = librosa.load(audio_path, sr=sr, mono=True)
        
        # Ekstraksi MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        
        # Rata-rata temporal untuk mendapatkan vektor fitur
        mfcc_scaled = np.mean(mfcc.T, axis=0)
        
        return mfcc_scaled
        
    except Exception as e:
        st.error(f"❌ Error extracting features: {str(e)}")
        return None

# ==========================================================
# 4. FUNGSI PREDIKSI (CPU-OPTIMIZED)
# ==========================================================
def predict_accent(audio_path, model):
    """
    Prediksi aksen dari file audio
    Optimized untuk CPU-only environment
    
    Returns:
    - tuple: (nama_aksen, confidence_score) atau (error_message, 0.0)
    """
    if model is None: 
        return "Model tidak tersedia", 0.0
    
    try:
        # Ekstraksi fitur MFCC
        mfcc_features = extract_mfcc_features(audio_path)
        
        if mfcc_features is None:
            return "Gagal ekstraksi fitur audio", 0.0
        
        # Prepare input dengan shape yang benar (batch_size, features)
        input_data = np.expand_dims(mfcc_features, axis=0).astype(np.float32)
        
        # Prediksi dengan CPU
        with tf.device('/CPU:0'):
            try:
                # Metode 1: Panggil model secara langsung
                prediction = model(input_data, training=False)
                
            except (TypeError, ValueError) as e:
                # Metode 2: Fallback - akses embedding layer langsung
                if hasattr(model, 'embedding') and model.embedding is not None:
                    prediction = model.embedding(input_data, training=False)
                else:
                    raise e

        # Convert ke numpy jika masih tensor
        if hasattr(prediction, 'numpy'):
            prediction = prediction.numpy()
        
        # Ensure prediction is 2D array
        if len(prediction.shape) == 1:
            prediction = np.expand_dims(prediction, axis=0)

        # Definisi kelas aksen (sesuaikan dengan model Anda)
        aksen_classes = ["Sunda", "Jawa Tengah", "Jawa Timur", "Yogyakarta", "Betawi"]
        
        # Dapatkan prediksi dan confidence
        predicted_index = np.argmax(prediction[0])
        confidence = np.max(prediction[0]) * 100
        
        # Validasi index
        if predicted_index >= len(aksen_classes):
            return "Index prediksi tidak valid", 0.0
        
        return aksen_classes[predicted_index], confidence
        
    except Exception as e:
        return f"Error Analisis: {str(e)}", 0.0

# ==========================================================
# 5. MAIN UI APPLICATION
# ==========================================================
def main():
    # ==================== PAGE CONFIG ====================
    st.set_page_config(
        page_title="Deteksi Aksen Indonesia", 
        page_icon="🎙️", 
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # ==================== CUSTOM CSS ====================
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 1rem;
        }
        .sub-header {
            color: #888;
            margin-bottom: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # ==================== LOAD RESOURCES ====================
    model_aksen = load_accent_model()
    df_metadata = load_metadata_df()

    # ==================== HEADER ====================
    st.markdown('<h1 class="main-header">🎙️ Sistem Deteksi Aksen Indonesia</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Aplikasi berbasis <b>Few-Shot Learning</b> dengan <b>Prototypical Network</b> untuk klasifikasi aksen daerah Indonesia</p>', unsafe_allow_html=True)
    st.divider()

    # ==================== SIDEBAR ====================
    with st.sidebar:
        st.header("🛸 Status Sistem")
        
        # Environment Info
        st.caption(f"🖥️ TensorFlow: {tf.__version__}")
        st.caption(f"💻 Device: CPU Only (Streamlit Cloud)")
        
        st.divider()
        
        # Status Model
        if model_aksen:
            st.success("✅ Model: Terhubung")
        else:
            st.error("❌ Model: Terputus")
            st.caption("Upload `model_embedding_aksen.keras` ke repository")

        # Status Metadata
        if df_metadata is not None:
            st.success(f"✅ Metadata: {len(df_metadata)} records")
        else:
            st.info("ℹ️ Metadata: Opsional")

        st.divider()
        
        # Info Aksen yang Didukung
        st.subheader("🗺️ Aksen yang Didukung")
        aksen_list = ["Sunda", "Jawa Tengah", "Jawa Timur", "Yogyakarta", "Betawi"]
        for i, aksen in enumerate(aksen_list, 1):
            st.caption(f"{i}. {aksen}")
        
        st.divider()
        
        # Tech Stack
        with st.expander("📚 Tech Stack"):
            st.caption("• TensorFlow (CPU)")
            st.caption("• Librosa (MFCC)")
            st.caption("• Streamlit Cloud")
            st.caption("• Prototypical Network")
        
        st.divider()
        st.caption("🎓 Skripsi Project - 2026")

    # ==================== MAIN CONTENT ====================
    col1, col2 = st.columns([1, 1.2], gap="large")

    # -------------------- COLUMN 1: INPUT --------------------
    with col1:
        st.subheader("📥 Input Audio")
        
        # File uploader
        audio_file = st.file_uploader(
            "Upload file audio (.wav, .mp3)", 
            type=["wav", "mp3"],
            help="Upload file audio untuk dianalisis aksennya"
        )

        if audio_file:
            # Display audio player
            st.audio(audio_file)
            
            # Info file
            file_size = len(audio_file.getvalue()) / 1024  # KB
            st.caption(f"📄 **{audio_file.name}** ({file_size:.1f} KB)")
            
            st.divider()
            
            # Tombol prediksi
            predict_button = st.button(
                "🚀 Analisis Aksen", 
                type="primary", 
                use_container_width=True,
                disabled=(model_aksen is None)
            )
            
            if model_aksen is None:
                st.warning("⚠️ Upload model untuk melakukan prediksi")
            
            # -------------------- PROSES PREDIKSI --------------------
            if predict_button and model_aksen:
                with st.spinner("🔍 Menganalisis karakteristik suara..."):
                    # Save uploaded file ke temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(audio_file.getbuffer())
                        tmp_path = tmp.name

                    # Prediksi
                    hasil_aksen, confidence = predict_accent(tmp_path, model_aksen)

                    # Cari metadata pembicara
                    user_info = None
                    if df_metadata is not None:
                        match = df_metadata[df_metadata['file_name'] == audio_file.name]
                        if not match.empty:
                            user_info = match.iloc[0].to_dict()

                    # -------------------- COLUMN 2: HASIL --------------------
                    with col2:
                        st.subheader("📊 Hasil Analisis")
                        
                        # Container untuk hasil prediksi
                        with st.container(border=True):
                            st.markdown("#### 🎭 Aksen Terdeteksi")
                            
                            if "Error" not in hasil_aksen:
                                # Sukses prediksi
                                st.success(f"### **{hasil_aksen}**")
                                
                                # Confidence score
                                if confidence > 0:
                                    col_conf1, col_conf2 = st.columns([2, 1])
                                    
                                    with col_conf1:
                                        st.metric(
                                            label="Tingkat Keyakinan", 
                                            value=f"{confidence:.2f}%"
                                        )
                                    
                                    with col_conf2:
                                        # Emoji berdasarkan confidence
                                        if confidence >= 80:
                                            st.markdown("### ✅")
                                        elif confidence >= 60:
                                            st.markdown("### ⚠️")
                                        else:
                                            st.markdown("### ❓")
                                    
                                    # Progress bar confidence
                                    st.progress(min(confidence / 100, 1.0))
                                    
                                    # Interpretasi confidence
                                    if confidence >= 80:
                                        st.caption("✅ Prediksi sangat yakin")
                                    elif confidence >= 60:
                                        st.caption("⚠️ Prediksi cukup yakin")
                                    else:
                                        st.caption("❓ Prediksi kurang yakin")
                            else:
                                # Error prediksi
                                st.error(hasil_aksen)

                        st.divider()
                        
                        # Info pembicara dari metadata
                        st.subheader("💎 Info Pembicara")
                        
                        if user_info:
                            col_info1, col_info2 = st.columns(2)
                            
                            with col_info1:
                                st.metric("🎂 Usia", f"{user_info.get('usia', '-')} Tahun")
                                st.metric("🚻 Gender", user_info.get('gender', '-'))
                            
                            with col_info2:
                                st.metric("🗺️ Provinsi", user_info.get('provinsi', '-'))
                                if 'kota' in user_info:
                                    st.metric("🏙️ Kota", user_info.get('kota', '-'))
                        else:
                            st.info("🕵️ Data pembicara tidak ditemukan di metadata")

                    # Cleanup temp file
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass

        else:
            # Placeholder jika belum ada file
            with col2:
                st.info("👈 Upload file audio untuk memulai analisis")
                
                st.markdown("### 📖 Cara Menggunakan")
                st.markdown("""
                1. **Upload** file audio (.wav atau .mp3)
                2. **Klik** tombol "Analisis Aksen"
                3. **Tunggu** hingga analisis selesai
                4. **Lihat** hasil prediksi aksen
                """)
                
                st.divider()
                
                st.markdown("### 🎯 Fitur Utama")
                col_f1, col_f2 = st.columns(2)
                
                with col_f1:
                    st.markdown("""
                    - ✅ 5 Aksen Indonesia
                    - ✅ CPU Optimized
                    - ✅ MFCC Features
                    """)
                
                with col_f2:
                    st.markdown("""
                    - ✅ Confidence Score
                    - ✅ Metadata Support
                    - ✅ Real-time Analysis
                    """)

    # ==================== FOOTER ====================
    st.divider()
    col_f1, col_f2, col_f3 = st.columns(3)
    
    with col_f1:
        st.caption("🔬 Few-Shot Learning")
    
    with col_f2:
        st.caption("🧠 Prototypical Network")
    
    with col_f3:
        st.caption("☁️ Streamlit Cloud")

# ==========================================================
# 6. RUN APPLICATION
# ==========================================================
if __name__ == "__main__":
    main()
