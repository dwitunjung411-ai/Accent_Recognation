# streamlit_app.py
import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf
from tensorflow import keras
import tempfile
import os
import sys
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Deteksi Aksen Bahasa Indonesia",
    page_icon="🎙️",
    layout="wide"
)

# Custom CSS untuk styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4B5563;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #F3F4F6;
        margin-top: 20px;
    }
    .confidence-bar {
        height: 20px;
        border-radius: 10px;
        margin: 5px 0;
        background: linear-gradient(90deg, #3B82F6 0%, #10B981 100%);
    }
    .upload-box {
        border: 2px dashed #3B82F6;
        border-radius: 10px;
        padding: 30px;
        text-align: center;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title dan deskripsi
st.markdown('<h1 class="main-header">🎙️ Deteksi Aksen Bahasa Indonesia</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Unggah file audio untuk mendeteksi aksen bahasa Indonesia (Betawi, Jawa Timur, Jawa Tengah, Sunda, YogyaKarta)</p>', unsafe_allow_html=True)

# Sidebar untuk informasi
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1998/1998610.png", width=100)
    st.title("📊 Informasi Aplikasi")
    st.markdown("""
    ### Aksen yang didukung:
    1. **Betawi** - DKI Jakarta
    2. **Jawa Timur** - Jawa Timur
    3. **Jawa Tengah** - Jawa Tengah
    4. **Sunda** - Jawa Barat
    5. **YogyaKarta** - D.I. Yogyakarta
    
    ### Cara penggunaan:
    1. Unggah file audio (.wav)
    2. Audio akan diproses otomatis
    3. Lihat hasil deteksi aksen
    
    ### Persyaratan audio:
    - Format: WAV
    - Durasi: 1-5 detik
    - Sample rate: 22050 Hz
    """)

# Fungsi untuk ekstraksi fitur (sama seperti di notebook)
def extract_mfcc(file_path, sr=22050, n_mfcc=40, max_len=174):
    """
    Ekstrak MFCC dengan delta dan delta-delta
    """
    try:
        # Load audio
        y, sr = librosa.load(file_path, sr=sr)

        # Normalisasi amplitude
        y = librosa.util.normalize(y)

        # Ekstrak MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=2048, hop_length=512)

        # Delta dan Delta-Delta
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)

        # Padding atau truncating untuk panjang seragam
        if mfcc.shape[1] < max_len:
            pad_width = max_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
            delta = np.pad(delta, ((0, 0), (0, pad_width)), mode='constant')
            delta2 = np.pad(delta2, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_len]
            delta = delta[:, :max_len]
            delta2 = delta2[:, :max_len]

        # Stack menjadi 3 channel
        features = np.stack([mfcc, delta, delta2], axis=-1)

        return features

    except Exception as e:
        st.error(f"Error extracting MFCC: {e}")
        return None

# Definisikan kelas PrototypicalNetwork untuk loading model
@keras.saving.register_keras_serializable()
class PrototypicalNetwork(tf.keras.Model):
    def __init__(self, embedding_model, **kwargs):
        super(PrototypicalNetwork, self).__init__(**kwargs)
        self.embedding = embedding_model

    def call(self, inputs_data, query_set=None, support_labels=None, n_way=None):
        if query_set is None and isinstance(inputs_data, (list, tuple)):
            support_set = inputs_data[0]
            query_set = inputs_data[1]
            support_labels = inputs_data[2]
            n_way = inputs_data[3]
        else:
            # Case: Inputs are passed as separate arguments
            support_set = inputs_data

        # Validate inputs
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
            # Masking to get embeddings for a specific class
            mask = tf.equal(support_labels, i)
            class_embeddings = tf.boolean_mask(support_embeddings, mask)

            # Mean pooling to get the class center (prototype)
            if tf.shape(class_embeddings)[0] == 0:
                prototype = tf.zeros_like(support_embeddings[0])
            else:
                prototype = tf.reduce_mean(class_embeddings, axis=0)
            prototypes.append(prototype)

        prototypes = tf.stack(prototypes)

        # Calculate Euclidean distances between queries and prototypes
        distances = tf.norm(
            tf.expand_dims(query_embeddings, 1) - tf.expand_dims(prototypes, 0),
            axis=2
        )

        # Convert distances to logits
        logits = -distances
        return logits

    def get_config(self):
        config = super(PrototypicalNetwork, self).get_config()
        config.update({
            "embedding_model": keras.saving.serialize_keras_object(self.embedding)
        })
        return config

    @classmethod
    def from_config(cls, config):
        embedding_config = config.pop("embedding_model")
        embedding_model = keras.saving.deserialize_keras_object(embedding_config)
        return cls(embedding_model, **config)

# Fungsi untuk loading model
@st.cache_resource
def load_models():
    """Load model dan support set"""
    try:
        # Definisikan custom objects
        custom_objects = {"PrototypicalNetwork": PrototypicalNetwork}
        
        # Load model (sesuaikan path)
        model = keras.models.load_model(
            "model_detect_aksen.keras", 
            custom_objects=custom_objects,
            compile=False
        )
        
        # Load support set dan labels
        support_set = np.load("support_set.npy")
        support_labels = np.load("support_labels.npy")
        
        # Label encoder untuk aksen
        le_y = LabelEncoder()
        # Sesuaikan dengan kelas yang ada di training
        aksen_classes = ['Betawi', 'Jawa _Timur', 'Jawa_Tengah', 'Sunda', 'YogyaKarta']
        le_y.fit(aksen_classes)
        
        return model, support_set, support_labels, le_y
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None

# Fungsi untuk memproses audio dan membuat prediksi
def predict_accent(audio_file, model, support_set, support_labels, le_y):
    """Prediksi aksen dari file audio"""
    try:
        # Simpan file audio sementara
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            # Ekstrak fitur dari audio
            audio_features = extract_mfcc(tmp_path)
            
            if audio_features is None:
                return None, None
            
            # Tambahkan dimensi batch
            audio_features = np.expand_dims(audio_features, axis=0)
            
            # Konversi ke tensor
            query_tensor = tf.convert_to_tensor(audio_features, dtype=tf.float32)
            support_tensor = tf.convert_to_tensor(support_set, dtype=tf.float32)
            support_labels_tensor = tf.convert_to_tensor(support_labels, dtype=tf.int32)
            
            # Jumlah kelas (n_way)
            n_way = len(np.unique(support_labels))
            
            # Prediksi dengan model
            logits = model.call(
                support_tensor,
                query_tensor,
                support_labels_tensor,
                n_way
            )
            
            # Hitung probabilitas
            probs = tf.nn.softmax(logits, axis=1).numpy()[0]
            pred_index = tf.argmax(logits, axis=1).numpy()[0]
            
            # Decode label
            pred_label = le_y.inverse_transform([pred_index])[0]
            
            # Buang file temporary
            os.unlink(tmp_path)
            
            return pred_label, probs
            
        except Exception as e:
            os.unlink(tmp_path)
            raise e
            
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None

# Main application
def main():
    # Load model
    with st.spinner("Memuat model..."):
        model, support_set, support_labels, le_y = load_models()
    
    if model is None:
        st.error("Gagal memuat model. Pastikan file model tersedia.")
        return
    
    # Upload file audio
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Unggah file audio (.wav)", 
        type=['wav'],
        help="Unggah file audio dalam format WAV"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Tampilkan informasi file
        col1, col2 = st.columns(2)
        
        with col1:
            st.audio(uploaded_file, format='audio/wav')
        
        with col2:
            file_details = {
                "Nama file": uploaded_file.name,
                "Ukuran file": f"{uploaded_file.size / 1024:.2f} KB",
                "Tipe file": uploaded_file.type
            }
            st.write("**Detail File:**")
            for key, value in file_details.items():
                st.write(f"{key}: {value}")
        
        # Tombol untuk prediksi
        if st.button("🔍 Deteksi Aksen", type="primary", use_container_width=True):
            with st.spinner("Memproses audio..."):
                # Prediksi aksen
                pred_label, probs = predict_accent(
                    uploaded_file, 
                    model, 
                    support_set, 
                    support_labels, 
                    le_y
                )
                
                if pred_label is not None:
                    # Tampilkan hasil
                    st.markdown('<div class="result-box">', unsafe_allow_html=True)
                    st.success("✅ **Hasil Deteksi:**")
                    
                    # Tampilkan aksen terprediksi
                    accent_colors = {
                        'Betawi': '#FF6B6B',
                        'Jawa _Timur': '#4ECDC4',
                        'Jawa_Tengah': '#45B7D1',
                        'Sunda': '#96CEB4',
                        'YogyaKarta': '#FFEAA7'
                    }
                    
                    color = accent_colors.get(pred_label, '#3B82F6')
                    st.markdown(f"""
                    <div style="padding: 15px; border-radius: 10px; background-color: {color}20; border-left: 5px solid {color};">
                        <h3 style="color: {color}; margin: 0;">{pred_label}</h3>
                        <p style="margin: 5px 0 0 0; color: #4B5563;">
                            Confidence: <strong>{probs[np.argmax(probs)]*100:.2f}%</strong>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Tampilkan semua probabilitas
                    st.markdown("---")
                    st.markdown("**Detail Probabilitas:**")
                    
                    # Urutkan berdasarkan probabilitas
                    sorted_indices = np.argsort(probs)[::-1]
                    
                    for idx in sorted_indices:
                        accent = le_y.classes_[idx]
                        prob = probs[idx] * 100
                        
                        col_prob, col_bar, col_percent = st.columns([2, 5, 1])
                        with col_prob:
                            st.write(f"{accent}")
                        with col_bar:
                            st.progress(int(prob))
                        with col_percent:
                            st.write(f"{prob:.1f}%")
                    
                    # Informasi tambahan berdasarkan aksen
                    st.markdown("---")
                    st.markdown("**Informasi Aksen:**")
                    
                    accent_info = {
                        'Betawi': 'Aksen khas dari DKI Jakarta dengan pengaruh Melayu dan bahasa daerah lainnya.',
                        'Jawa _Timur': 'Aksen dari Jawa Timur dengan ciri khas logat yang tegas dan sedikit kasar.',
                        'Jawa_Tengah': 'Aksen dari Jawa Tengah dengan logat yang lembut dan halus.',
                        'Sunda': 'Aksen dari Jawa Barat dengan intonasi yang khas dan melodius.',
                        'YogyaKarta': 'Aksen dari Daerah Istimewa Yogyakarta dengan pengaruh budaya keraton.'
                    }
                    
                    info = accent_info.get(pred_label, "Informasi detail tentang aksen ini sedang dalam pengembangan.")
                    st.info(f"**{pred_label}:** {info}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Tombol untuk prediksi lagi
                    if st.button("🔄 Prediksi Audio Lain", use_container_width=True):
                        st.rerun()
    
    # Bagian contoh jika tidak ada file diupload
    else:
        st.markdown("---")
        st.markdown("### 📝 Contoh Penggunaan")
        
        example_col1, example_col2, example_col3 = st.columns(3)
        
        with example_col1:
            st.markdown("**1. Siapkan Audio**")
            st.markdown("- Rekam suara berbicara")
            st.markdown("- Format: .wav")
            st.markdown("- Durasi: 1-5 detik")
        
        with example_col2:
            st.markdown("**2. Unggah File**")
            st.markdown("- Klik area upload")
            st.markdown("- Pilih file audio")
            st.markdown("- Pastikan format benar")
        
        with example_col3:
            st.markdown("**3. Dapatkan Hasil**")
            st.markdown("- Klik tombol deteksi")
            st.markdown("- Tunggu proses")
            st.markdown("- Lihat hasil aksen")

if __name__ == "__main__":
    main()
