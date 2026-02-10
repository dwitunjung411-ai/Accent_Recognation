import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tempfile
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

# ==========================================================
# 1. DEFINISI CLASS PROTOTYPICAL NETWORK YANG BENAR
# ==========================================================
@tf.keras.utils.register_keras_serializable(package="Custom")
class PrototypicalNetwork(tf.keras.Model):
    def __init__(self, encoder, **kwargs):
        super(PrototypicalNetwork, self).__init__(**kwargs)
        self.encoder = encoder  # Ini seharusnya model encoder
    
    def call(self, inputs):
        # Untuk prediksi inference, cukup encode input
        return self.encoder(inputs)
    
    def compute_prototypes(self, support_set, support_labels):
        """Menghitung prototype untuk setiap kelas"""
        # Encode support set
        support_embeddings = self.encoder(support_set)
        
        # Kelompokkan berdasarkan label dan hitung mean
        prototypes = []
        unique_labels = tf.unique(support_labels)[0]
        
        for label in unique_labels:
            mask = tf.equal(support_labels, label)
            class_embeddings = tf.boolean_mask(support_embeddings, mask)
            prototype = tf.reduce_mean(class_embeddings, axis=0)
            prototypes.append(prototype)
        
        return tf.stack(prototypes), unique_labels
    
    def euclidean_distance(self, query_embeddings, prototypes):
        """Menghitung Euclidean distance"""
        # query_embeddings: [n_queries, embedding_dim]
        # prototypes: [n_classes, embedding_dim]
        
        # Expand dimensions untuk broadcasting
        query_exp = tf.expand_dims(query_embeddings, 1)  # [n_queries, 1, embedding_dim]
        proto_exp = tf.expand_dims(prototypes, 0)        # [1, n_classes, embedding_dim]
        
        # Hitung Euclidean distance
        distances = tf.reduce_sum(tf.square(query_exp - proto_exp), axis=2)
        return distances
    
    def predict_on_batch(self, support_set, support_labels, query_set):
        """Prediksi untuk prototypical network"""
        # Hitung prototypes
        prototypes, class_labels = self.compute_prototypes(support_set, support_labels)
        
        # Encode query set
        query_embeddings = self.encoder(query_set)
        
        # Hitung distances
        distances = self.euclidean_distance(query_embeddings, prototypes)
        
        # Convert distances ke probabilities (softmax over negative distances)
        probabilities = tf.nn.softmax(-distances)
        
        return probabilities, class_labels, distances
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "encoder": tf.keras.layers.serialize(self.encoder)
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        encoder = tf.keras.layers.deserialize(config.pop("encoder"))
        return cls(encoder, **config)


# ==========================================================
# 2. FUNGSI LOAD DATA & MODEL
# ==========================================================
@st.cache_resource
def load_accent_model():
    model_name = "model_detect_aksen.keras"
    current_dir = os.getcwd()
    model_path = os.path.join(current_dir, model_name)
    
    if os.path.exists(model_path):
        try:
            custom_objects = {"PrototypicalNetwork": PrototypicalNetwork}
            model = tf.keras.models.load_model(
                model_path, 
                custom_objects=custom_objects, 
                compile=False
            )
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    else:
        st.error(f"Model file not found at: {model_path}")
        return None


@st.cache_data
def load_metadata_df():
    csv_path = "metadata.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return df
    return None


@st.cache_data
def load_support_set():
    """Load contoh audio untuk setiap aksen (support set)"""
    # Anda perlu menyiapkan beberapa contoh audio untuk setiap aksen
    # atau menggunakan data dari metadata
    support_data = []
    support_labels = []
    
    # Contoh path - sesuaikan dengan struktur data Anda
    aksen_folders = {
        "Sunda": "data/train/sunda",
        "Jawa Tengah": "data/train/jawa_tengah", 
        "Jawa Timur": "data/train/jawa_timur",
        "Yogyakarta": "data/train/yogyakarta",
        "Betawi": "data/train/betawi"
    }
    
    for label, folder in aksen_folders.items():
        if os.path.exists(folder):
            # Ambil 5 file pertama sebagai support set
            files = [f for f in os.listdir(folder) if f.endswith('.wav')][:5]
            for file in files:
                file_path = os.path.join(folder, file)
                try:
                    features = extract_features(file_path)
                    if features is not None:
                        support_data.append(features)
                        support_labels.append(label)
                except:
                    continue
    
    if len(support_data) > 0:
        support_data = np.array(support_data)
        # Encode labels ke angka
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        support_labels_encoded = le.fit_transform(support_labels)
        return support_data, support_labels_encoded, le
    return None, None, None


# ==========================================================
# 3. FUNGSI EXTRACT FEATURES (SESUAI TRAINING)
# ==========================================================
def extract_features(audio_path, sr=16000, n_mfcc=40, max_frames=100):
    """Ekstrak fitur MFCC sama seperti saat training"""
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=sr)
        
        # Normalize
        y = y / np.max(np.abs(y))
        
        # Ekstrak MFCC
        mfcc = librosa.feature.mfcc(
            y=y, 
            sr=sr, 
            n_mfcc=n_mfcc,
            n_fft=2048,
            hop_length=512,
            fmin=0,
            fmax=sr/2
        )
        
        # Delta dan Delta-Delta
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Gabungkan
        mfcc_features = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
        
        # Padding atau truncation
        if mfcc_features.shape[1] > max_frames:
            mfcc_features = mfcc_features[:, :max_frames]
        else:
            pad_width = max_frames - mfcc_features.shape[1]
            mfcc_features = np.pad(
                mfcc_features, 
                pad_width=((0, 0), (0, pad_width)), 
                mode='constant'
            )
        
        # Normalize per fitur
        mfcc_features = (mfcc_features - np.mean(mfcc_features)) / np.std(mfcc_features)
        
        # Reshape untuk model [frames, features, channels]
        mfcc_features = np.transpose(mfcc_features)  # [max_frames, n_features]
        mfcc_features = np.expand_dims(mfcc_features, axis=-1)  # [max_frames, n_features, 1]
        
        return mfcc_features
    
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None


# ==========================================================
# 4. FUNGSI PREDIKSI PROTOTYPICAL NETWORK
# ==========================================================
def predict_accent_prototypical(audio_path, model, support_data, support_labels, label_encoder):
    """Prediksi menggunakan Prototypical Network"""
    try:
        # Extract features dari query audio
        query_features = extract_features(audio_path)
        if query_features is None:
            return "Error: Gagal ekstrak fitur", None, None
        
        # Reshape untuk batch
        query_batch = np.expand_dims(query_features, axis=0)  # [1, frames, features, 1]
        
        # Jika model adalah PrototypicalNetwork
        if isinstance(model, PrototypicalNetwork):
            # Gunakan method khusus untuk prototypical
            probabilities, class_indices, distances = model.predict_on_batch(
                support_set=support_data,
                support_labels=support_labels,
                query_set=query_batch
            )
            
            # Decode label
            predicted_idx = np.argmax(probabilities[0])
            predicted_label = label_encoder.inverse_transform([class_indices[predicted_idx]])[0]
            confidence = float(probabilities[0][predicted_idx] * 100)
            
            # Get all probabilities
            all_probs = {}
            for idx, class_idx in enumerate(class_indices):
                label = label_encoder.inverse_transform([class_idx])[0]
                all_probs[label] = float(probabilities[0][idx] * 100)
            
            return predicted_label, confidence, all_probs
            
        else:
            # Jika model biasa (bukan PrototypicalNetwork)
            prediction = model.predict(query_batch, verbose=0)
            aksen_classes = ["Sunda", "Jawa Tengah", "Jawa Timur", "Yogyakarta", "Betawi"]
            predicted_idx = np.argmax(prediction[0])
            predicted_label = aksen_classes[predicted_idx]
            confidence = float(prediction[0][predicted_idx] * 100)
            
            # Get all probabilities
            all_probs = {}
            for i, label in enumerate(aksen_classes):
                all_probs[label] = float(prediction[0][i] * 100)
            
            return predicted_label, confidence, all_probs
            
    except Exception as e:
        return f"Error: {str(e)}", None, None


# ==========================================================
# 5. MAIN UI
# ==========================================================
def main():
    # Set page config
    st.set_page_config(
        page_title="Deteksi Aksen Prototypical Network", 
        page_icon="🎙️", 
        layout="wide"
    )
    
    # Title
    st.title("🎙️ Sistem Deteksi Aksen Prototypical Network")
    st.markdown("""
    Aplikasi berbasis **Few-Shot Learning** untuk klasifikasi aksen daerah Indonesia.
    Menggunakan **Prototypical Networks** yang efektif untuk data terbatas.
    """)
    st.divider()
    
    # Load model dan data
    with st.spinner("Memuat model dan data..."):
        model_aksen = load_accent_model()
        df_metadata = load_metadata_df()
        
        # Load support set untuk prototypical network
        support_data, support_labels, label_encoder = load_support_set()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Status Sistem")
        
        # Model status
        if model_aksen:
            st.success("✅ Model: Terhubung")
            # Tampilkan info model
            model_type = "PrototypicalNetwork" if isinstance(model_aksen, PrototypicalNetwork) else "Standard Model"
            st.info(f"**Tipe Model:** {model_type}")
        else:
            st.error("❌ Model: Tidak ditemukan")
        
        # Metadata status
        if df_metadata is not None:
            st.success(f"✅ Metadata: {len(df_metadata)} sampel")
        else:
            st.warning("⚠️ Metadata: Tidak ditemukan")
        
        # Support set status
        if support_data is not None:
            st.success(f"✅ Support Set: {len(support_data)} sampel ({len(np.unique(support_labels))} kelas)")
        else:
            st.warning("⚠️ Support Set: Tidak ditemukan (gunakan model standar)")
        
        st.divider()
        st.caption("🎓 Skripsi Project - 2026")
    
    # Main content
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.subheader("📥 Input Audio")
        
        # File uploader
        audio_file = st.file_uploader(
            "Upload file audio (.wav, .mp3, .flac)", 
            type=["wav", "mp3", "flac"],
            help="Durasi optimal: 3-10 detik, sample rate: 16000 Hz"
        )
        
        if audio_file:
            # Display audio
            st.audio(audio_file)
            
            # File info
            file_size = audio_file.size / 1024  # KB
            st.caption(f"**Info File:** {audio_file.name} ({file_size:.1f} KB)")
            
            # Detect button
            if st.button(
                "🔍 Deteksi Aksen", 
                type="primary", 
                use_container_width=True,
                disabled=(model_aksen is None)
            ):
                with st.spinner("🔄 Menganalisis audio..."):
                    # Save temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(audio_file.getbuffer())
                        tmp_path = tmp.name
                    
                    try:
                        # Predict
                        if isinstance(model_aksen, PrototypicalNetwork) and support_data is not None:
                            predicted_label, confidence, all_probs = predict_accent_prototypical(
                                tmp_path, model_aksen, support_data, support_labels, label_encoder
                            )
                        else:
                            # Fallback to standard prediction
                            predicted_label, confidence, all_probs = predict_accent_prototypical(
                                tmp_path, model_aksen, None, None, None
                            )
                        
                        # Display results in col2
                        with col2:
                            st.subheader("📊 Hasil Analisis")
                            
                            # Result card
                            with st.container(border=True):
                                st.markdown(f"### 🎭 Aksen Terdeteksi")
                                st.markdown(f"# **{predicted_label}**")
                                st.metric("Tingkat Kepercayaan", f"{confidence:.1f}%")
                            
                            # Probabilities
                            st.subheader("📈 Distribusi Probabilitas")
                            for label, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
                                # Progress bar dengan warna berbeda
                                if label == predicted_label:
                                    st.progress(prob/100, text=f"**{label}**: {prob:.1f}%")
                                else:
                                    # Progress bar dengan transparency
                                    st.progress(prob/100, text=f"{label}: {prob:.1f}%")
                            
                            # Metadata info
                            if df_metadata is not None and audio_file.name in df_metadata['file_name'].values:
                                st.divider()
                                st.subheader("📋 Informasi Pembicara")
                                
                                user_data = df_metadata[df_metadata['file_name'] == audio_file.name].iloc[0]
                                
                                info_cols = st.columns(3)
                                with info_cols[0]:
                                    st.metric("Usia", f"{user_data.get('usia', '-')}")
                                with info_cols[1]:
                                    st.metric("Gender", user_data.get('gender', '-'))
                                with info_cols[2]:
                                    st.metric("Provinsi", user_data.get('provinsi', '-'))
                                
                                # Additional info jika ada
                                if 'kota' in user_data:
                                    st.info(f"**Kota Asal:** {user_data['kota']}")
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                    finally:
                        # Cleanup temp file
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)
    
    with col2:
        if not audio_file:
            # Placeholder/instructions
            st.subheader("ℹ️ Cara Penggunaan")
            st.info("""
            1. **Upload file audio** di kolom kiri
            2. **Tekan tombol 'Deteksi Aksen'**
            3. **Lihat hasil** di panel ini
            
            **Persyaratan Audio:**
            - Format: WAV, MP3, atau FLAC
            - Durasi: 3-10 detik optimal
            - Sample rate: 16000 Hz direkomendasikan
            - Mono/stereo: Keduanya didukung
            
            **Aksen yang Didukung:**
            - Sunda (Jawa Barat)
            - Jawa Tengah
            - Jawa Timur  
            - Yogyakarta
            - Betawi (Jakarta)
            """)
            
            # Feature visualization placeholder
            st.divider()
            st.subheader("🎯 Teknologi Prototypical Network")
            st.markdown("""
            **Prototypical Networks** adalah metode *few-shot learning* yang:
            - Mampu belajar dari sedikit contoh
            - Membuat *prototype* untuk setiap kelas
            - Mengklasifikasi berdasarkan jarak ke prototype
            
            **Keunggulan:**
            ✓ Efektif untuk data terbatas  
            ✓ Generalisasi lebih baik  
            ✓ Tidak perlu retrain untuk kelas baru
            """)

if __name__ == "__main__":
    main()
