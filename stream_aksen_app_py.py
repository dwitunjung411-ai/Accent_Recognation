import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tempfile
import os
import tensorflow as tf
import pickle
import random

# ==========================================================
# CLASS PROTOTYPICAL NETWORK (KOMPATIBEL SEMUA VERSI TF)
# ==========================================================
@tf.keras.utils.register_keras_serializable(package="Custom")
class PrototypicalNetwork(tf.keras.Model):
    def __init__(self, embedding_model=None, **kwargs):
        super(PrototypicalNetwork, self).__init__(**kwargs)
        if embedding_model is not None and not isinstance(embedding_model, dict):
            self._embedding_model = embedding_model
        else:
            self._embedding_model = None
    
    def call(self, support_set, query_set, support_labels, n_way, training=None):
        """
        Args:
            support_set: (n_support, height, width, channels)
            query_set: (n_query, height, width, channels)
            support_labels: (n_support,)
            n_way: jumlah kelas
        Returns:
            logits: (n_query, n_way)
        """
        # Dapatkan embedding layer
        embedding_layer = self._get_embedding_layer()
        
        if embedding_layer is None:
            raise ValueError("Embedding layer tidak tersedia!")
        
        # Embedding untuk support dan query
        support_embeddings = embedding_layer(support_set, training=training)
        query_embeddings = embedding_layer(query_set, training=training)
        
        # Hitung prototype untuk setiap kelas
        prototypes = []
        for i in range(n_way):
            class_mask = tf.equal(support_labels, i)
            class_embeddings = tf.boolean_mask(support_embeddings, class_mask)
            prototype = tf.reduce_mean(class_embeddings, axis=0)
            prototypes.append(prototype)
        
        prototypes = tf.stack(prototypes)
        
        # Hitung jarak euclidean
        query_expanded = tf.expand_dims(query_embeddings, 1)
        prototypes_expanded = tf.expand_dims(prototypes, 0)
        
        distances = tf.reduce_sum(
            tf.square(query_expanded - prototypes_expanded),
            axis=-1
        )
        
        logits = -distances
        return logits
    
    def _get_embedding_layer(self):
        """Ekstrak embedding layer dengan berbagai metode"""
        # 1. Cek _embedding_model
        if hasattr(self, '_embedding_model') and self._embedding_model is not None:
            if callable(self._embedding_model):
                return self._embedding_model
        
        # 2. Cek embedding (tanpa underscore)
        if hasattr(self, 'embedding') and self.embedding is not None:
            if callable(self.embedding):
                return self.embedding
        
        # 3. Cek di self.layers
        if hasattr(self, 'layers') and len(self.layers) > 0:
            for layer in self.layers:
                if isinstance(layer, (tf.keras.Model, tf.keras.Sequential)):
                    return layer
        
        # 4. Cek di __dict__
        for key, value in self.__dict__.items():
            if isinstance(value, (tf.keras.Model, tf.keras.Sequential)):
                if value != self:  # Jangan ambil diri sendiri
                    return value
        
        # 5. Cek _layers (internal)
        if hasattr(self, '_layers') and len(self._layers) > 0:
            for layer in self._layers:
                if isinstance(layer, (tf.keras.Model, tf.keras.Sequential)):
                    return layer
        
        return None
    
    def get_config(self):
        config = super().get_config()
        return config

# ==========================================================
# FUNGSI HELPER
# ==========================================================
def extract_mfcc(audio_path, sr=22050, n_mfcc=40, max_len=174):
    """Extract MFCC 3-channel seperti di notebook"""
    try:
        y, sr = librosa.load(audio_path, sr=sr, duration=10)
        
        # MFCC utama
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        
        # Delta
        mfcc_delta = librosa.feature.delta(mfcc)
        
        # Delta-delta
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Padding/truncating
        def pad_or_truncate(arr, max_len):
            if arr.shape[1] < max_len:
                pad_width = max_len - arr.shape[1]
                arr = np.pad(arr, ((0, 0), (0, pad_width)), mode='constant')
            else:
                arr = arr[:, :max_len]
            return arr
        
        mfcc = pad_or_truncate(mfcc, max_len)
        mfcc_delta = pad_or_truncate(mfcc_delta, max_len)
        mfcc_delta2 = pad_or_truncate(mfcc_delta2, max_len)
        
        # Stack menjadi 3 channels: (n_mfcc, max_len, 3)
        mfcc_3channel = np.stack([mfcc, mfcc_delta, mfcc_delta2], axis=-1)
        
        return mfcc_3channel.astype(np.float32)
        
    except Exception as e:
        st.error(f"Error extracting MFCC: {e}")
        return None

# ==========================================================
# LOAD RESOURCES
# ==========================================================
@st.cache_resource
def load_model_and_support():
    """Load model dan support set"""
    try:
        # Load model
        custom_objects = {"PrototypicalNetwork": PrototypicalNetwork}
        model = tf.keras.models.load_model(
            "model_embedding_aksen.keras",
            custom_objects=custom_objects,
            compile=False
        )
        
        st.sidebar.info(f"📦 Model type: {type(model).__name__}")
        
        # Debug: cek struktur model
        st.sidebar.write("🔍 Model attributes:")
        model_attrs = [attr for attr in dir(model) if not attr.startswith('_')]
        st.sidebar.text(f"Total: {len(model_attrs)}")
        
        # Cek apakah ada embedding
        if hasattr(model, 'embedding'):
            st.sidebar.success("✅ Found: model.embedding")
        if hasattr(model, '_embedding_model'):
            st.sidebar.success("✅ Found: model._embedding_model")
        if hasattr(model, 'layers'):
            st.sidebar.info(f"✅ Found: {len(model.layers)} layers")
        
        # Load support set
        support_set = np.load('support_set.npy')
        support_labels = np.load('support_labels.npy')
        
        st.sidebar.success(f"✅ Support Set: {support_set.shape}")
        st.sidebar.success(f"✅ Support Labels: {support_labels.shape}")
        
        return model, support_set, support_labels
        
    except FileNotFoundError as e:
        st.sidebar.error(f"❌ File tidak ditemukan: {str(e)}")
        st.sidebar.info("💡 Pastikan file support_set.npy dan support_labels.npy ada!")
        return None, None, None
    except Exception as e:
        st.sidebar.error(f"❌ Error loading: {str(e)}")
        import traceback
        st.sidebar.code(traceback.format_exc())
        return None, None, None

@st.cache_data
def load_metadata():
    """Load metadata CSV"""
    if os.path.exists("metadata.csv"):
        return pd.read_csv("metadata.csv")
    return None

@st.cache_data
def load_label_encoder():
    """Load label encoder"""
    try:
        with open('label_encoder.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        le.classes_ = np.array(['Betawi', 'Jawa Tengah', 'Jawa Timur', 'Sunda', 'Yogyakarta'])
        return le

# ==========================================================
# PREDIKSI
# ==========================================================
def predict_accent(audio_path, model, support_set, support_labels, n_way, label_encoder):
    """Prediksi aksen menggunakan Prototypical Network"""
    try:
        # 1. Extract MFCC dari audio query
        mfcc_feat = extract_mfcc(audio_path)
        
        if mfcc_feat is None:
            return "❌ Error extracting features"
        
        st.info(f"✅ MFCC shape: {mfcc_feat.shape}")
        
        # 2. Expand batch dimension
        query_features = np.expand_dims(mfcc_feat, axis=0).astype(np.float32)
        st.info(f"✅ Query shape: {query_features.shape}")
        st.info(f"✅ Support shape: {support_set.shape}")
        
        # 3. Convert to tensors
        support_tensor = tf.convert_to_tensor(support_set, dtype=tf.float32)
        query_tensor = tf.convert_to_tensor(query_features, dtype=tf.float32)
        support_labels_tensor = tf.convert_to_tensor(support_labels, dtype=tf.int32)
        
        # 4. Forward pass
        st.info("🔄 Calling model...")
        logits = model.call(
            support_tensor,
            query_tensor,
            support_labels_tensor,
            n_way
        )
        
        st.success(f"✅ Logits shape: {logits.shape}")
        
        # 5. Get prediction
        pred_index = tf.argmax(logits, axis=1).numpy()[0]
        probs = tf.nn.softmax(logits, axis=1).numpy()[0]
        
        # 6. Convert to label
        pred_label = label_encoder.inverse_transform([pred_index])[0]
        confidence = probs[pred_index] * 100
        
        # 7. Detail probabilitas
        detail_lines = []
        for i, (cls, prob) in enumerate(zip(label_encoder.classes_, probs)):
            marker = "👉 " if i == pred_index else "   "
            detail_lines.append(f"{marker}{cls}: {prob*100:.2f}%")
        
        detail = "\n".join(detail_lines)
        
        result = f"{pred_label} ({confidence:.1f}%)\n\n📊 Detail Probabilitas:\n{detail}"
        
        return result
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        st.error("❌ Full Error:")
        st.code(error_detail)
        return f"❌ Error: {str(e)}"

# ==========================================================
# STREAMLIT UI
# ==========================================================
st.set_page_config(
    page_title="Deteksi Aksen Indonesia",
    page_icon="🎙️",
    layout="wide"
)

st.title("🎙️ Sistem Deteksi Aksen Indonesia (Few-Shot Learning)")
st.write("Aplikasi berbasis *Prototypical Network* untuk klasifikasi aksen daerah.")
st.divider()

# Load resources
model, support_set, support_labels = load_model_and_support()
metadata = load_metadata()
label_encoder = load_label_encoder()

# Sidebar info
with st.sidebar:
    st.header("🛸 Status Sistem")
    
    if metadata is not None:
        st.info(f"📁 Metadata: {len(metadata)} records")
    
    st.divider()
    st.caption("🎓 Few-Shot Learning - 2026")

# Main layout
col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("📥 Input Audio")
    
    audio_file = st.file_uploader(
        "Upload file audio (.wav, .mp3)",
        type=["wav", "mp3"]
    )
    
    if audio_file:
        st.audio(audio_file)
        
        if st.button("🚀 Analisis Aksen", type="primary", use_container_width=True):
            if model is not None and support_set is not None:
                with st.spinner("🔍 Menganalisis karakteristik suara..."):
                    # Save temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(audio_file.getbuffer())
                        tmp_path = tmp.name
                    
                    # Predict
                    n_way = len(label_encoder.classes_)
                    hasil = predict_accent(
                        tmp_path, 
                        model, 
                        support_set, 
                        support_labels, 
                        n_way, 
                        label_encoder
                    )
                    
                    # Get metadata
                    user_info = None
                    if metadata is not None:
                        match = metadata[metadata['file_name'] == audio_file.name]
                        if not match.empty:
                            user_info = match.iloc[0].to_dict()
                    
                    # Display results
                    with col2:
                        st.subheader("📊 Hasil Analisis")
                        
                        with st.container(border=True):
                            st.markdown("#### 🎭 Aksen Terdeteksi:")
                            if "❌" in hasil:
                                st.error(hasil)
                            else:
                                st.text(hasil)
                        
                        st.divider()
                        
                        st.subheader("💎 Info Pembicara (dari Metadata)")
                        if user_info:
                            province_to_accent = {
                                'DKI Jakarta': 'Betawi',
                                'Jawa Barat': 'Sunda',
                                'Jawa Tengah': 'Jawa Tengah',
                                'Jawa Timur': 'Jawa Timur',
                                'Yogyakarta': 'Yogyakarta',
                                'D.I YogyaKarta': 'Yogyakarta'
                            }
                            
                            actual_province = user_info.get('provinsi', '-')
                            actual_accent = province_to_accent.get(actual_province, '-')
                            
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("🎂 Usia", f"{user_info.get('usia', '-')} Tahun")
                                st.metric("🚻 Gender", user_info.get('gender', '-'))
                            with col_b:
                                st.metric("🗺️ Provinsi", actual_province)
                                st.metric("✅ Aksen Sebenarnya", actual_accent)
                            
                            # Check accuracy
                            if actual_accent != '-':
                                predicted_accent = hasil.split('(')[0].strip()
                                if actual_accent == predicted_accent:
                                    st.success("🎯 Prediksi BENAR!")
                                else:
                                    st.warning(f"⚠️ Prediksi tidak sesuai! Seharusnya: **{actual_accent}**")
                        else:
                            st.info("🕵️ File tidak terdaftar dalam metadata")
                    
                    # Cleanup
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
            else:
                st.error("⚠️ Model atau Support Set tidak tersedia")
    else:
        with col2:
            st.info("👈 Upload file audio untuk memulai analisis")

# Info tambahan
with st.expander("ℹ️ Tentang Few-Shot Learning"):
    st.write("""
    **Prototypical Network** adalah metode Few-Shot Learning yang:
    - Menggunakan **Support Set** sebagai referensi untuk setiap kelas
    - Menghitung **prototype** (centroid) dari embedding setiap kelas
    - Mengklasifikasikan query berdasarkan jarak ke prototype terdekat
    
    **File yang diperlukan:**
    - `model_embedding_aksen.keras` - Model yang sudah di-training
    - `support_set.npy` - Support set dari notebook (cell 43)
    - `support_labels.npy` - Label support set dari notebook (cell 43)
    - `metadata.csv` - Metadata pembicara (optional)
    """)
