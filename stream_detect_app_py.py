import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import tempfile
import os
import traceback

# ===============================
# CONFIG
# ===============================
N_WAY = 5
K_SHOT = 3  # Sesuaikan dengan training
SR = 22050

st.set_page_config(
    page_title="Deteksi Aksen Bahasa Indonesia",
    layout="centered"
)

# ===============================
# MODEL DEFINITION (HARUS SAMA DENGAN TRAINING)
# ===============================
@tf.keras.utils.register_keras_serializable()
class PrototypicalNetwork(tf.keras.Model):
    def __init__(self, embedding_model=None, **kwargs):
        super().__init__(**kwargs)
        self.embedding = embedding_model

    def call(self, inputs):
        # Untuk inference sederhana, kita hanya perlu embedding
        return self.embedding(inputs)

# ===============================
# LOAD MODEL & SUPPORT SET
# ===============================
@st.cache_resource
def load_all():
    """Load semua model dan data yang diperlukan"""
    try:
        # Load embedding model (lebih kecil dan mudah)
        embedding_model = tf.keras.models.load_model(
            "embedding_model.keras",
            compile=False
        )
        st.success("✅ Embedding model loaded")
        
        # Buat prototypical network dengan embedding model
        model = PrototypicalNetwork(embedding_model)
        model.build(input_shape=(None, 40, 174, 6))  # Sesuaikan dengan input shape Anda
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        # Fallback: buat model sederhana
        st.info("ℹ️ Creating simple embedding model...")
        embedding_model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(40, 174, 6)),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='linear')
        ])
        model = PrototypicalNetwork(embedding_model)
    
    try:
        # Load support set
        support_set = np.load("support_set.npy")
        support_labels = np.load("support_labels.npy")
        st.success(f"✅ Support set loaded: {support_set.shape}")
        
    except FileNotFoundError:
        st.error("❌ Support set files not found!")
        st.info("""
        ℹ️ Anda perlu:
        1. Jalankan bagian 'create_fixed_support_set' di training script
        2. Download support_set.npy dan support_labels.npy
        3. Upload ke folder Streamlit
        """)
        return None, None, None
    
    return model, support_set, support_labels

# ===============================
# FEATURE EXTRACTION (SAMA DENGAN TRAINING)
# ===============================
def extract_mfcc_metadata(audio_path, metadata_dict=None, sr=22050, n_mfcc=40, max_len=174):
    """
    Ekstrak MFCC + metadata seperti di training script
    metadata_dict: {'usia': 70, 'gender': 'perempuan', 'provinsi': 'DKI Jakarta'}
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=sr, duration=3.0)
        y = librosa.util.normalize(y)
        
        # Ekstrak MFCC
        mfcc = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=n_mfcc, n_fft=2048, hop_length=512
        )
        
        # Delta dan Delta-Delta
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Padding atau truncating
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
        audio_features = np.stack([mfcc, delta, delta2], axis=-1)  # (40, 174, 3)
        
        # Jika ada metadata, tambahkan (sesuai dengan training script Anda)
        if metadata_dict:
            # Anda perlu menyesuaikan ini dengan preprocessing metadata di training
            # Ini contoh sederhana
            metadata_features = np.zeros((40, 174, 3))  # Placeholder
            features = np.concatenate([audio_features, metadata_features], axis=-1)
        else:
            features = audio_features
        
        # Add batch dimension
        features = features[np.newaxis, ...]  # (1, 40, 174, 3 atau 6)
        
        return features.astype(np.float32)
        
    except Exception as e:
        st.error(f"Error extracting features: {str(e)}")
        # Return zeros sebagai fallback
        return np.zeros((1, 40, 174, 3), dtype=np.float32)

# ===============================
# PROTOTYPE COMPUTATION
# ===============================
@st.cache_data
def compute_prototypes(_model, _support_set, _support_labels, n_way):
    """Hitung prototipe dari support set"""
    # Dapatkan embeddings untuk support set
    support_embeddings = _model.predict(_support_set, verbose=0)
    
    prototypes = []
    for i in range(n_way):
        mask = _support_labels == i
        class_emb = support_embeddings[mask]
        if len(class_emb) > 0:
            prototype = np.mean(class_emb, axis=0)
        else:
            prototype = np.zeros(support_embeddings.shape[1])
        prototypes.append(prototype)
    
    return np.array(prototypes)

# ===============================
# STREAMLIT UI
# ===============================
def main():
    st.title("🎙️ Deteksi Aksen Bahasa Indonesia")
    st.write("Menggunakan Prototypical Network (Few-Shot Learning)")
    
    # Load model dan data
    with st.spinner("Memuat model dan data..."):
        model, support_set, support_labels = load_all()
    
    if model is None:
        st.error("Tidak dapat memuat model. Pastikan file-file sudah diupload.")
        st.stop()
    
    # Hitung prototipe
    with st.spinner("Menghitung prototipe..."):
        prototypes = compute_prototypes(model, support_set, support_labels, N_WAY)
        st.success(f"✅ Prototipe dihitung: {prototypes.shape}")
    
    # Sidebar untuk metadata (opsional)
    with st.sidebar:
        st.header("📋 Metadata (Opsional)")
        usia = st.number_input("Usia", min_value=10, max_value=100, value=25)
        gender = st.selectbox("Gender", ["perempuan", "laki-laki"])
        provinsi = st.selectbox("Provinsi", [
            "DKI Jakarta", "Jawa Barat", "Jawa Tengah", 
            "Jawa Timur", "Yogyakarta"
        ])
        
        metadata_dict = {
            'usia': usia,
            'gender': gender,
            'provinsi': provinsi
        }
    
    # Main content
    st.divider()
    st.subheader("📤 Upload Audio")
    
    audio_file = st.file_uploader(
        "Pilih file audio (.wav, .mp3)",
        type=["wav", "mp3"],
        help="File audio akan diproses untuk deteksi aksen"
    )
    
    if audio_file:
        # Tampilkan audio
        st.audio(audio_file)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔍 Deteksi Aksen", type="primary", use_container_width=True):
                with st.spinner("Memproses audio..."):
                    # Simpan file sementara
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(audio_file.read())
                        audio_path = tmp.name
                    
                    try:
                        # 1. Ekstrak fitur
                        features = extract_mfcc_metadata(audio_path, metadata_dict)
                        
                        # 2. Dapatkan embedding query
                        query_embedding = model.predict(features, verbose=0)
                        
                        # 3. Hitung jarak Euclidean ke prototipe
                        distances = np.linalg.norm(
                            query_embedding[:, np.newaxis, :] - prototypes[np.newaxis, :, :],
                            axis=2
                        )
                        
                        # 4. Konversi ke probabilitas (softmax over negative distances)
                        logits = -distances
                        exp_logits = np.exp(logits - np.max(logits))
                        probs = exp_logits / np.sum(exp_logits)
                        
                        # 5. Ambil prediksi
                        pred_idx = np.argmax(probs[0])
                        
                        # Kelas aksen
                        aksen_classes = [
                            "Sunda (Jawa Barat)",
                            "Jawa Tengah", 
                            "Jawa Timur",
                            "Yogyakarta",
                            "Betawi (Jakarta)"
                        ]
                        
                        # TAMPILKAN HASIL
                        st.divider()
                        st.subheader("📊 Hasil Deteksi")
                        
                        # Hasil utama
                        confidence = probs[0][pred_idx] * 100
                        if confidence > 70:
                            st.success(f"🎯 **Aksen Terdeteksi: {aksen_classes[pred_idx]}**")
                        elif confidence > 50:
                            st.warning(f"⚠️ **Aksen Terdeteksi: {aksen_classes[pred_idx]}**")
                        else:
                            st.info(f"💡 **Aksen Terdeteksi: {aksen_classes[pred_idx]}**")
                        
                        st.write(f"**Confidence: {confidence:.1f}%**")
                        
                        # Grafik probabilitas
                        st.write("### 📈 Probabilitas per Kelas:")
                        
                        for i, (cls, prob) in enumerate(zip(aksen_classes, probs[0])):
                            cols = st.columns([3, 5, 2])
                            with cols[0]:
                                if i == pred_idx:
                                    st.markdown(f"**🏆 {cls}**")
                                else:
                                    st.write(cls)
                            with cols[1]:
                                st.progress(float(prob))
                            with cols[2]:
                                st.write(f"{prob*100:.1f}%")
                        
                        # Detail teknis
                        with st.expander("🔧 Detail Teknis"):
                            st.write(f"**Shape fitur:** {features.shape}")
                            st.write(f"**Shape embedding:** {query_embedding.shape}")
                            st.write("**Jarak ke prototipe:**")
                            for i, (cls, dist) in enumerate(zip(aksen_classes, distances[0])):
                                st.write(f"{cls}: {dist:.3f}")
                                
                    except Exception as e:
                        st.error(f"❌ Error selama pemrosesan: {str(e)}")
                        st.code(traceback.format_exc())
                        
                    finally:
                        # Bersihkan file temp
                        if os.path.exists(audio_path):
                            os.unlink(audio_path)
        
        with col2:
            if st.button("🔄 Reset", use_container_width=True):
                st.rerun()
    
    # Penjelasan sistem
    st.divider()
    with st.expander("ℹ️ Tentang Sistem Ini"):
        st.write("""
        **Cara Kerja:**
        1. Sistem mengekstrak fitur MFCC dari audio
        2. Fitur dipetakan ke embedding space menggunakan neural network
        3. Dihitung jarak ke prototipe setiap kelas aksen
        4. Kelas dengan jarak terdekat dipilih sebagai prediksi
        
        **Prototipe** adalah rata-rata dari beberapa sampel setiap aksen.
        
        **Teknologi:** Few-Shot Learning dengan Prototypical Network
        """)

if __name__ == "__main__":
    main()
