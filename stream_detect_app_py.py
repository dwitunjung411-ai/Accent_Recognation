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
SR = 22050

st.set_page_config(
    page_title="Deteksi Aksen - Few Shot",
    layout="centered"
)

# ===============================
# MODEL DEFINITION (SAMA DENGAN TRAINING)
# ===============================
@tf.keras.utils.register_keras_serializable()
class PrototypicalNetwork(tf.keras.Model):
    def __init__(self, embedding_model=None, **kwargs):
        super().__init__(**kwargs)
        if embedding_model is None:
            # Buat embedding model default jika tidak disediakan
            self.embedding = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(13,)),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(32, activation='relu')
            ])
        else:
            self.embedding = embedding_model
    
    def call(self, inputs, training=False):
        # Model ini hanya melakukan embedding
        return self.embedding(inputs)

# ===============================
# LOAD MODEL & SUPPORT SET
# ===============================
@st.cache_resource
def load_all():
    try:
        # Load model dengan custom objects
        model = tf.keras.models.load_model(
            "model_detect_aksen.keras",  # Ganti dengan nama file yang benar
            custom_objects={"PrototypicalNetwork": PrototypicalNetwork},
            compile=False  # Tidak perlu compile untuk inference
        )
        
        st.success("✅ Model loaded successfully")
        
        # Debug info
        with st.expander("🔍 Model Architecture Details", expanded=True):
            st.write(f"**Model class:** {type(model)}")
            st.write(f"**Input shape:** {model.input_shape}")
            st.write(f"**Output shape:** {model.output_shape}")
            
            # Coba forward pass dengan dummy input
            if model.input_shape[1] is not None:
                input_dim = model.input_shape[1]
                dummy_input = np.random.randn(1, input_dim).astype(np.float32)
                try:
                    dummy_output = model.predict(dummy_input, verbose=0)
                    st.write(f"**Dummy test - Input:** {dummy_input.shape}, **Output:** {dummy_output.shape}")
                except:
                    pass
            
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.code(traceback.format_exc())
        return None, None, None
    
    try:
        # Load support set
        support_set = np.load("support_set.npy")
        support_labels = np.load("support_labels.npy")
        
        st.success(f"✅ Support set loaded: {support_set.shape}")
        st.success(f"✅ Support labels: {support_labels.shape}")
        
        # Validasi kesesuaian dengan model
        if len(support_set.shape) == 2 and model.input_shape[1] is not None:
            if support_set.shape[1] != model.input_shape[1]:
                st.warning(f"⚠️ Dimension mismatch: Support set has {support_set.shape[1]} features, model expects {model.input_shape[1]}")
                
    except Exception as e:
        st.error(f"❌ Error loading support files: {str(e)}")
        return model, None, None
    
    return model, support_set, support_labels

# ===============================
# FEATURE EXTRACTION
# ===============================
def extract_13_features(audio_path, sr=22050):
    """Ekstrak 13 fitur MFCC rata-rata"""
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=sr, duration=3.0)  # Ambil 3 detik pertama
        y = librosa.util.normalize(y)
        
        # Ekstrak 13 MFCC
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=13,
            n_fft=2048,
            hop_length=512
        )
        
        # Ambil rata-rata per koefisien
        features = np.mean(mfcc, axis=1)  # Shape: (13,)
        
        # Pastikan ada 13 fitur
        if len(features) != 13:
            # Jika kurang, pad dengan nol
            features = np.pad(features, (0, 13 - len(features)), 'constant')
        elif len(features) > 13:
            # Jika lebih, ambil 13 pertama
            features = features[:13]
            
        return features.astype(np.float32)
        
    except Exception as e:
        st.error(f"Error extracting features: {str(e)}")
        return np.zeros(13, dtype=np.float32)

# ===============================
# STREAMLIT UI
# ===============================
def main():
    st.title("🎙️ Deteksi Aksen Bahasa Indonesia")
    st.write("Menggunakan Few-Shot Learning dengan Prototypical Network")
    
    # Load model dan data
    with st.spinner("Memuat model dan data..."):
        model, support_set, support_labels = load_all()
    
    if model is None or support_set is None:
        st.error("Gagal memuat model atau data support set.")
        st.stop()
    
    # Hitung prototipe
    with st.spinner("Menghitung prototipe dari support set..."):
        try:
            # Dapatkan embedding untuk support set
            support_embeddings = model.predict(support_set, verbose=0)
            
            # Hitung prototipe per kelas
            prototypes = []
            aksen_classes = ["Sunda", "Jawa Tengah", "Jawa Timur", "Yogyakarta", "Betawi"]
            
            for i in range(N_WAY):
                mask = support_labels == i
                class_emb = support_embeddings[mask]
                if len(class_emb) > 0:
                    proto = np.mean(class_emb, axis=0)
                    prototypes.append(proto)
                else:
                    prototypes.append(np.zeros(support_embeddings.shape[1]))
            
            prototypes = np.array(prototypes)
            st.success(f"✅ Prototipe dihitung: {prototypes.shape}")
            
        except Exception as e:
            st.error(f"Error menghitung prototipe: {str(e)}")
            st.stop()
    
    # UI Upload
    st.divider()
    st.subheader("📤 Upload Audio")
    
    audio_file = st.file_uploader(
        "Pilih file audio (.wav, .mp3, .m4a)",
        type=["wav", "mp3", "m4a"],
        help="Upload rekaman suara untuk dideteksi aksennya"
    )
    
    if audio_file:
        # Tampilkan audio
        st.audio(audio_file, format="audio/wav")
        
        if st.button("🔍 Deteksi Aksen", type="primary", use_container_width=True):
            with st.spinner("Memproses audio..."):
                # Simpan file sementara
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(audio_file.read())
                    audio_path = tmp.name
                
                try:
                    # 1. Ekstrak fitur
                    features = extract_13_features(audio_path)
                    
                    # 2. Reshape untuk model (batch, 13)
                    features_batch = features.reshape(1, -1)
                    
                    # 3. Dapatkan embedding
                    query_embedding = model.predict(features_batch, verbose=0)
                    
                    # 4. Hitung jarak Euclidean ke setiap prototipe
                    distances = np.linalg.norm(
                        query_embedding - prototypes,
                        axis=1
                    )
                    
                    # 5. Konversi ke probabilitas (softmax over negative distances)
                    logits = -distances
                    exp_logits = np.exp(logits - np.max(logits))  # Stabilisasi numerik
                    probs = exp_logits / np.sum(exp_logits)
                    
                    # 6. Ambil prediksi
                    pred_idx = np.argmax(probs)
                    
                    # TAMPILKAN HASIL
                    st.divider()
                    st.subheader("📊 Hasil Deteksi")
                    
                    # Card untuk hasil utama
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.markdown(f"""
                        <div style="text-align: center; padding: 20px; border-radius: 10px; 
                                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                    color: white;">
                            <h2 style="margin: 0;">{aksen_classes[pred_idx]}</h2>
                            <p style="margin: 5px 0 0 0; font-size: 1.2em;">
                                {probs[pred_idx]*100:.1f}% confidence
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Grafik probabilitas
                    st.write("### 📈 Probabilitas per Kelas:")
                    
                    for i, (cls, prob) in enumerate(zip(aksen_classes, probs)):
                        col1, col2, col3 = st.columns([2, 6, 2])
                        with col1:
                            if i == pred_idx:
                                st.markdown(f"**🏆 {cls}**")
                            else:
                                st.write(cls)
                        with col2:
                            st.progress(float(prob))
                        with col3:
                            st.write(f"{prob*100:.1f}%")
                    
                    # Tampilkan fitur yang diekstrak
                    with st.expander("🔧 Detail Teknis"):
                        st.write(f"**Fitur yang diekstrak:** {features.shape}")
                        st.write(f"**Nilai fitur:**")
                        st.code(f"Min: {features.min():.3f}, Max: {features.max():.3f}, Mean: {features.mean():.3f}")
                        
                        st.write("**Jarak ke prototipe:**")
                        for i, (cls, dist) in enumerate(zip(aksen_classes, distances)):
                            st.write(f"{cls}: {dist:.3f}")
                            
                except Exception as e:
                    st.error(f"❌ Error selama inference: {str(e)}")
                    st.code(traceback.format_exc())
                    
                finally:
                    # Bersihkan file temp
                    os.unlink(audio_path)

if __name__ == "__main__":
    main()
