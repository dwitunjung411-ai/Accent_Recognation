import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import tempfile
import os

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
# MODEL DEFINITION
# ===============================
@tf.keras.utils.register_keras_serializable()
class PrototypicalNetwork(tf.keras.Model):
    def __init__(self, embedding_model=None, **kwargs):
        super().__init__(**kwargs)
        self.embedding = embedding_model

    def call(self, inputs, training=False):
        # Jika model hanya menerima query set saja
        return self.embedding(inputs)

# ===============================
# LOAD MODEL & SUPPORT SET
# ===============================
@st.cache_resource
def load_all():
    try:
        model = tf.keras.models.load_model(
            "model_aksen.keras",
            custom_objects={"PrototypicalNetwork": PrototypicalNetwork}
        )
        print("✅ Model loaded successfully")
        print(f"Model type: {type(model)}")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None
    
    try:
        support_set = np.load("support_set.npy")
        support_labels = np.load("support_labels.npy")
        print(f"✅ Support set loaded: {support_set.shape}")
        print(f"✅ Support labels: {support_labels.shape}")
    except Exception as e:
        st.error(f"Error loading support set: {e}")
        return model, None, None
    
    return model, support_set, support_labels

model, support_set, support_labels = load_all()

# ===============================
# FEATURE EXTRACTION
# ===============================
def extract_mfcc(audio_path, sr=22050, n_mfcc=40, max_len=174):
    y, sr = librosa.load(audio_path, sr=sr)
    y = librosa.util.normalize(y)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=2048,
        hop_length=512
    )

    if mfcc.shape[1] < max_len:
        pad = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0,0),(0,pad)), mode="constant")
    else:
        mfcc = mfcc[:, :max_len]

    mfcc = mfcc[..., np.newaxis]  # (40,174,1)
    return mfcc.astype(np.float32)

# ===============================
# PROTOTYPE COMPUTATION
# ===============================
def compute_prototypes(model, support_set, support_labels, n_way):
    """Hitung prototipe dari support set menggunakan model embedding"""
    support_emb = model.predict(support_set, verbose=0)
    
    prototypes = []
    for i in range(n_way):
        mask = support_labels == i
        class_emb = support_emb[mask]
        proto = np.mean(class_emb, axis=0)
        prototypes.append(proto)
    
    return np.array(prototypes)

# ===============================
# STREAMLIT UI
# ===============================
st.title("🎙️ Deteksi Aksen Bahasa (Few-Shot Learning)")
st.write("Model Prototypical Network")

if model is None or support_set is None:
    st.error("❌ Gagal memuat model atau data support set.")
    st.stop()

# Hitung prototipe sekali saja saat aplikasi dimulai
@st.cache_data
def get_prototypes():
    return compute_prototypes(model, support_set, support_labels, N_WAY)

prototypes = get_prototypes()

audio_file = st.file_uploader(
    "Upload audio (.wav / .mp3)",
    type=["wav", "mp3"]
)

if audio_file:
    st.audio(audio_file)

    if st.button("🚀 Deteksi Aksen"):
        with st.spinner("Memproses audio..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_file.read())
                audio_path = tmp.name

            # QUERY FEATURE
            query_feat = extract_mfcc(audio_path)
            query_feat = query_feat[np.newaxis, ...]  # (1,40,174,1)
            
            # DAPATKAN EMBEDDING QUERY
            query_emb = model.predict(query_feat, verbose=0)
            
            # HITUNG JARAK KE PROTOTIPE
            distances = np.linalg.norm(
                query_emb[:, np.newaxis, :] - prototypes[np.newaxis, :, :],
                axis=2
            )
            
            logits = -distances
            probs = tf.nn.softmax(logits, axis=1).numpy()[0]
            pred_idx = np.argmax(probs)

            aksen_classes = [
                "Sunda",
                "Jawa Tengah",
                "Jawa Timur",
                "Yogyakarta",
                "Betawi"
            ]

            st.success(f"🎭 **Aksen Terdeteksi: {aksen_classes[pred_idx]}**")
            st.write("Probabilitas:")
            
            for i, cls in enumerate(aksen_classes):
                col1, col2, col3 = st.columns([2, 5, 2])
                with col1:
                    st.write(f"{cls}")
                with col2:
                    st.progress(float(probs[i]))
                with col3:
                    st.write(f"{probs[i]*100:.2f}%")

            os.unlink(audio_path)
