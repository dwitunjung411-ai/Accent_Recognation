import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tempfile
import os

# ==========================================================
# LOAD MODEL PALING SEDERHANA - FIX REGISTRATION ERROR
# ==========================================================
@st.cache_resource
def load_accent_model():
    import tensorflow as tf
    
    # BERSIHKAN REGISTRASI SEBELUMNYA JIKA ADA
    # Ini mencegah error "has already been registered"
    registered_name = "Custom>PrototypicalNetwork"
    if registered_name in tf.keras.saving.object_registration._GLOBAL_CUSTOM_OBJECTS:
        del tf.keras.saving.object_registration._GLOBAL_CUSTOM_OBJECTS[registered_name]
    
    model_path = "model_detect_aksen.keras"
    
    # Cek file ada atau tidak
    if not os.path.exists(model_path):
        st.sidebar.error(f"❌ File '{model_path}' tidak ditemukan")
        return None
    
    try:
        # Load langsung tanpa apapun
        model = tf.keras.models.load_model(model_path, compile=False)
        st.sidebar.success("✅ Model loaded")
        return model
    except Exception as e1:
        error_msg = str(e1)
        
        # Jika error karena custom object, coba dengan custom_objects
        if "PrototypicalNetwork" in error_msg or "custom" in error_msg.lower():
            try:
                # Definisikan class minimal untuk load model
                class PrototypicalNetwork(tf.keras.Model):
                    def __init__(self, embedding_model=None, **kwargs):
                        super().__init__(**kwargs)
                        self.embedding = embedding_model
                    
                    def call(self, inputs):
                        return self.embedding(inputs)
                    
                    def get_config(self):
                        config = super().get_config()
                        config.update({"embedding_model": tf.keras.layers.serialize(self.embedding)})
                        return config
                
                model = tf.keras.models.load_model(
                    model_path, 
                    compile=False, 
                    custom_objects={"PrototypicalNetwork": PrototypicalNetwork}
                )
                st.sidebar.success("✅ Model loaded (with custom_objects)")
                return model
                
            except Exception as e2:
                st.sidebar.error(f"❌ Custom object fail: {str(e2)[:100]}")
                return None
        
        # Coba dengan safe_mode=False untuk model yang tidak compatible
        try:
            model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
            st.sidebar.success("✅ Model loaded (safe_mode=False)")
            return model
        except Exception as e3:
            st.sidebar.error(f"❌ Gagal load: {str(e3)[:150]}")
            return None

# ==========================================================
# LOAD METADATA
# ==========================================================
@st.cache_data
def load_metadata_df():
    if os.path.exists("metadata.csv"):
        return pd.read_csv("metadata.csv")
    return None

# ==========================================================
# PREDIKSI
# ==========================================================
def predict_accent(audio_path, model):
    if model is None:
        return "Model tidak tersedia"
    
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=16000)
        
        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        
        # Input
        X = np.expand_dims(mfcc_mean, axis=0)
        
        # Predict
        pred = model.predict(X, verbose=0)
        
        # Hasil
        classes = ["Sunda", "Jawa Tengah", "Jawa Timur", "Yogyakarta", "Betawi"]
        idx = np.argmax(pred[0])
        conf = pred[0][idx] * 100
        
        return f"{classes[idx]} ({conf:.1f}%)"
        
    except Exception as e:
        return f"Error: {str(e)}"

# ==========================================================
# UI
# ==========================================================
st.set_page_config(page_title="Deteksi Aksen", page_icon="🎙️", layout="wide")

st.title("🎙️ Deteksi Aksen Indonesia")
st.divider()

# Load
model = load_accent_model()
metadata = load_metadata_df()

# Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📥 Input Audio")
    
    audio = st.file_uploader("Upload (.wav, .mp3)", type=["wav", "mp3"])
    
    if audio:
        st.audio(audio)
        
        if st.button("🚀 Analisis", type="primary", use_container_width=True):
            if model:
                with st.spinner("Analyzing..."):
                    # Save temp
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                        f.write(audio.getbuffer())
                        path = f.name
                    
                    # Predict
                    result = predict_accent(path, model)
                    
                    # Show
                    with col2:
                        st.subheader("📊 Hasil")
                        st.success(result)
                        
                        st.divider()
                        
                        # Metadata
                        if metadata is not None:
                            match = metadata[metadata['file_name'] == audio.name]
                            if not match.empty:
                                info = match.iloc[0]
                                st.write(f"🎂 Usia: {info.get('usia', '-')} Tahun")
                                st.write(f"🚻 Gender: {info.get('gender', '-')}")
                                st.write(f"🗺️ Provinsi: {info.get('provinsi', '-')}")
                    
                    # Cleanup
                    os.unlink(path)
            else:
                st.error("Model tidak tersedia")
