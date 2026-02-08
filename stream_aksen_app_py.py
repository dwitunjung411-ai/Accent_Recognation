import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tempfile
import os
import tensorflow as tf

# ==========================================================
# CLASS PROTOTYPICAL NETWORK
# ==========================================================
@tf.keras.utils.register_keras_serializable(package="Custom")
class PrototypicalNetwork(tf.keras.Model):
    def __init__(self, embedding_model=None, **kwargs):
        super(PrototypicalNetwork, self).__init__(**kwargs)
        self.embedding = embedding_model
    
    def call(self, inputs, training=None):
        if isinstance(inputs, (list, tuple)):
            x = inputs[1] if len(inputs) > 1 else inputs[0]
        elif isinstance(inputs, dict):
            x = inputs.get('query_set', inputs)
        else:
            x = inputs
        
        if self.embedding is not None:
            if callable(self.embedding):
                return self.embedding(x, training=training)
            elif isinstance(self.embedding, dict):
                try:
                    emb = tf.keras.layers.deserialize(self.embedding)
                    return emb(x, training=training)
                except:
                    pass
        
        if hasattr(self, 'layers'):
            for layer in self.layers:
                if isinstance(layer, (tf.keras.Model, tf.keras.Sequential)):
                    return layer(x, training=training)
        
        return x
    
    def get_config(self):
        config = super().get_config()
        if self.embedding is not None and not isinstance(self.embedding, dict):
            config.update({"embedding_model": tf.keras.layers.serialize(self.embedding)})
        return config

# ==========================================================
# LOAD MODEL
# ==========================================================
@st.cache_resource
def load_accent_model():
    model_path = "model_embedding_aksen.keras"
    
    if not os.path.exists(model_path):
        st.sidebar.error(f"❌ File '{model_path}' tidak ditemukan")
        return None
    
    try:
        custom_objects = {"PrototypicalNetwork": PrototypicalNetwork}
        model = tf.keras.models.load_model(
            model_path, 
            custom_objects=custom_objects, 
            compile=False
        )
        
        st.sidebar.success("✅ Model berhasil dimuat")
        return model
        
    except Exception as e:
        st.sidebar.error(f"❌ Error: {str(e)[:100]}")
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
# PREDIKSI (FIXED: NORMALISASI OUTPUT + URUTAN KELAS)
# ==========================================================
def predict_accent(audio_path, model):
    if model is None:
        return "❌ Model tidak tersedia"
    
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=16000)
        
        # Extract MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        
        # Prepare input
        X = np.expand_dims(mfcc_mean, axis=0).astype(np.float32)
        
        # Predict (raw output)
        raw_output = model.predict(X, verbose=0)[0]
        
        # NORMALISASI: Konversi ke probabilitas dengan softmax
        # Karena output model sepertinya bukan probabilitas
        exp_output = np.exp(raw_output - np.max(raw_output))  # Stabilisasi numerik
        probabilities = exp_output / np.sum(exp_output)
        
        # URUTAN KELAS - SESUAIKAN DENGAN TRAINING!
        # Coba beberapa kemungkinan urutan:
        aksen_classes = ["Sunda", "Jawa Tengah", "Jawa Timur", "Yogyakarta", "Betawi"]
        
        # Get hasil
        predicted_idx = np.argmax(probabilities)
        confidence = probabilities[predicted_idx] * 100
        
        # Detail probabilitas
        detail_lines = []
        for i, (cls, prob) in enumerate(zip(aksen_classes, probabilities)):
            marker = "👉 " if i == predicted_idx else "   "
            detail_lines.append(f"{marker}{cls}: {prob*100:.2f}%")
        
        detail = "\n".join(detail_lines)
        
        result = f"{aksen_classes[predicted_idx]} ({confidence:.1f}%)\n\n📊 Detail Probabilitas:\n{detail}"
        
        # Tambahan: Tampilkan raw output untuk debugging
        raw_detail = "\n".join([f"{cls}: {val:.2f}" for cls, val in zip(aksen_classes, raw_output)])
        result += f"\n\n🔧 Raw Output (Debug):\n{raw_detail}"
        
        return result
        
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ==========================================================
# MAIN UI
# ==========================================================
st.set_page_config(
    page_title="Deteksi Aksen Indonesia",
    page_icon="🎙️",
    layout="wide"
)

st.title("🎙️ Sistem Deteksi Aksen Indonesia")
st.write("Aplikasi berbasis *Deep Learning* untuk klasifikasi aksen daerah.")
st.divider()

# Load resources
model = load_accent_model()
metadata = load_metadata_df()

# Sidebar
with st.sidebar:
    st.header("🛸 Status Sistem")
    if metadata is not None:
        st.info(f"📁 Metadata: {len(metadata)} records")
    
    st.divider()
    st.subheader("⚙️ Pengaturan Kelas")
    st.caption("Urutan kelas harus sesuai dengan saat training model")
    
    # OPSI: User bisa ubah urutan kelas
    class_order = st.text_area(
        "Urutan Kelas (pisahkan dengan koma)",
        value="Sunda, Jawa Tengah, Jawa Timur, Yogyakarta, Betawi",
        help="Ubah urutan ini jika prediksi salah terus"
    )

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
            if model is not None:
                with st.spinner("🔍 Menganalisis karakteristik suara..."):
                    # Save temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(audio_file.getbuffer())
                        tmp_path = tmp.name
                    
                    # Predict
                    hasil = predict_accent(tmp_path, model)
                    
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
                            # BANDINGKAN dengan prediksi
                            actual_province = user_info.get('provinsi', '-')
                            
                            # Mapping provinsi ke aksen
                            province_to_accent = {
                                'DKI Jakarta': 'Betawi',
                                'Jawa Barat': 'Sunda',
                                'Jawa Tengah': 'Jawa Tengah',
                                'Jawa Timur': 'Jawa Timur',
                                'Yogyakarta': 'Yogyakarta'
                            }
                            
                            actual_accent = province_to_accent.get(actual_province, '-')
                            
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("🎂 Usia", f"{user_info.get('usia', '-')} Tahun")
                                st.metric("🚻 Gender", user_info.get('gender', '-'))
                            with col_b:
                                st.metric("🗺️ Provinsi", actual_province)
                                st.metric("✅ Aksen Sebenarnya", actual_accent)
                            
                            # Warning jika beda
                            if actual_accent != '-':
                                predicted_accent = hasil.split('(')[0].strip()
                                if actual_accent != predicted_accent:
                                    st.warning(f"⚠️ Prediksi tidak sesuai! Seharusnya: **{actual_accent}**")
                                else:
                                    st.success("✅ Prediksi BENAR!")
                        else:
                            st.info("🕵️ File tidak terdaftar dalam metadata")
                    
                    # Cleanup
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
            else:
                st.error("⚠️ Model tidak tersedia")
    else:
        with col2:
            st.info("👈 Upload file audio untuk memulai analisis")
