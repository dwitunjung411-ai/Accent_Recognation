import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tempfile
import os
import tensorflow as tf

# ==========================================================
# 1. CLASS PROTOTYPICAL NETWORK (MULTIPLE STRATEGIES)
# ==========================================================
@tf.keras.utils.register_keras_serializable(package="Custom")
class PrototypicalNetwork(tf.keras.Model):
    def __init__(self, embedding_model=None, **kwargs):
        super(PrototypicalNetwork, self).__init__(**kwargs)
        self._embedding = embedding_model
    
    def call(self, inputs, training=None):
        if isinstance(inputs, (list, tuple)):
            x = inputs[1] if len(inputs) > 1 else inputs[0]
        elif isinstance(inputs, dict):
            x = inputs.get('query_set', inputs.get('inputs', inputs))
        else:
            x = inputs
        
        if self._embedding is not None and callable(self._embedding):
            return self._embedding(x, training=training)
        
        for attr_name in ['embedding', '_embedding_layer']:
            if hasattr(self, attr_name):
                attr = getattr(self, attr_name)
                if callable(attr):
                    try:
                        return attr(x, training=training)
                    except:
                        pass
        
        return x
    
    def get_config(self):
        config = super().get_config()
        return config

# ==========================================================
# 2. LOAD MODEL (DENGAN DEBUGGING DETAIL)
# ==========================================================
@st.cache_resource
def load_accent_model():
    model_name = "model_embedding_aksen.keras"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, model_name)

    # Debug info
    debug_info = []
    debug_info.append(f"📁 Current directory: {current_dir}")
    debug_info.append(f"🔍 Looking for: {model_path}")
    debug_info.append(f"✅ File exists: {os.path.exists(model_path)}")
    
    if os.path.exists(model_path):
        debug_info.append(f"📊 File size: {os.path.getsize(model_path) / 1024 / 1024:.2f} MB")

    # Tampilkan debug info
    with st.sidebar:
        with st.expander("🔧 Debug Info", expanded=False):
            st.code("\n".join(debug_info))

    if not os.path.exists(model_path):
        st.error(f"❌ File model tidak ditemukan!")
        st.info(f"💡 Pastikan file '{model_name}' ada di folder yang sama dengan script")
        
        # Cek file apa aja yang ada
        files_in_dir = [f for f in os.listdir(current_dir) if f.endswith(('.keras', '.h5', '.pkl'))]
        if files_in_dir:
            st.warning(f"📂 File model yang ditemukan: {', '.join(files_in_dir)}")
        
        return None

    # STRATEGI 1: Load dengan custom objects
    try:
        st.info("🔄 Mencoba strategi 1: Load dengan PrototypicalNetwork...")
        custom_objects = {"PrototypicalNetwork": PrototypicalNetwork}
        model = tf.keras.models.load_model(
            model_path, 
            custom_objects=custom_objects, 
            compile=False
        )
        
        # Test model
        test_input = np.random.rand(1, 40).astype(np.float32)
        test_output = model.predict(test_input, verbose=0)
        
        if test_output.shape[-1] == 5:
            st.success("✅ Strategi 1 BERHASIL - Model dimuat dengan PrototypicalNetwork")
            return model
        else:
            st.warning(f"⚠️ Strategi 1: Output shape {test_output.shape}, bukan (1, 5)")
            
    except Exception as e1:
        st.warning(f"⚠️ Strategi 1 gagal: {str(e1)[:150]}")

    # STRATEGI 2: Load tanpa custom objects
    try:
        st.info("🔄 Mencoba strategi 2: Load tanpa custom objects...")
        model = tf.keras.models.load_model(model_path, compile=False)
        
        # Test model
        test_input = np.random.rand(1, 40).astype(np.float32)
        test_output = model.predict(test_input, verbose=0)
        
        if test_output.shape[-1] == 5:
            st.success("✅ Strategi 2 BERHASIL - Model dimuat tanpa custom objects")
            return model
        else:
            st.warning(f"⚠️ Strategi 2: Output shape {test_output.shape}")
            
    except Exception as e2:
        st.warning(f"⚠️ Strategi 2 gagal: {str(e2)[:150]}")

    # STRATEGI 3: Load dengan safe_mode
    try:
        st.info("🔄 Mencoba strategi 3: Load dengan safe_mode...")
        model = tf.keras.models.load_model(
            model_path, 
            compile=False,
            safe_mode=False
        )
        
        # Test model
        test_input = np.random.rand(1, 40).astype(np.float32)
        test_output = model.predict(test_input, verbose=0)
        
        if test_output.shape[-1] == 5:
            st.success("✅ Strategi 3 BERHASIL - Model dimuat dengan safe_mode=False")
            return model
            
    except Exception as e3:
        st.warning(f"⚠️ Strategi 3 gagal: {str(e3)[:150]}")

    # STRATEGI 4: Coba load sebagai saved_model format
    try:
        st.info("🔄 Mencoba strategi 4: Load sebagai SavedModel...")
        # Kadang .keras sebenarnya saved_model
        model_dir = model_path.replace('.keras', '')
        if os.path.exists(model_dir):
            model = tf.keras.models.load_model(model_dir)
            
            test_input = np.random.rand(1, 40).astype(np.float32)
            test_output = model.predict(test_input, verbose=0)
            
            if test_output.shape[-1] == 5:
                st.success("✅ Strategi 4 BERHASIL - Model dimuat sebagai SavedModel")
                return model
                
    except Exception as e4:
        st.warning(f"⚠️ Strategi 4 gagal: {str(e4)[:150]}")

    st.error("❌ Semua strategi loading gagal!")
    st.info("💡 Tips: Coba train ulang model dan save dengan format yang lebih sederhana")
    return None

# ==========================================================
# 3. LOAD METADATA
# ==========================================================
@st.cache_data
def load_metadata_df():
    csv_path = "metadata.csv"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

# ==========================================================
# 4. PREDIKSI AKSEN
# ==========================================================
def predict_accent(audio_path, model):
    if model is None:
        return "❌ Model tidak tersedia"
    
    try:
        # Load dan preprocess audio
        y, sr = librosa.load(audio_path, sr=16000, duration=10)
        
        # Extract MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_scaled = np.mean(mfcc.T, axis=0)
        
        # Prepare input
        input_data = np.expand_dims(mfcc_scaled, axis=0).astype(np.float32)
        
        # Prediksi
        prediction = model.predict(input_data, verbose=0)
        
        # Kelas aksen
        aksen_classes = ["Sunda", "Jawa Tengah", "Jawa Timur", "Yogyakarta", "Betawi"]
        predicted_idx = np.argmax(prediction[0])
        confidence = prediction[0][predicted_idx] * 100
        
        # Detail probabilitas semua kelas
        prob_detail = "\n".join([f"{cls}: {prob*100:.1f}%" for cls, prob in zip(aksen_classes, prediction[0])])
        
        return f"{aksen_classes[predicted_idx]} ({confidence:.1f}%)\n\n📊 Detail Probabilitas:\n{prob_detail}"
        
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ==========================================================
# 5. MAIN APPLICATION
# ==========================================================
def main():
    st.set_page_config(
        page_title="Deteksi Aksen Indonesia",
        page_icon="🎙️",
        layout="wide"
    )

    # Load resources
    model_aksen = load_accent_model()
    df_metadata = load_metadata_df()

    # Header
    st.title("🎙️ Sistem Deteksi Aksen Indonesia")
    st.write("Aplikasi berbasis *Deep Learning* untuk klasifikasi aksen daerah.")
    st.divider()

    # Sidebar
    with st.sidebar:
        st.header("🛸 Status Sistem")
        
        if model_aksen is not None:
            st.success("🤖 Model: Aktif")
        else:
            st.error("🚫 Model: Tidak Tersedia")
        
        if df_metadata is not None:
            st.success(f"📁 Metadata: {len(df_metadata)} records")
        else:
            st.warning("⚠️ Metadata: Tidak ada")
        
        st.divider()
        st.caption("🎓 Skripsi Project - 2026")

    # Main content
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader("📥 Input Audio")
        audio_file = st.file_uploader(
            "Upload file audio (.wav, .mp3)",
            type=["wav", "mp3"]
        )

        if audio_file:
            st.audio(audio_file, format=f'audio/{audio_file.name.split(".")[-1]}')
            
            if st.button("🚀 Analisis Aksen", type="primary", use_container_width=True):
                if model_aksen is not None:
                    with st.spinner("🔍 Menganalisis karakteristik suara..."):
                        # Save temp file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                            tmp.write(audio_file.getbuffer())
                            tmp_path = tmp.name

                        # Predict
                        hasil_aksen = predict_accent(tmp_path, model_aksen)

                        # Get metadata
                        user_info = None
                        if df_metadata is not None:
                            match = df_metadata[df_metadata['file_name'] == audio_file.name]
                            if not match.empty:
                                user_info = match.iloc[0].to_dict()

                        # Display results
                        with col2:
                            st.subheader("📊 Hasil Analisis")
                            
                            with st.container(border=True):
                                st.markdown("#### 🎭 Aksen Terdeteksi:")
                                if "❌" in hasil_aksen:
                                    st.error(hasil_aksen)
                                else:
                                    st.text(hasil_aksen)

                            st.divider()
                            
                            st.subheader("💎 Info Pembicara")
                            if user_info:
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    st.metric("🎂 Usia", f"{user_info.get('usia', '-')} Tahun")
                                    st.metric("🚻 Gender", user_info.get('gender', '-'))
                                with col_b:
                                    st.metric("🗺️ Provinsi", user_info.get('provinsi', '-'))
                            else:
                                st.info("🕵️ File tidak terdaftar dalam metadata")

                        # Cleanup
                        try:
                            os.unlink(tmp_path)
                        except:
                            pass
                else:
                    st.error("⚠️ Model tidak tersedia. Tidak dapat melakukan analisis.")
        else:
            with col2:
                st.info("👈 Upload file audio untuk memulai analisis")

if __name__ == "__main__":
    main()
