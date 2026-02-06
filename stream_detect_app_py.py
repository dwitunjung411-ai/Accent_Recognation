import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import tempfile
import os
import traceback
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

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
# FUNGSI UTAMA - TANPA LOAD MODEL
# ===============================

@st.cache_resource
def load_and_prepare():
    """Load support set dan siapkan scaler"""
    try:
        # Load support data
        support_set = np.load("support_set.npy")
        support_labels = np.load("support_labels.npy")
        
        st.success(f"✅ Support set loaded: {support_set.shape}")
        st.success(f"✅ Support labels: {support_labels.shape}")
        
        # Normalisasi data
        scaler = StandardScaler()
        support_set_normalized = scaler.fit_transform(support_set)
        
        # Hitung prototipe secara langsung
        prototypes = []
        aksen_classes = ["Sunda", "Jawa Tengah", "Jawa Timur", "Yogyakarta", "Betawi"]
        
        for i in range(N_WAY):
            mask = support_labels == i
            class_data = support_set_normalized[mask]
            if len(class_data) > 0:
                proto = np.mean(class_data, axis=0)
                prototypes.append(proto)
            else:
                prototypes.append(np.zeros(support_set.shape[1]))
        
        prototypes = np.array(prototypes)
        
        st.success(f"✅ Prototypes calculated: {prototypes.shape}")
        
        return support_set_normalized, support_labels, prototypes, scaler, aksen_classes
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None, None, None, None, None

def extract_features_simple(audio_path, sr=22050):
    """Ekstrak fitur sederhana 13-dimensi"""
    try:
        # Load audio (ambil 3 detik pertama)
        y, sr = librosa.load(audio_path, sr=sr, duration=3.0)
        
        # Normalisasi audio
        y = librosa.util.normalize(y)
        
        # Ekstrak 13 MFCC
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=13,
            n_fft=2048,
            hop_length=512
        )
        
        # Ambil statistik: mean, std, min, max dari 13 koefisien pertama
        features = []
        for i in range(min(13, mfcc.shape[0])):
            features.append(np.mean(mfcc[i]))
            features.append(np.std(mfcc[i]))
            # features.append(np.min(mfcc[i]))
            # features.append(np.max(mfcc[i]))
        
        # Pastikan ada 13 fitur
        features = np.array(features[:13])  # Ambil 13 pertama
        
        return features.astype(np.float32)
        
    except Exception as e:
        st.error(f"Error extracting features: {str(e)}")
        return np.zeros(13, dtype=np.float32)

def calculate_distances(query_features, prototypes):
    """Hitung jarak Euclidean"""
    # Pastikan shape cocok
    if len(query_features.shape) == 1:
        query_features = query_features.reshape(1, -1)
    
    # Hitung jarak Euclidean
    distances = np.sqrt(np.sum((query_features - prototypes) ** 2, axis=1))
    
    # Konversi ke similarity score (lebih tinggi = lebih mirim)
    similarities = 1 / (1 + distances)
    
    return distances, similarities

# ===============================
# STREAMLIT UI
# ===============================

def main():
    st.title("🎙️ Deteksi Aksen Bahasa Indonesia")
    st.write("Menggunakan Few-Shot Learning dengan Prototypical Network")
    st.write("**Mode: Direct Distance Calculation**")
    
    # Load data
    with st.spinner("Memuat data support set..."):
        support_set_norm, support_labels, prototypes, scaler, aksen_classes = load_and_prepare()
    
    if support_set_norm is None:
        st.error("Gagal memuat data. Pastikan file support_set.npy dan support_labels.npy ada.")
        st.stop()
    
    st.divider()
    
    # Sidebar info
    with st.sidebar:
        st.header("📊 Informasi Sistem")
        st.write(f"**Jumlah kelas:** {N_WAY}")
        st.write(f"**Fitur dimensi:** {prototypes.shape[1]}")
        st.write(f"**Samples support:** {len(support_set_norm)}")
        
        st.divider()
        
        st.header("📋 Kelas Aksen")
        for i, cls in enumerate(aksen_classes):
            count = np.sum(support_labels == i)
            st.write(f"{i+1}. {cls} ({count} samples)")
    
    # Main content
    st.subheader("📤 Upload Audio")
    
    audio_file = st.file_uploader(
        "Pilih file audio (.wav, .mp3)",
        type=["wav", "mp3"],
        help="Upload rekaman suara untuk dideteksi aksennya"
    )
    
    if audio_file:
        # Tampilkan audio
        st.audio(audio_file, format="audio/wav")
        
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
                        features = extract_features_simple(audio_path)
                        
                        # 2. Normalisasi dengan scaler yang sama
                        features_normalized = scaler.transform(features.reshape(1, -1))
                        
                        # 3. Hitung jarak ke prototipe
                        distances, similarities = calculate_distances(features_normalized, prototypes)
                        
                        # 4. Konversi ke probabilitas dengan softmax
                        # Gunakan negative distances untuk softmax (lebih kecil = lebih baik)
                        exp_scores = np.exp(-distances)
                        probs = exp_scores / np.sum(exp_scores)
                        
                        # 5. Ambil prediksi
                        pred_idx = np.argmax(probs)
                        
                        # TAMPILKAN HASIL
                        st.divider()
                        st.subheader("📊 Hasil Deteksi")
                        
                        # Hasil utama
                        col_a, col_b, col_c = st.columns([1, 2, 1])
                        with col_b:
                            color = "#4CAF50" if probs[pred_idx] > 0.5 else "#FF9800"
                            st.markdown(f"""
                            <div style="text-align: center; padding: 20px; border-radius: 10px; 
                                        background-color: {color};
                                        color: white; margin: 20px 0;">
                                <h2 style="margin: 0;">{aksen_classes[pred_idx]}</h2>
                                <p style="margin: 5px 0 0 0; font-size: 1.2em;">
                                    Confidence: {probs[pred_idx]*100:.1f}%
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Grafik probabilitas
                        st.write("### 📈 Probabilitas per Kelas:")
                        
                        for i, (cls, prob) in enumerate(zip(aksen_classes, probs)):
                            cols = st.columns([2, 6, 2])
                            with cols[0]:
                                if i == pred_idx:
                                    st.markdown(f"**🏆 {cls}**")
                                else:
                                    st.write(cls)
                            with cols[1]:
                                st.progress(float(prob))
                            with cols[2]:
                                st.write(f"{prob*100:.1f}%")
                        
                        # Detail teknis (expander)
                        with st.expander("🔧 Detail Teknis"):
                            st.write("**Fitur yang diekstrak:**")
                            st.dataframe(features.reshape(1, -1), use_container_width=True)
                            
                            st.write("**Jarak ke prototipe:**")
                            for i, (cls, dist) in enumerate(zip(aksen_classes, distances)):
                                st.write(f"{cls}: {dist:.3f}")
                            
                            st.write("**Similarity scores:**")
                            for i, (cls, sim) in enumerate(zip(aksen_classes, similarities)):
                                st.write(f"{cls}: {sim:.3f}")
                                
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
    
    # Tambahkan penjelasan
    st.divider()
    with st.expander("ℹ️ Cara Kerja Sistem"):
        st.write("""
        1. **Ekstraksi Fitur**: Sistem mengekstrak 13 fitur MFCC dari audio
        2. **Normalisasi**: Fitur dinormalisasi menggunakan StandardScaler
        3. **Perhitungan Jarak**: Dihitung jarak Euclidean ke prototipe setiap kelas
        4. **Klasifikasi**: Kelas dengan jarak terdekat dipilih sebagai prediksi
        
        **Prototipe** dihitung sebagai rata-rata dari support set setiap kelas.
        """)

if __name__ == "__main__":
    main()
