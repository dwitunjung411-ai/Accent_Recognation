import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tempfile
import os
import tensorflow as tf
import requests
from tensorflow.keras.models import load_model

# ==========================================================
# 1. DEFINISI CLASS MODEL (HARUS ADA SEBELUM LOAD MODEL)
# ==========================================================
@tf.keras.utils.register_keras_serializable(package="Custom")
class PrototypicalNetwork(tf.keras.Model):
    def __init__(self, embedding_model=None, **kwargs):
        super(PrototypicalNetwork, self).__init__(**kwargs)
        self.embedding = embedding_model

    def call(self, support_set, query_set, support_labels, n_way, training=False):
        # Memastikan input dalam bentuk tensor
        support_set = tf.convert_to_tensor(support_set, dtype=tf.float32)
        query_set = tf.convert_to_tensor(query_set, dtype=tf.float32)
        support_labels = tf.convert_to_tensor(support_labels, dtype=tf.int32)

        # Mendapatkan embedding fitur
        support_embeddings = self.embedding(support_set)
        query_embeddings = self.embedding(query_set)

        # Kalkulasi Prototype per kelas aksen
        prototypes = []
        for i in range(n_way):
            mask = tf.equal(support_labels, i)
            class_embeddings = tf.boolean_mask(support_embeddings, mask)
            
            # Jika kelas kosong, beri nilai nol (cegah error pembagian)
            if tf.shape(class_embeddings)[0] == 0:
                prototype = tf.zeros_like(support_embeddings[0])
            else:
                prototype = tf.reduce_mean(class_embeddings, axis=0)
            prototypes.append(prototype)

        prototypes = tf.stack(prototypes)

        # Kalkulasi jarak Euclidean
        distances = tf.norm(
            tf.expand_dims(query_embeddings, 1) - tf.expand_dims(prototypes, 0),
            axis=2
        )
        return -distances

    def get_config(self):
        config = super().get_config()
        if self.embedding:
            config.update({"embedding_model": tf.keras.layers.serialize(self.embedding)})
        return config

# ==========================================================
# 2. FUNGSI LOAD RESOURCES DARI GITHUB (PERBAIKAN)
# ==========================================================
@st.cache_resource
def load_model_from_url(model_url):
    """Load model langsung dari URL"""
    try:
        # Download model
        response = requests.get(model_url)
        response.raise_for_status()
        
        # Simpan sementara
        with tempfile.NamedTemporaryFile(delete=False, suffix='.keras') as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name
        
        # Load model dengan custom objects
        custom_objects = {"PrototypicalNetwork": PrototypicalNetwork}
        model = tf.keras.models.load_model(tmp_path, custom_objects=custom_objects, compile=False)
        
        # Hapus file temporary
        os.unlink(tmp_path)
        
        return model
    except Exception as e:
        st.error(f"Failed to load model from URL: {str(e)}")
        return None

@st.cache_resource
def load_numpy_from_url(url):
    """Load numpy array dari URL"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # Simpan sementara
        with tempfile.NamedTemporaryFile(delete=False, suffix='.npy') as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name
        
        # Load numpy
        data = np.load(tmp_path, allow_pickle=True)
        
        # Hapus file temporary
        os.unlink(tmp_path)
        
        return data
    except Exception as e:
        st.error(f"Failed to load data from URL: {str(e)}")
        return None

def load_resources():
    """Load semua resources yang diperlukan"""
    # PERBAIKAN: Gunakan raw.githubusercontent.com bukan github.com biasa
    GITHUB_BASE = "https://raw.githubusercontent.com/dwitunjung411-ai/Accent_Recognation/main/"
    
    MODEL_URL = GITHUB_BASE + "model_detect_aksen.keras"
    SUPPORT_SET_URL = GITHUB_BASE + "support_set.npy"
    SUPPORT_LABELS_URL = GITHUB_BASE + "support_labels.npy"
    
    try:
        with st.spinner("Loading model from GitHub..."):
            model = load_model_from_url(MODEL_URL)
        
        with st.spinner("Loading support set from GitHub..."):
            support_set = load_numpy_from_url(SUPPORT_SET_URL)
        
        with st.spinner("Loading support labels from GitHub..."):
            support_labels = load_numpy_from_url(SUPPORT_LABELS_URL)
        
        if model is not None and support_set is not None and support_labels is not None:
            st.success("✅ All resources loaded successfully from GitHub!")
            return model, support_set, support_labels
        else:
            st.error("Failed to load one or more resources.")
            return None, None, None
            
    except Exception as e:
        st.error(f"Error loading from GitHub: {str(e)}")
        return None, None, None

def load_accent_model():
    """Wrapper untuk load model (kompatibilitas dengan kode lama)"""
    model, _, _ = load_resources()
    return model

def load_metadata_df():
    """Load metadata.csv jika ada"""
    try:
        # Coba load metadata dari GitHub
        metadata_url = "https://raw.githubusercontent.com/dwitunjung411-ai/Accent_Recognation/main/metadata.csv"
        response = requests.get(metadata_url)
        if response.status_code == 200:
            import io
            df = pd.read_csv(io.StringIO(response.text))
            return df
    except:
        pass
    
    # Coba dari file lokal
    if os.path.exists("metadata.csv"):
        return pd.read_csv("metadata.csv")
    
    return None

# ==========================================================
# 3. FUNGSI EKSTRAKSI MFCC 3-CHANNEL
# ==========================================================
def extract_mfcc_3channel(file_path, sr=22050, n_mfcc=40, max_len=174):
    """Ekstrak MFCC 3-channel sesuai format model"""
    try:
        y, sr = librosa.load(file_path, sr=sr)
        y = librosa.util.normalize(y)
        
        # MFCC dengan delta dan delta-delta
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=2048, hop_length=512)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Padding atau truncating ke max_len
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
        return np.stack([mfcc, delta, delta2], axis=-1)  # Shape: (40, 174, 3)
        
    except Exception as e:
        st.error(f"Gagal mengekstrak fitur: {e}")
        return None

# ==========================================================
# 4. FUNGSI PREDIKSI AKSEN YANG BENAR
# ==========================================================
def predict_accent(audio_path, model, support_set, support_labels):
    """Prediksi aksen menggunakan model Prototypical"""
    if model is None: 
        return "Model tidak tersedia", None
    
    try:
        # 1. Ekstrak fitur dari audio query
        query_feat = extract_mfcc_3channel(audio_path)
        if query_feat is None:
            return "Gagal mengekstrak fitur", None
        
        # 2. Siapkan input untuk model Prototypical
        query_tensor = np.expand_dims(query_feat, axis=0)  # (1, 40, 174, 3)
        
        # 3. Lakukan prediksi dengan model Prototypical
        # Model Prototypical membutuhkan: support_set, query_set, support_labels, n_way
        n_way = len(np.unique(support_labels))
        
        logits = model.call(
            support_set,      # support set
            query_tensor,     # query set
            support_labels,   # support labels
            n_way,            # jumlah kelas
            training=False    # mode inference
        )
        
        # 4. Konversi logits ke probabilitas
        if hasattr(logits, 'numpy'):
            logits = logits.numpy()
        
        probabilities = tf.nn.softmax(logits[0]).numpy()
        predicted_idx = np.argmax(probabilities)
        confidence = probabilities[predicted_idx] * 100
        
        # 5. Mapping ke nama aksen
        aksen_classes = ["Sunda", "Jawa Tengah", "Jawa Timur", "Yogyakarta", "Betawi"]
        
        # Pastikan predicted_idx dalam range
        if predicted_idx >= len(aksen_classes):
            predicted_idx = 0
        
        result = aksen_classes[predicted_idx]
        
        return result, {
            'confidence': confidence,
            'probabilities': probabilities,
            'all_classes': aksen_classes
        }
        
    except Exception as e:
        return f"Error Analisis: {str(e)}", None

# ==========================================================
# 5. MAIN UI APPLICATION
# ==========================================================
def main():
    # Set layout lebar
    st.set_page_config(
        page_title="Deteksi Aksen Prototypical", 
        page_icon="🎙️", 
        layout="wide"
    )

    # Load semua resources sekaligus
    with st.spinner("Memuat model dan data dari GitHub..."):
        model, support_set, support_labels = load_resources()
        df_metadata = load_metadata_df()

    st.title("🎙️ Sistem Deteksi Aksen Prototypical Indonesia")
    st.write("Aplikasi berbasis *Few-Shot Learning* untuk klasifikasi aksen daerah.")
    st.divider()

    # Sidebar
    with st.sidebar:
        st.header("🛸 Status Sistem")
        
        if model is not None:
            st.success("🤖 Model: Terhubung")
        else:
            st.error("🚫 Model: Terputus")
        
        if support_set is not None:
            st.success(f"📊 Support Set: {support_set.shape[0]} samples")
        else:
            st.error("📊 Support Set: Tidak tersedia")
        
        if df_metadata is not None:
            st.success(f"📁 Metadata: {len(df_metadata)} records")
        else:
            st.warning("⚠️ Metadata: Kosong")
        
        st.divider()
        st.caption("Skripsi Project - 2026")
        
        # Informasi tambahan
        if model is not None and support_set is not None:
            st.info(f"**Format Input:** {support_set.shape[1:]}")
            st.info(f"**Jumlah Kelas:** {len(np.unique(support_labels))}")

    # Pembagian kolom utama
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader("📥 Input Audio")
        st.markdown("""
        **Format yang didukung:**
        - WAV, MP3
        - Durasi optimal: 3-10 detik
        - Sample rate: 22.05 kHz (akan di-resample otomatis)
        """)
        
        audio_file = st.file_uploader("Upload file audio", type=["wav", "mp3"], label_visibility="collapsed")

        if audio_file:
            st.audio(audio_file)
            
            # Tombol analisis
            if st.button("🚀 Extract Feature and Detect", type="primary", use_container_width=True):
                if model is not None and support_set is not None and support_labels is not None:
                    with st.spinner("Menganalisis karakteristik suara..."):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                            tmp.write(audio_file.getbuffer())
                            tmp_path = tmp.name

                        # Prediksi aksen
                        hasil_aksen, details = predict_accent(tmp_path, model, support_set, support_labels)
                        
                        # Pencarian metadata
                        user_info = None
                        if df_metadata is not None:
                            match = df_metadata[df_metadata['file_name'] == audio_file.name]
                            if not match.empty:
                                user_info = match.iloc[0].to_dict()

                        # Tampilkan hasil di kolom 2
                        with col2:
                            st.subheader("📊 Hasil Analisis")
                            
                            # Container hasil
                            with st.container(border=True):
                                if isinstance(hasil_aksen, str) and "Error" in hasil_aksen:
                                    st.error(f"**Error:** {hasil_aksen}")
                                else:
                                    col_a, col_b = st.columns([2, 1])
                                    with col_a:
                                        st.markdown(f"#### 🎭 Aksen Terdeteksi:")
                                        st.success(f"**{hasil_aksen}**")
                                    with col_b:
                                        if details and 'confidence' in details:
                                            st.metric("Confidence", f"{details['confidence']:.1f}%")
                            
                            # Detail probabilitas jika ada
                            if details and 'probabilities' in details:
                                st.divider()
                                st.subheader("📈 Probabilitas per Aksen")
                                
                                prob_df = pd.DataFrame({
                                    'Aksen': details['all_classes'],
                                    'Probabilitas': details['probabilities'],
                                    'Persentase': [f"{p*100:.1f}%" for p in details['probabilities']]
                                })
                                
                                st.dataframe(
                                    prob_df.sort_values('Probabilitas', ascending=False),
                                    use_container_width=True,
                                    hide_index=True
                                )
                            
                            st.divider()
                            st.subheader("💎 Info Pembicara")
                            
                            if user_info:
                                info_cols = st.columns(3)
                                with info_cols[0]:
                                    st.metric("🎂 Usia", f"{user_info.get('usia', '-')} Tahun")
                                with info_cols[1]:
                                    st.metric("🚻 Gender", user_info.get('gender', '-'))
                                with info_cols[2]:
                                    st.metric("🗺️ Provinsi", user_info.get('provinsi', '-'))
                            else:
                                st.warning("🕵️ Data file tidak terdaftar di metadata.csv")
                                
                                # Form input manual
                                with st.expander("➕ Tambah Informasi Manual"):
                                    manual_age = st.number_input("Usia", min_value=10, max_value=100, value=30)
                                    manual_gender = st.selectbox("Gender", ["Laki-laki", "Perempuan"])
                                    manual_province = st.selectbox("Provinsi", [
                                        "DKI Jakarta", "Jawa Barat", "Jawa Tengah", 
                                        "Jawa Timur", "DI Yogyakarta", "Banten", "Lainnya"
                                    ])
                                    
                                    if st.button("Simpan Info Manual"):
                                        st.success("Informasi disimpan untuk sesi ini")

                        # Cleanup
                        os.unlink(tmp_path)
                else:
                    st.error("Gagal memproses: Model atau data support tidak ditemukan.")
                    
                    # Debug info
                    with st.expander("ℹ️ Informasi Debug"):
                        st.write(f"Model: {type(model)}")
                        st.write(f"Support Set: {type(support_set)}")
                        st.write(f"Support Labels: {type(support_labels)}")
                        
                        # Link ke GitHub
                        st.markdown("""
                        **File yang diperlukan:**
                        - [model_detect_aksen.keras](https://raw.githubusercontent.com/dwitunjung411-ai/Accent_Recognation/main/model_detect_aksen.keras)
                        - [support_set.npy](https://raw.githubusercontent.com/dwitunjung411-ai/Accent_Recognation/main/support_set.npy)
                        - [support_labels.npy](https://raw.githubusercontent.com/dwitunjung411-ai/Accent_Recognation/main/support_labels.npy)
                        """)

    # Jika belum ada file diupload, tampilkan instruksi di kolom 2
    if not audio_file:
        with col2:
            st.subheader("ℹ️ Panduan Penggunaan")
            st.info("""
            1. **Upload** file audio di kolom kiri
            2. **Klik tombol** "Extract Feature and Detect"
            3. **Hasil** akan muncul di kolom kanan
            
            **Model yang digunakan:**
            - Prototypical Network untuk few-shot learning
            - MFCC 3-channel sebagai fitur
            - 5 kelas aksen: Sunda, Jawa Tengah, Jawa Timur, Yogyakarta, Betawi
            """)
            
            # Contoh format
            st.subheader("📋 Format Input Model")
            if support_set is not None:
                st.code(f"""
                Input Shape: {support_set.shape[1:]}
                Support Samples: {support_set.shape[0]}
                Jumlah Kelas: {len(np.unique(support_labels)) if support_labels is not None else 'N/A'}
                """)

if __name__ == "__main__":
    main()
