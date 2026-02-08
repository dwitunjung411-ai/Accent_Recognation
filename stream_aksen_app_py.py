import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tempfile
import os
import tensorflow as tf
import traceback
from sklearn.preprocessing import LabelEncoder

# ==========================================================
# 1. DEFINISI CLASS MODEL (PERBAIKAN)
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
# 2. FUNGSI EKSTRAKSI MFCC 3-CHANNEL
# ==========================================================
def extract_mfcc_3channel(file_path, sr=22050, n_mfcc=40, max_len=174):
    try:
        y, sr = librosa.load(file_path, sr=sr)
        y = librosa.util.normalize(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=2048, hop_length=512)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)

        if mfcc.shape[1] < max_len:
            pad_width = max_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
            delta = np.pad(delta, ((0, 0), (0, pad_width)), mode='constant')
            delta2 = np.pad(delta2, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc, delta, delta2 = mfcc[:, :max_len], delta[:, :max_len], delta2[:, :max_len]

        return np.stack([mfcc, delta, delta2], axis=-1)
    except Exception as e:
        st.error(f"Gagal memproses audio: {e}")
        return None

# ==========================================================
# 3. LOADING MODEL & SUPPORT SET ASLI
# ==========================================================
@st.cache_resource
def load_model_and_support_set():
    """Memuat model dan support set yang sesungguhnya dari training"""
    model_path = "model_detect_aksen.keras"
    support_set_path = "support_set.npy"
    support_labels_path = "support_labels.npy"
    
    # Cek keberadaan file
    missing_files = []
    if not os.path.exists(model_path):
        missing_files.append("model_detect_aksen.keras")
    if not os.path.exists(support_set_path):
        missing_files.append("support_set.npy")
    if not os.path.exists(support_labels_path):
        missing_files.append("support_labels.npy")
    
    if missing_files:
        st.error(f"File berikut tidak ditemukan: {', '.join(missing_files)}")
        st.info("""
        **Solusi:**
        1. Pastikan file-file berikut ada di direktori yang sama:
           - model_detect_aksen.keras (model hasil training)
           - support_set.npy (support set dari training)
           - support_labels.npy (labels dari training)
        
        2. Jika belum ada support_set.npy, buat dari skrip training dengan:
           ```python
           np.save('support_set.npy', support_set_used_in_training)
           np.save('support_labels.npy', support_labels_used_in_training)
           ```
        """)
        return None, None, None, None
    
    try:
        # Load model
        custom_objects = {"PrototypicalNetwork": PrototypicalNetwork}
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
        
        # Load support set asli dari training
        support_set = np.load(support_set_path, allow_pickle=True)
        support_labels = np.load(support_labels_path, allow_pickle=True)
        
        # Pastikan support_set adalah array numpy
        if not isinstance(support_set, np.ndarray):
            support_set = np.array(support_set)
        if not isinstance(support_labels, np.ndarray):
            support_labels = np.array(support_labels)
        
        # Validasi shape
        expected_shape = (None, 40, 174, 3)
        if len(support_set.shape) != 4 or support_set.shape[1:] != expected_shape[1:]:
            st.error(f"Shape support set tidak valid: {support_set.shape}. Diharapkan: {expected_shape}")
            return None, None, None, None
        
        # Hitung n_way (jumlah kelas unik)
        n_way = len(np.unique(support_labels))
        
        # Mendapatkan mapping label ke nama aksen
        # Asumsi: support_labels sudah encoded sebagai 0, 1, 2, ...
        label_mapping = {
            0: "Sunda",
            1: "Jawa Tengah", 
            2: "Jawa Timur",
            3: "Yogyakarta",
            4: "Betawi"
        }
        
        # Buat list aksen berdasarkan n_way
        aksen_list = [label_mapping.get(i, f"Aksen_{i}") for i in range(n_way)]
        
        st.success(f"✅ Model dan support set berhasil dimuat!")
        st.success(f"   • Jumlah kelas: {n_way}")
        st.success(f"   • Jumlah sample support: {len(support_set)}")
        st.success(f"   • Shape support set: {support_set.shape}")
        
        return model, support_set, support_labels, aksen_list
        
    except Exception as e:
        st.error(f"Error saat memuat resources: {str(e)}")
        if st.checkbox("Tampilkan detail error"):
            st.code(traceback.format_exc())
        return None, None, None, None

# ==========================================================
# 4. FUNGSI INFERENCE
# ==========================================================
def predict_accent(model, support_set, support_labels, query_feat, n_way):
    """Melakukan prediksi aksen menggunakan prototypical network"""
    try:
        # Siapkan query tensor
        query_tensor = np.expand_dims(query_feat, axis=0)
        
        # Validasi shape
        if query_tensor.shape[1:] != support_set.shape[1:]:
            st.error(f"Shape tidak cocok! Query: {query_tensor.shape}, Support: {support_set.shape}")
            return None
        
        # Lakukan inference
        logits = model.call(
            support_set,      # support set asli
            query_tensor,     # query audio
            support_labels,   # labels support set
            n_way,            # jumlah kelas
            training=False    # mode inference
        )
        
        # Konversi ke numpy jika perlu
        if hasattr(logits, 'numpy'):
            logits = logits.numpy()
        
        return logits[0]  # Return logits untuk satu sample query
        
    except Exception as e:
        st.error(f"Error dalam prediksi: {str(e)}")
        return None

# ==========================================================
# 5. ANTARMUKA UTAMA
# ==========================================================
def main():
    st.set_page_config(
        page_title="Sistem Deteksi Aksen Bahasa Indonesia",
        page_icon="🎙️",
        layout="centered"
    )
    
    st.title("🎙️ Sistem Deteksi Aksen Bahasa Indonesia")
    st.markdown("""
    Sistem ini menggunakan **Prototypical Network** untuk mengidentifikasi aksen bahasa Indonesia 
    berdasarkan rekaman suara. Model membandingkan audio input dengan support set dari 5 aksen utama.
    """)
    st.markdown("---")
    
    # Sidebar untuk informasi dan konfigurasi
    with st.sidebar:
        st.header("⚙️ Konfigurasi")
        debug_mode = st.checkbox("Mode Debug", value=False)
        
        st.header("ℹ️ Informasi Pembicara")
        usia = st.number_input("Usia", min_value=5, max_value=100, value=30, step=1)
        gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan", "Lainnya"])
        provinsi = st.selectbox("Asal Provinsi", [
            "DKI Jakarta", "Jawa Barat", "Jawa Tengah", 
            "Jawa Timur", "DI Yogyakarta", "Banten", "Lainnya"
        ])
        
        st.markdown("---")
        st.header("📊 Info Model")
        if st.button("ℹ️ Tampilkan Info Model"):
            st.info("""
            **Arsitektur Model:**
            - Prototypical Network
            - Input: MFCC 3-channel (40, 174, 3)
            - Embedding: CNN-based feature extractor
            - Jumlah kelas: 5 aksen utama
            """)
    
    # Load model dan support set
    with st.spinner("Memuat model dan support set..."):
        model, support_set, support_labels, aksen_list = load_model_and_support_set()
    
    if model is None:
        st.error("Tidak dapat melanjutkan tanpa model dan support set yang valid.")
        st.stop()
    
    # Debug information
    if debug_mode:
        with st.expander("🔍 Debug Information", expanded=False):
            st.write(f"**Model Info:**")
            st.write(f"- Type: {type(model)}")
            st.write(f"- Embedding model: {model.embedding}")
            
            st.write(f"\n**Support Set Info:**")
            st.write(f"- Shape: {support_set.shape}")
            st.write(f"- Min value: {support_set.min():.4f}")
            st.write(f"- Max value: {support_set.max():.4f}")
            st.write(f"- Mean: {support_set.mean():.4f}")
            
            st.write(f"\n**Support Labels:**")
            st.write(f"- Shape: {support_labels.shape}")
            st.write(f"- Unique labels: {np.unique(support_labels)}")
            st.write(f"- Label distribution:")
            unique, counts = np.unique(support_labels, return_counts=True)
            for label, count in zip(unique, counts):
                st.write(f"  - {aksen_list[label] if label < len(aksen_list) else label}: {count} samples")
    
    # Bagian utama: Upload audio
    st.subheader("📤 Upload Audio")
    st.markdown("""
    **Format yang didukung:** WAV (16-bit PCM, mono/stereo, 22.05kHz recommended)
    **Durasi optimal:** 3-10 detik
    """)
    
    audio_file = st.file_uploader(
        "Pilih file audio WAV",
        type=["wav"],
        help="Upload file audio dalam format WAV"
    )
    
    if audio_file is not None:
        # Tampilkan audio player
        st.audio(audio_file, format="audio/wav")
        
        # Tombol analisis
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            analyze_button = st.button(
                "🔍 Analisis Aksen",
                type="primary",
                use_container_width=True
            )
        
        if analyze_button:
            with st.spinner("Memproses audio..."):
                try:
                    # Simpan file audio sementara
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                        tmp_file.write(audio_file.getbuffer())
                        tmp_path = tmp_file.name
                    
                    # Ekstrak fitur MFCC
                    query_features = extract_mfcc_3channel(tmp_path)
                    
                    if query_features is None:
                        st.error("Gagal mengekstrak fitur dari audio.")
                        os.unlink(tmp_path)
                        return
                    
                    # Debug: tampilkan shape fitur
                    if debug_mode:
                        st.info(f"Shape fitur audio: {query_features.shape}")
                    
                    # Lakukan prediksi
                    n_way = len(aksen_list)
                    logits = predict_accent(model, support_set, support_labels, query_features, n_way)
                    
                    if logits is None:
                        st.error("Gagal melakukan prediksi.")
                        os.unlink(tmp_path)
                        return
                    
                    # Hitung probabilitas
                    probabilities = tf.nn.softmax(logits).numpy()
                    predicted_idx = np.argmax(probabilities)
                    confidence = probabilities[predicted_idx] * 100
                    
                    # Tampilkan hasil
                    st.markdown("---")
                    st.subheader("📊 Hasil Analisis")
                    
                    # Hasil utama dengan visual
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.metric(
                            label="🎯 Aksen Terdeteksi",
                            value=aksen_list[predicted_idx],
                            delta=f"{confidence:.1f}% confidence"
                        )
                    with col2:
                        # Progress bar untuk confidence
                        st.progress(float(confidence/100))
                        st.caption(f"Tingkat keyakinan: {confidence:.2f}%")
                    
                    # Informasi pembicara
                    st.markdown("---")
                    st.subheader("👤 Informasi Pembicara")
                    info_cols = st.columns(3)
                    with info_cols[0]:
                        st.metric("Usia", usia)
                    with info_cols[1]:
                        st.metric("Jenis Kelamin", gender)
                    with info_cols[2]:
                        st.metric("Asal Provinsi", provinsi)
                    
                    # Detail probabilitas semua kelas
                    st.markdown("---")
                    st.subheader("📈 Probabilitas per Aksen")
                    
                    # Buat dataframe untuk probabilitas
                    prob_df = pd.DataFrame({
                        "Aksen": aksen_list,
                        "Probabilitas": probabilities,
                        "Persentase": [f"{p*100:.2f}%" for p in probabilities]
                    })
                    
                    # Urutkan berdasarkan probabilitas
                    prob_df = prob_df.sort_values("Probabilitas", ascending=False)
                    
                    # Tampilkan tabel
                    st.dataframe(
                        prob_df[["Aksen", "Persentase", "Probabilitas"]],
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Visualisasi bar chart
                    st.bar_chart(
                        prob_df.set_index("Aksen")["Probabilitas"],
                        use_container_width=True
                    )
                    
                    # Interpretasi hasil
                    st.markdown("---")
                    st.subheader("📝 Interpretasi")
                    
                    if confidence > 70:
                        st.success(f"**Tingkat keyakinan tinggi** ({confidence:.1f}%). Audio memiliki karakteristik yang kuat menyerupai aksen **{aksen_list[predicted_idx]}**.")
                    elif confidence > 40:
                        st.warning(f"**Tingkat keyakinan sedang** ({confidence:.1f}%). Audio memiliki kemiripan dengan aksen **{aksen_list[predicted_idx]}**, namun ada kemungkinan pengaruh aksen lain.")
                    else:
                        st.info(f"**Tingkat keyakinan rendah** ({confidence:.1f}%). Karakteristik audio tidak jelas mendominasi satu aksen tertentu.")
                    
                    # Hapus file temporary
                    os.unlink(tmp_path)
                    
                except Exception as e:
                    st.error(f"Terjadi kesalahan: {str(e)}")
                    if debug_mode:
                        st.code(traceback.format_exc())
                    
                    # Cleanup jika ada file temporary
                    if 'tmp_path' in locals() and os.path.exists(tmp_path):
                        os.unlink(tmp_path)
    
    # Footer
    st.markdown("---")
    st.caption("""
    **Sistem Deteksi Aksen v1.0** | 
    Prototypical Network dengan MFCC 3-channel |
    Untuk keperluan penelitian skripsi
    """)

# ==========================================================
# 6. RUN APPLICATION
# ==========================================================
if __name__ == "__main__":
    main()
