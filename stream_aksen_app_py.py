import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tempfile
import os
import tensorflow as tf
import traceback
import pickle
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ==========================================================
# 1. KONFIGURASI AWAL
# ==========================================================
# Set page config
st.set_page_config(
    page_title="Sistem Deteksi Aksen & Analisis Suara Multi-Task",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #374151;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #E5E7EB;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #F9FAFB;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .result-card {
        background-color: #F0F9FF;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #BFDBFE;
        margin-bottom: 1rem;
    }
    .warning-card {
        background-color: #FEF3C7;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #F59E0B;
        margin-bottom: 1rem;
    }
    .success-card {
        background-color: #D1FAE5;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #10B981;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================================
# 2. DEFINISI MODEL MULTI-TASK
# ==========================================================
@tf.keras.utils.register_keras_serializable(package="Custom")
class MultiTaskPrototypicalNetwork(tf.keras.Model):
    def __init__(self, 
                 accent_embedding_model=None,
                 demographic_model=None,
                 **kwargs):
        super(MultiTaskPrototypicalNetwork, self).__init__(**kwargs)
        
        # Model untuk setiap task
        self.accent_embedding = accent_embedding_model
        self.demographic_embedding = demographic_model
        
        # Heads untuk output
        self.age_head = tf.keras.layers.Dense(1, activation='relu', name='age_output')
        self.gender_head = tf.keras.layers.Dense(2, activation='softmax', name='gender_output')
        
    def call(self, inputs, training=False):
        # Unpack inputs: [accent_features, demographic_features]
        accent_features = inputs[0]
        demo_features = inputs[1]
        
        # Task 1: Aksen (embedding untuk prototypical)
        accent_embeddings = self.accent_embedding(accent_features)
        
        # Task 2: Demografik (usia & gender)
        demo_embeddings = self.demographic_embedding(demo_features)
        
        # Output predictions
        age_pred = self.age_head(demo_embeddings)
        gender_pred = self.gender_head(demo_embeddings)
        
        return accent_embeddings, age_pred, gender_pred
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "accent_embedding_model": tf.keras.layers.serialize(self.accent_embedding),
            "demographic_model": tf.keras.layers.serialize(self.demographic_embedding)
        })
        return config

# ==========================================================
# 3. FUNGSI EKSTRAKSI FITUR MULTI-TASK
# ==========================================================
def extract_multitask_features(file_path, sr=22050, n_mfcc=40, max_len=174):
    """Ekstrak fitur untuk semua tasks: aksen, usia, gender"""
    try:
        # Load audio
        y, sr = librosa.load(file_path, sr=sr)
        y = librosa.util.normalize(y)
        
        # 1. FITUR UNTUK AKSEN (MFCC 3-channel)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=2048, hop_length=512)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Padding atau truncating
        if mfcc.shape[1] < max_len:
            pad_width = max_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
            delta = np.pad(delta, ((0, 0), (0, pad_width)), mode='constant')
            delta2 = np.pad(delta2, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc, delta, delta2 = mfcc[:, :max_len], delta[:, :max_len], delta2[:, :max_len]
        
        accent_features = np.stack([mfcc, delta, delta2], axis=-1)  # (40, 174, 3)
        
        # 2. FITUR UNTUK DEMOGRAFIK (usia & gender)
        # Statistik MFCC
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        mfcc_skew = pd.Series(mfcc.flatten()).skew()
        mfcc_kurt = pd.Series(mfcc.flatten()).kurtosis()
        
        # Fitur temporal
        rms = librosa.feature.rms(y=y).mean()
        zcr = librosa.feature.zero_crossing_rate(y=y).mean()
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
        
        # Fitur pitch untuk gender
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitches[magnitudes > np.max(magnitudes) * 0.1]
        pitch_mean = np.mean(pitch_values) if len(pitch_values) > 0 else 0
        pitch_std = np.std(pitch_values) if len(pitch_values) > 0 else 0
        
        # Formants (untuk gender)
        try:
            # Menggunakan LPC untuk estimasi formants
            lpc_coeff = librosa.lpc(y, order=8)
            roots = np.roots(lpc_coeff)
            roots = roots[np.imag(roots) >= 0]
            angz = np.arctan2(np.imag(roots), np.real(roots))
            formants = angz * (sr / (2 * np.pi))
            formants = np.sort(formants)
            f1 = formants[0] if len(formants) > 0 else 0
            f2 = formants[1] if len(formants) > 1 else 0
        except:
            f1, f2 = 0, 0
        
        # Gabungkan semua fitur demografik
        demographic_features = np.array([
            *mfcc_mean[:10],  # Ambil 10 pertama
            mfcc_std.mean(),
            mfcc_skew,
            mfcc_kurt,
            rms,
            zcr,
            spectral_centroid,
            spectral_bandwidth,
            pitch_mean,
            pitch_std,
            f1,
            f2,
            len(y) / sr  # durasi
        ])
        
        # 3. FITUR TAMBAHAN UNTUK ANALISIS
        analysis_features = {
            'duration': len(y) / sr,
            'energy': rms,
            'pitch_mean': pitch_mean,
            'pitch_std': pitch_std,
            'zcr': zcr,
            'spectral_centroid': spectral_centroid
        }
        
        return accent_features, demographic_features, analysis_features
        
    except Exception as e:
        st.error(f"Error dalam ekstraksi fitur: {e}")
        return None, None, None

# ==========================================================
# 4. LOAD MODEL & RESOURCES
# ==========================================================
@st.cache_resource
def load_resources():
    """Memuat semua resources yang diperlukan"""
    resources = {
        'model': None,
        'support_set': None,
        'support_labels': None,
        'scaler': None,
        'label_encoder': None,
        'config': None
    }
    
    # Daftar file yang diperlukan
    required_files = {
        'model': 'multitask_model.keras',
        'support_set': 'support_set.npy',
        'support_labels': 'support_labels.npy',
        'scaler': 'demographic_scaler.pkl',
        'config': 'model_config.json'
    }
    
    # Cek file yang ada
    missing_files = []
    for key, filename in required_files.items():
        if not os.path.exists(filename):
            missing_files.append(filename)
    
    if missing_files:
        st.warning(f"⚠️ File berikut tidak ditemukan: {', '.join(missing_files)}")
        
        # Buat resources dummy untuk testing
        if st.checkbox("Gunakan mode demo dengan data dummy?"):
            st.info("Membuat model dan data dummy untuk testing...")
            
            # Buat model dummy
            accent_embedding = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(40, 174, 3)),
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation='relu')
            ])
            
            demographic_model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(25,)),  # 25 fitur demografik
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(32, activation='relu')
            ])
            
            model = MultiTaskPrototypicalNetwork(
                accent_embedding_model=accent_embedding,
                demographic_model=demographic_model
            )
            
            # Buat support set dummy
            resources['support_set'] = np.random.randn(15, 40, 174, 3).astype(np.float32) * 0.1
            resources['support_labels'] = np.random.choice([0, 1, 2, 3, 4], size=15)
            
            # Buat scaler dummy
            class DummyScaler:
                def transform(self, X):
                    return X
            resources['scaler'] = DummyScaler()
            
            resources['label_encoder'] = {
                0: "Sunda", 1: "Jawa Tengah", 2: "Jawa Timur", 
                3: "Yogyakarta", 4: "Betawi"
            }
            
            resources['model'] = model
            st.success("Mode demo aktif! Hasil adalah prediksi dummy.")
            return resources
            
        return None
    
    try:
        # 1. Load model
        custom_objects = {"MultiTaskPrototypicalNetwork": MultiTaskPrototypicalNetwork}
        model = tf.keras.models.load_model('multitask_model.keras', 
                                          custom_objects=custom_objects,
                                          compile=False)
        model.compile(optimizer='adam', loss='mse')
        resources['model'] = model
        
        # 2. Load support set
        support_set = np.load('support_set.npy', allow_pickle=True)
        support_labels = np.load('support_labels.npy', allow_pickle=True)
        
        # Validasi shape
        if support_set.shape[1:] != (40, 174, 3):
            st.error(f"Shape support set tidak valid: {support_set.shape}")
            return None
        
        resources['support_set'] = support_set
        resources['support_labels'] = support_labels
        
        # 3. Load scaler untuk fitur demografik
        with open('demographic_scaler.pkl', 'rb') as f:
            resources['scaler'] = pickle.load(f)
        
        # 4. Load label encoder (atau buat mapping)
        resources['label_encoder'] = {
            0: "Sunda", 1: "Jawa Tengah", 2: "Jawa Timur", 
            3: "Yogyakarta", 4: "Betawi"
        }
        
        st.success("✅ Semua resources berhasil dimuat!")
        return resources
        
    except Exception as e:
        st.error(f"❌ Error loading resources: {str(e)}")
        if st.checkbox("Tampilkan traceback error"):
            st.code(traceback.format_exc())
        return None

# ==========================================================
# 5. FUNGSI PREDIKSI MULTI-TASK
# ==========================================================
def predict_multitask(resources, audio_path, user_info=None):
    """Melakukan prediksi multi-task: aksen, usia, gender"""
    
    if resources is None:
        return None
    
    try:
        # 1. Ekstrak fitur
        accent_feat, demo_feat, analysis_feat = extract_multitask_features(audio_path)
        
        if accent_feat is None:
            return None
        
        # 2. Normalisasi fitur demografik
        demo_feat_scaled = resources['scaler'].transform(demo_feat.reshape(1, -1))
        
        # 3. Siapkan input untuk model
        accent_input = np.expand_dims(accent_feat, axis=0)  # (1, 40, 174, 3)
        demo_input = demo_feat_scaled  # (1, n_features)
        
        # 4. Prediksi menggunakan model
        accent_embeddings, age_pred, gender_pred = resources['model'](
            [accent_input, demo_input], 
            training=False
        )
        
        # 5. Prototypical matching untuk aksen
        support_embeddings = resources['model'].accent_embedding(resources['support_set'])
        
        # Hitung jarak ke prototype setiap kelas
        n_way = len(resources['label_encoder'])
        distances = []
        
        for i in range(n_way):
            mask = resources['support_labels'] == i
            if np.any(mask):
                class_embeddings = support_embeddings[mask]
                prototype = tf.reduce_mean(class_embeddings, axis=0)
                distance = tf.norm(accent_embeddings[0] - prototype)
                distances.append(distance.numpy())
            else:
                distances.append(1000)  # nilai besar jika kelas tidak ada
        
        # Konversi distance ke similarity score
        distances = np.array(distances)
        epsilon = 1e-10
        similarities = 1 / (distances + epsilon)
        similarities = similarities / similarities.sum()  # normalisasi ke probabilitas
        
        # Prediksi aksen
        accent_idx = np.argmax(similarities)
        accent_confidence = similarities[accent_idx] * 100
        
        # 6. Process usia dan gender predictions
        age_value = float(age_pred.numpy()[0][0])
        
        # Adjust age jika ada info dari user
        if user_info and 'usia' in user_info:
            user_age = user_info['usia']
            # Weighted average antara prediksi model dan input user
            age_value = (age_value * 0.7) + (user_age * 0.3)
        
        gender_probs = gender_pred.numpy()[0]
        gender_idx = np.argmax(gender_probs)
        gender_confidence = gender_probs[gender_idx] * 100
        gender_label = "Laki-laki" if gender_idx == 0 else "Perempuan"
        
        # Adjust gender jika ada info dari user
        if user_info and 'gender' in user_info:
            if user_info['gender'] == "Laki-laki" and gender_label == "Perempuan":
                gender_confidence *= 0.8  # Kurangi confidence jika berbeda
            elif user_info['gender'] == "Perempuan" and gender_label == "Laki-laki":
                gender_confidence *= 0.8
        
        # 7. Hitung confidence usia (semakin dekat dengan rata-rata, semakin tinggi)
        # Asumsi usia normal 20-60 tahun
        age_confidence = 100 - min(50, abs(age_value - 40) * 2)
        age_confidence = max(30, age_confidence)  # Minimum 30%
        
        # 8. Return results
        return {
            'accent': {
                'prediction': accent_idx,
                'label': resources['label_encoder'][accent_idx],
                'probabilities': similarities,
                'confidence': float(accent_confidence)
            },
            'age': {
                'prediction': int(round(age_value)),
                'confidence': float(age_confidence),
                'raw': float(age_value)
            },
            'gender': {
                'prediction': gender_label,
                'confidence': float(gender_confidence),
                'probabilities': gender_probs.tolist()
            },
            'analysis': analysis_feat,
            'features': {
                'accent_shape': accent_feat.shape,
                'demo_features': len(demo_feat)
            },
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
    except Exception as e:
        st.error(f"Error dalam prediksi: {str(e)}")
        return None

# ==========================================================
# 6. FUNGSI VISUALISASI
# ==========================================================
def create_visualizations(results):
    """Membuat visualisasi untuk hasil prediksi"""
    visualizations = {}
    
    try:
        # 1. Bar chart untuk probabilitas aksen
        accent_labels = ["Sunda", "Jawa Tengah", "Jawa Timur", "Yogyakarta", "Betawi"]
        accent_probs = results['accent']['probabilities']
        
        fig_accents = go.Figure(data=[
            go.Bar(
                x=accent_labels,
                y=accent_probs * 100,
                marker_color=['#3B82F6' if i == results['accent']['prediction'] 
                             else '#9CA3AF' for i in range(len(accent_labels))],
                text=[f'{p*100:.1f}%' for p in accent_probs],
                textposition='auto',
            )
        ])
        
        fig_accents.update_layout(
            title="Probabilitas Aksen",
            xaxis_title="Aksen",
            yaxis_title="Probabilitas (%)",
            yaxis_range=[0, 100],
            template="plotly_white"
        )
        visualizations['accent_chart'] = fig_accents
        
        # 2. Gauge chart untuk confidence
        fig_gauges = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Aksen', 'Usia', 'Gender'),
            specs=[[{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]]
        )
        
        # Gauge aksen
        fig_gauges.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=results['accent']['confidence'],
                title={'text': "Confidence"},
                domain={'row': 0, 'column': 0},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#3B82F6"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "gray"},
                        {'range': [80, 100], 'color': "darkgray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ),
            row=1, col=1
        )
        
        # Gauge usia
        fig_gauges.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=results['age']['confidence'],
                title={'text': "Confidence"},
                domain={'row': 0, 'column': 1},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#10B981"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "gray"},
                        {'range': [80, 100], 'color': "darkgray"}
                    ]
                }
            ),
            row=1, col=2
        )
        
        # Gauge gender
        fig_gauges.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=results['gender']['confidence'],
                title={'text': "Confidence"},
                domain={'row': 0, 'column': 2},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#8B5CF6"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "gray"},
                        {'range': [80, 100], 'color': "darkgray"}
                    ]
                }
            ),
            row=1, col=3
        )
        
        fig_gauges.update_layout(height=300, margin=dict(t=50, b=10))
        visualizations['confidence_gauges'] = fig_gauges
        
        # 3. Radar chart untuk fitur audio
        if 'analysis' in results:
            analysis = results['analysis']
            features = ['Energy', 'Pitch Mean', 'Pitch Var', 'ZCR', 'Spectral Centroid']
            values = [
                analysis['energy'] * 100,
                min(100, analysis['pitch_mean'] / 5),
                min(100, analysis['pitch_std'] * 10),
                analysis['zcr'] * 1000,
                min(100, analysis['spectral_centroid'] / 100)
            ]
            
            fig_radar = go.Figure(data=go.Scatterpolar(
                r=values,
                theta=features,
                fill='toself',
                line_color='#F59E0B'
            ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                showlegend=False,
                title="Analisis Fitur Audio",
                height=300
            )
            visualizations['radar_chart'] = fig_radar
        
        # 4. Gender probability pie chart
        gender_labels = ['Laki-laki', 'Perempuan']
        gender_probs = results['gender']['probabilities']
        
        fig_gender = go.Figure(data=[go.Pie(
            labels=gender_labels,
            values=gender_probs,
            hole=.3,
            marker_colors=['#3B82F6', '#EC4899']
        )])
        
        fig_gender.update_layout(
            title="Probabilitas Gender",
            height=300
        )
        visualizations['gender_pie'] = fig_gender
        
    except Exception as e:
        st.warning(f"Tidak dapat membuat visualisasi: {e}")
    
    return visualizations

# ==========================================================
# 7. FUNGSI UTAMA STREAMLIT
# ==========================================================
def main():
    # Header
    st.markdown('<h1 class="main-header">🎙️ Sistem Analisis Suara Multi-Task</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; color: #6B7280; margin-bottom: 2rem;'>
    Deteksi aksen bahasa Indonesia, estimasi usia, dan identifikasi gender dari rekaman suara
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Konfigurasi")
        
        # Mode operasi
        app_mode = st.radio(
            "Pilih Mode:",
            ["Analisis Audio", "Info Model", "Panduan Penggunaan"]
        )
        
        if app_mode == "Analisis Audio":
            st.markdown("---")
            st.markdown("### 👤 Informasi Pembicara")
            
            # Input informasi pembicara
            col1, col2 = st.columns(2)
            with col1:
                usia = st.number_input("Usia", min_value=5, max_value=100, value=30, step=1)
            with col2:
                gender = st.selectbox("Gender", ["Laki-laki", "Perempuan"])
            
            provinsi = st.selectbox("Provinsi Asal", [
                "DKI Jakarta", "Jawa Barat", "Jawa Tengah", 
                "Jawa Timur", "DI Yogyakarta", "Banten", "Lainnya"
            ])
            
            st.markdown("---")
            st.markdown("### 🔧 Pengaturan Lanjutan")
            debug_mode = st.checkbox("Mode Debug", value=False)
            save_results = st.checkbox("Simpan Hasil Analisis", value=False)
            
            user_info = {
                'usia': usia,
                'gender': gender,
                'provinsi': provinsi
            }
    
    # Load resources sekali di awal
    if 'resources' not in st.session_state:
        with st.spinner("Memuat model dan resources..."):
            st.session_state.resources = load_resources()
    
    resources = st.session_state.resources
    
    # Main content berdasarkan mode
    if app_mode == "Analisis Audio":
        if resources is None:
            st.error("Tidak dapat memuat resources yang diperlukan.")
            if st.button("Coba Muat Ulang"):
                st.session_state.resources = load_resources()
                st.rerun()
            return
        
        # Upload section
        st.markdown('<h2 class="sub-header">📤 Upload Audio untuk Analisis</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            audio_file = st.file_uploader(
                "Pilih file audio (.wav)",
                type=["wav", "mp3"],
                help="Format yang disarankan: WAV 16-bit, 22.05kHz"
            )
        
        with col2:
            st.markdown("""
            **Format optimal:**
            - WAV, 16-bit PCM
            - 22.05 kHz sample rate
            - Mono channel
            - Durasi: 3-10 detik
            """)
        
        if audio_file is not None:
            # Display audio player
            st.audio(audio_file, format="audio/wav")
            
            # Tombol analisis
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                analyze_btn = st.button(
                    "🚀 Mulai Analisis Multi-Task",
                    type="primary",
                    use_container_width=True,
                    disabled=audio_file is None
                )
            
            if analyze_btn:
                with st.spinner("Melakukan analisis multi-task..."):
                    try:
                        # Save temporary file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                            tmp_file.write(audio_file.getbuffer())
                            tmp_path = tmp_file.name
                        
                        # Perform prediction
                        results = predict_multitask(resources, tmp_path, user_info)
                        
                        if results is None:
                            st.error("Gagal melakukan analisis.")
                            os.unlink(tmp_path)
                            return
                        
                        # Save to session state
                        st.session_state.last_results = results
                        st.session_state.audio_file = audio_file.name
                        
                        # Cleanup
                        os.unlink(tmp_path)
                        
                        # Display results
                        st.markdown("---")
                        st.markdown('<h2 class="sub-header">📊 Hasil Analisis Multi-Task</h2>', unsafe_allow_html=True)
                        
                        # Results cards
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            with st.container():
                                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                                st.metric(
                                    label="🎯 **Aksen Terdeteksi**",
                                    value=results['accent']['label'],
                                    delta=f"{results['accent']['confidence']:.1f}% confidence"
                                )
                                st.caption(f"Prediction confidence: {results['accent']['confidence']:.1f}%")
                                st.markdown('</div>', unsafe_allow_html=True)
                        
                        with col2:
                            with st.container():
                                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                                st.metric(
                                    label="👤 **Gender**",
                                    value=results['gender']['prediction'],
                                    delta=f"{results['gender']['confidence']:.1f}% confidence"
                                )
                                st.caption(f"Male: {results['gender']['probabilities'][0]*100:.1f}%, "
                                         f"Female: {results['gender']['probabilities'][1]*100:.1f}%")
                                st.markdown('</div>', unsafe_allow_html=True)
                        
                        with col3:
                            with st.container():
                                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                                st.metric(
                                    label="📅 **Estimasi Usia**",
                                    value=f"{results['age']['prediction']} tahun",
                                    delta=f"{results['age']['confidence']:.1f}% confidence"
                                )
                                st.caption(f"Raw prediction: {results['age']['raw']:.1f} tahun")
                                st.markdown('</div>', unsafe_allow_html=True)
                        
                        # User information
                        st.markdown('<div class="result-card">', unsafe_allow_html=True)
                        st.markdown("### 👤 Informasi Pembicara yang Diinput")
                        user_cols = st.columns(3)
                        with user_cols[0]:
                            st.info(f"**Usia:** {user_info['usia']} tahun")
                        with user_cols[1]:
                            st.info(f"**Gender:** {user_info['gender']}")
                        with user_cols[2]:
                            st.info(f"**Provinsi:** {user_info['provinsi']}")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Visualizations
                        st.markdown("---")
                        st.markdown('<h3 class="sub-header">📈 Visualisasi Hasil</h3>', unsafe_allow_html=True)
                        
                        viz = create_visualizations(results)
                        
                        # Row 1: Accent probabilities and confidence gauges
                        viz_col1, viz_col2 = st.columns([2, 1])
                        with viz_col1:
                            if 'accent_chart' in viz:
                                st.plotly_chart(viz['accent_chart'], use_container_width=True)
                        
                        with viz_col2:
                            if 'confidence_gauges' in viz:
                                st.plotly_chart(viz['confidence_gauges'], use_container_width=True)
                        
                        # Row 2: Gender pie and radar chart
                        viz_col3, viz_col4 = st.columns(2)
                        with viz_col3:
                            if 'gender_pie' in viz:
                                st.plotly_chart(viz['gender_pie'], use_container_width=True)
                        
                        with viz_col4:
                            if 'radar_chart' in viz:
                                st.plotly_chart(viz['radar_chart'], use_container_width=True)
                        
                        # Detailed tables
                        st.markdown("---")
                        st.markdown('<h3 class="sub-header">📋 Detail Probabilitas</h3>', unsafe_allow_html=True)
                        
                        # Accent probabilities table
                        accent_df = pd.DataFrame({
                            'Aksen': ["Sunda", "Jawa Tengah", "Jawa Timur", "Yogyakarta", "Betawi"],
                            'Probabilitas': results['accent']['probabilities'],
                            'Persentase': [f"{p*100:.2f}%" for p in results['accent']['probabilities']]
                        }).sort_values('Probabilitas', ascending=False)
                        
                        st.dataframe(
                            accent_df,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Aksen": st.column_config.TextColumn("Aksen"),
                                "Probabilitas": st.column_config.ProgressColumn(
                                    "Probabilitas",
                                    format="%.3f",
                                    min_value=0,
                                    max_value=1
                                ),
                                "Persentase": st.column_config.TextColumn("Persentase")
                            }
                        )
                        
                        # Technical details (debug mode)
                        if debug_mode:
                            st.markdown("---")
                            st.markdown('<h3 class="sub-header">🔍 Informasi Teknis</h3>', unsafe_allow_html=True)
                            
                            with st.expander("Detail Fitur dan Hasil", expanded=False):
                                st.json({
                                    'timestamp': results['timestamp'],
                                    'accent_features_shape': str(results['features']['accent_shape']),
                                    'demographic_features_count': results['features']['demo_features'],
                                    'analysis_features': results['analysis']
                                })
                        
                        # Save results option
                        if save_results:
                            st.markdown("---")
                            st.markdown('<h3 class="sub-header">💾 Simpan Hasil</h3>', unsafe_allow_html=True)
                            
                            result_data = {
                                'audio_file': audio_file.name,
                                'timestamp': results['timestamp'],
                                'user_info': user_info,
                                'predictions': {
                                    'accent': results['accent'],
                                    'age': results['age'],
                                    'gender': results['gender']
                                }
                            }
                            
                            # Convert to JSON for download
                            import json
                            result_json = json.dumps(result_data, indent=2, default=str)
                            
                            st.download_button(
                                label="📥 Download Hasil (JSON)",
                                data=result_json,
                                file_name=f"analisis_suara_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )
                        
                    except Exception as e:
                        st.error(f"Error selama analisis: {str(e)}")
                        if debug_mode:
                            st.code(traceback.format_exc())
    
    elif app_mode == "Info Model":
        st.markdown('<h2 class="sub-header">ℹ️ Informasi Model</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        ### Arsitektur Model Multi-Task
        
        Model ini menggunakan pendekatan **Multi-Task Learning** dengan arsitektur sebagai berikut:
        
        #### 1. **Task Aksen (Prototypical Network)**
        - **Input:** MFCC 3-channel (40 coefficients × 174 frames × 3 channels)
        - **Feature Extractor:** CNN dengan 2 layer convolutional
        - **Output:** Embedding vector untuk prototypical matching
        
        #### 2. **Task Demografik (Usia & Gender)**
        - **Input:** 25 fitur audio (statistik MFCC, pitch, formants, dll)
        - **Feature Extractor:** MLP dengan 2 layer fully connected
        - **Outputs:**
          - Usia: Regresi (single neuron dengan aktivasi ReLU)
          - Gender: Klasifikasi biner (
