import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tempfile
import os
import tensorflow as tf
import pickle

# ==========================================================
# INSPECT DAN REBUILD MODEL
# ==========================================================
def inspect_model_structure(model):
    """Inspect struktur model untuk debugging"""
    info = []
    info.append(f"Model Type: {type(model).__name__}")
    info.append(f"Model Class: {model.__class__.__name__}")
    
    # List semua attributes
    info.append("\n=== Attributes ===")
    for attr_name in sorted(dir(model)):
        if not attr_name.startswith('__'):
            try:
                attr_value = getattr(model, attr_name)
                attr_type = type(attr_value).__name__
                if not callable(attr_value):
                    info.append(f"  {attr_name}: {attr_type}")
            except:
                pass
    
    # Check layers
    if hasattr(model, 'layers'):
        info.append(f"\n=== Layers ({len(model.layers)}) ===")
        for i, layer in enumerate(model.layers):
            info.append(f"  [{i}] {layer.__class__.__name__}: {layer.name}")
    
    # Check weights
    if hasattr(model, 'weights'):
        info.append(f"\n=== Weights ({len(model.weights)}) ===")
        for w in model.weights[:5]:  # First 5
            info.append(f"  {w.name}: {w.shape}")
    
    return "\n".join(info)

def extract_embedding_from_model(model):
    """
    Ekstrak embedding model dari PrototypicalNetwork
    Coba berbagai cara untuk mendapatkan embedding layer
    """
    st.info("🔍 Mencari embedding layer...")
    
    # Method 1: Check _embedding_model
    if hasattr(model, '_embedding_model'):
        if model._embedding_model is not None and callable(model._embedding_model):
            st.success("✅ Found via _embedding_model")
            return model._embedding_model
    
    # Method 2: Check embedding
    if hasattr(model, 'embedding'):
        if model.embedding is not None and callable(model.embedding):
            st.success("✅ Found via embedding")
            return model.embedding
    
    # Method 3: Check layers
    if hasattr(model, 'layers') and len(model.layers) > 0:
        for i, layer in enumerate(model.layers):
            if isinstance(layer, (tf.keras.Model, tf.keras.Sequential)):
                st.success(f"✅ Found via layers[{i}]: {layer.name}")
                return layer
    
    # Method 4: Rebuild dari weights
    st.warning("⚠️ Embedding tidak ditemukan, mencoba rebuild dari weights...")
    
    # Inspect weights untuk mengetahui arsitektur
    if hasattr(model, 'weights'):
        st.info(f"📊 Model memiliki {len(model.weights)} weights")
        
        # Analisa shape weights untuk rekonstruksi
        weight_shapes = [w.shape for w in model.weights]
        st.code(f"Weight shapes: {weight_shapes[:10]}")
        
        # Rebuild embedding model berdasarkan weight shapes
        # Ini adalah arsitektur dari notebook (build_embedding_model)
        try:
            embedding_model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(40, 174, 3)),
                tf.keras.layers.MaxPooling2D((2,2)),
                tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2,2)),
                tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2,2)),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(128, activation='relu')
            ], name='rebuilt_embedding')
            
            # Build model
            dummy_input = np.random.rand(1, 40, 174, 3).astype(np.float32)
            _ = embedding_model(dummy_input)
            
            # Copy weights dari model asli
            st.info("🔄 Copying weights...")
            model_weight_idx = 0
            for layer in embedding_model.layers:
                if len(layer.weights) > 0:
                    try:
                        # Copy weights
                        new_weights = []
                        for _ in layer.weights:
                            if model_weight_idx < len(model.weights):
                                new_weights.append(model.weights[model_weight_idx].numpy())
                                model_weight_idx += 1
                        
                        if new_weights:
                            layer.set_weights(new_weights)
                            st.success(f"✅ Copied weights to {layer.name}")
                    except Exception as e:
                        st.warning(f"⚠️ Could not copy to {layer.name}: {e}")
            
            st.success("✅ Successfully rebuilt embedding model!")
            return embedding_model
            
        except Exception as e:
            st.error(f"❌ Failed to rebuild: {e}")
            return None
    
    return None

# ==========================================================
# CLASS PROTOTYPICAL NETWORK (SIMPLIFIED)
# ==========================================================
@tf.keras.utils.register_keras_serializable(package="Custom")
class PrototypicalNetwork(tf.keras.Model):
    def __init__(self, embedding_model=None, **kwargs):
        super(PrototypicalNetwork, self).__init__(**kwargs)
        self.embedding_model = embedding_model
    
    def call(self, support_set, query_set, support_labels, n_way, training=None):
        if self.embedding_model is None:
            raise ValueError("Embedding model is None!")
        
        # Embedding
        support_embeddings = self.embedding_model(support_set, training=training)
        query_embeddings = self.embedding_model(query_set, training=training)
        
        # Prototypes
        prototypes = []
        for i in range(n_way):
            class_mask = tf.equal(support_labels, i)
            class_embeddings = tf.boolean_mask(support_embeddings, class_mask)
            prototype = tf.reduce_mean(class_embeddings, axis=0)
            prototypes.append(prototype)
        
        prototypes = tf.stack(prototypes)
        
        # Distances
        query_expanded = tf.expand_dims(query_embeddings, 1)
        prototypes_expanded = tf.expand_dims(prototypes, 0)
        
        distances = tf.reduce_sum(
            tf.square(query_expanded - prototypes_expanded),
            axis=-1
        )
        
        return -distances
    
    def get_config(self):
        return super().get_config()

# ==========================================================
# LOAD MODEL WITH EXTRACTION
# ==========================================================
@st.cache_resource
def load_model_and_support():
    """Load dan extract embedding dari model"""
    try:
        # Load model
        custom_objects = {"PrototypicalNetwork": PrototypicalNetwork}
        loaded_model = tf.keras.models.load_model(
            "model_embedding_aksen.keras",
            custom_objects=custom_objects,
            compile=False
        )
        
        # Inspect structure
        with st.sidebar.expander("🔍 Model Structure", expanded=False):
            structure = inspect_model_structure(loaded_model)
            st.code(structure, language='text')
        
        # Extract embedding
        embedding = extract_embedding_from_model(loaded_model)
        
        if embedding is None:
            st.sidebar.error("❌ Gagal mengekstrak embedding!")
            return None, None, None
        
        # Buat model baru dengan embedding yang sudah diekstrak
        model = PrototypicalNetwork(embedding_model=embedding)
        
        # Load support set
        support_set = np.load('support_set.npy')
        support_labels = np.load('support_labels.npy')
        
        st.sidebar.success(f"✅ Support Set: {support_set.shape}")
        
        # Test model
        st.sidebar.info("🧪 Testing model...")
        dummy_support = support_set[:5]
        dummy_query = support_set[:1]
        dummy_labels = support_labels[:5]
        
        logits = model.call(
            tf.convert_to_tensor(dummy_support, dtype=tf.float32),
            tf.convert_to_tensor(dummy_query, dtype=tf.float32),
            tf.convert_to_tensor(dummy_labels, dtype=tf.int32),
            5
        )
        st.sidebar.success(f"✅ Test passed! Logits shape: {logits.shape}")
        
        return model, support_set, support_labels
        
    except Exception as e:
        st.sidebar.error(f"❌ Error: {str(e)}")
        import traceback
        st.sidebar.code(traceback.format_exc())
        return None, None, None

# ==========================================================
# HELPER FUNCTIONS
# ==========================================================
def extract_mfcc(audio_path, sr=22050, n_mfcc=40, max_len=174):
    """Extract MFCC 3-channel"""
    try:
        y, sr = librosa.load(audio_path, sr=sr, duration=10)
        
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        def pad_or_truncate(arr, max_len):
            if arr.shape[1] < max_len:
                pad_width = max_len - arr.shape[1]
                arr = np.pad(arr, ((0, 0), (0, pad_width)), mode='constant')
            else:
                arr = arr[:, :max_len]
            return arr
        
        mfcc = pad_or_truncate(mfcc, max_len)
        mfcc_delta = pad_or_truncate(mfcc_delta, max_len)
        mfcc_delta2 = pad_or_truncate(mfcc_delta2, max_len)
        
        mfcc_3channel = np.stack([mfcc, mfcc_delta, mfcc_delta2], axis=-1)
        
        return mfcc_3channel.astype(np.float32)
        
    except Exception as e:
        st.error(f"Error extracting MFCC: {e}")
        return None

@st.cache_data
def load_metadata():
    if os.path.exists("metadata.csv"):
        return pd.read_csv("metadata.csv")
    return None

@st.cache_data
def load_label_encoder():
    try:
        with open('label_encoder.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        le.classes_ = np.array(['Betawi', 'Jawa Tengah', 'Jawa Timur', 'Sunda', 'Yogyakarta'])
        return le

# ==========================================================
# PREDICT FUNCTION
# ==========================================================
def predict_accent(audio_path, model, support_set, support_labels, n_way, label_encoder):
    try:
        mfcc_feat = extract_mfcc(audio_path)
        
        if mfcc_feat is None:
            return "❌ Error extracting features"
        
        query_features = np.expand_dims(mfcc_feat, axis=0).astype(np.float32)
        
        support_tensor = tf.convert_to_tensor(support_set, dtype=tf.float32)
        query_tensor = tf.convert_to_tensor(query_features, dtype=tf.float32)
        support_labels_tensor = tf.convert_to_tensor(support_labels, dtype=tf.int32)
        
        logits = model.call(
            support_tensor,
            query_tensor,
            support_labels_tensor,
            n_way
        )
        
        pred_index = tf.argmax(logits, axis=1).numpy()[0]
        probs = tf.nn.softmax(logits, axis=1).numpy()[0]
        
        pred_label = label_encoder.inverse_transform([pred_index])[0]
        confidence = probs[pred_index] * 100
        
        detail_lines = []
        for i, (cls, prob) in enumerate(zip(label_encoder.classes_, probs)):
            marker = "👉 " if i == pred_index else "   "
            detail_lines.append(f"{marker}{cls}: {prob*100:.2f}%")
        
        detail = "\n".join(detail_lines)
        
        return f"{pred_label} ({confidence:.1f}%)\n\n📊 Detail:\n{detail}"
        
    except Exception as e:
        import traceback
        st.error(traceback.format_exc())
        return f"❌ Error: {str(e)}"

# ==========================================================
# STREAMLIT UI
# ==========================================================
st.set_page_config(page_title="Deteksi Aksen", page_icon="🎙️", layout="wide")

st.title("🎙️ Sistem Deteksi Aksen Indonesia")
st.write("Aplikasi berbasis *Prototypical Network* untuk klasifikasi aksen daerah.")
st.divider()

model, support_set, support_labels = load_model_and_support()
metadata = load_metadata()
label_encoder = load_label_encoder()

with st.sidebar:
    st.header("🛸 Status")
    if model is not None:
        st.success("🤖 Model: Ready")
    if metadata is not None:
        st.info(f"📁 Metadata: {len(metadata)} records")

col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("📥 Input Audio")
    audio_file = st.file_uploader("Upload (.wav, .mp3)", type=["wav", "mp3"])
    
    if audio_file:
        st.audio(audio_file)
        
        if st.button("🚀 Analisis", type="primary", use_container_width=True):
            if model and support_set is not None:
                with st.spinner("Analyzing..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(audio_file.getbuffer())
                        tmp_path = tmp.name
                    
                    n_way = len(label_encoder.classes_)
                    hasil = predict_accent(tmp_path, model, support_set, support_labels, n_way, label_encoder)
                    
                    user_info = None
                    if metadata is not None:
                        match = metadata[metadata['file_name'] == audio_file.name]
                        if not match.empty:
                            user_info = match.iloc[0].to_dict()
                    
                    with col2:
                        st.subheader("📊 Hasil")
                        with st.container(border=True):
                            st.text(hasil)
                        
                        st.divider()
                        st.subheader("💎 Info Pembicara")
                        if user_info:
                            province_map = {
                                'DKI Jakarta': 'Betawi',
                                'Jawa Barat': 'Sunda',
                                'Jawa Tengah': 'Jawa Tengah',
                                'Jawa Timur': 'Jawa Timur',
                                'Yogyakarta': 'Yogyakarta',
                                'D.I YogyaKarta': 'Yogyakarta'
                            }
                            
                            actual_province = user_info.get('provinsi', '-')
                            actual_accent = province_map.get(actual_province, '-')
                            
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("Usia", f"{user_info.get('usia', '-')} Tahun")
                                st.metric("Gender", user_info.get('gender', '-'))
                            with col_b:
                                st.metric("Provinsi", actual_province)
                                st.metric("Aksen Sebenarnya", actual_accent)
                            
                            if actual_accent != '-':
                                predicted = hasil.split('(')[0].strip()
                                if actual_accent == predicted:
                                    st.success("🎯 BENAR!")
                                else:
                                    st.warning(f"⚠️ Seharusnya: {actual_accent}")
                        else:
                            st.info("File tidak di metadata")
                    
                    os.unlink(tmp_path)
            else:
                st.error("Model tidak ready")
    else:
        with col2:
            st.info("👈 Upload audio")
