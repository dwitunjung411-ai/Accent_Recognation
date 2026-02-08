import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tempfile
import os
import tensorflow as tf
import pickle

# Load model dan label encoder
@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model('model_accent_simple.keras', compile=False)
    
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    return model, label_encoder

model, label_encoder = load_resources()

# Prediksi
def predict_accent(audio_path):
    # Extract MFCC
    y, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    
    # Predict
    X = np.expand_dims(mfcc_mean, axis=0)
    probabilities = model.predict(X, verbose=0)[0]
    
    # Get result
    predicted_idx = np.argmax(probabilities)
    predicted_class = label_encoder.classes_[predicted_idx]
    confidence = probabilities[predicted_idx] * 100
    
    # Detail
    detail = "\n".join([
        f"{'👉 ' if i == predicted_idx else '   '}{cls}: {prob*100:.2f}%" 
        for i, (cls, prob) in enumerate(zip(label_encoder.classes_, probabilities))
    ])
    
    return f"{predicted_class} ({confidence:.1f}%)\n\n📊 Detail:\n{detail}"

# UI (sama seperti sebelumnya)
st.title("🎙️ Deteksi Aksen Indonesia")
audio = st.file_uploader("Upload Audio", type=["wav", "mp3"])

if audio and st.button("Analisis"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio.getbuffer())
        result = predict_accent(tmp.name)
        st.success(result)
        os.unlink(tmp.name)
