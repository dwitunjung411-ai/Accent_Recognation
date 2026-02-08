import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tempfile
import os
import tensorflow as tf

# ==========================================================
# LOAD MODEL
# ==========================================================
@st.cache(allow_output_mutation=True)
def load_embedding_model():
    return tf.keras.models.load_model("model_embedding_aksen.keras", compile=False)

# ==========================================================
# LOAD CENTROID
# ==========================================================
@st.cache(allow_output_mutation=True)
def load_centroids():
    return np.load("accent_centroids.npy", allow_pickle=True).item()

# ==========================================================
# LOAD METADATA
# ==========================================================
@st.cache(allow_output_mutation=True)
def load_metadata():
    if os.path.exists("metadata.csv"):
        return pd.read_csv("metadata.csv")
    return None

# ==========================================================
# MFCC
# ==========================================================
def extract_mfcc(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

# ==========================================================
# PREDIKSI AKSEN
# ==========================================================
def predict_accent(audio_path, model, centroids):
    mfcc = extract_mfcc(audio_path)
    mfcc = np.expand_dims(mfcc, axis=0)

    embedding = model.predict(mfcc)
    embedding = np.squeeze(embedding)

    distances = {}
    for label, centroid in centroids.ite
