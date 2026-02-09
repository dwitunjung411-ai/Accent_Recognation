import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import librosa
import os
import joblib

# ===============================
# CONFIG
# ===============================
SR = 22050
N_MFCC = 40
MAX_LEN = 174

st.set_page_config(page_title="Deteksi Aksen & Metadata", layout="centered")

# ===============================
# LOAD ASSETS
# ===============================
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model(
        "model_ditek.keras",
        custom_objects={"PrototypicalNetwork": tf.keras.Model}
    )

    le_aksen = joblib.load("model/le_aksen.pkl")
    le_gender = joblib.load("model/le_gender.pkl")
    le_provinsi = joblib.load("model/le_provinsi.pkl")
    scaler_usia = joblib.load("model/scaler_usia.pkl")
    ohe = joblib.load("model/ohe.pkl")

    metadata = pd.read_csv("metadata.csv")

    return model, le_aksen, le_gender, le_provinsi, scaler_usia, ohe, metadata


# ===============================
# FEATURE EXTRACTION
# ===============================
def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, sr=SR)
    y = librosa.util.normalize(y)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=2048,
        hop_length=512
    )

    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    if mfcc.shape[1] < MAX_LEN:
        pad = MAX_LEN - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad)))
        delta = np.pad(delta, ((0, 0), (0, pad)))
        delta2 = np.pad(delta2, ((0, 0), (0, pad)))
    else:
        mfcc = mfcc[:, :MAX_LEN]
        delta = delta[:, :MAX_LEN]
        delta2 = delta2[:, :MAX_LEN]

    return np.stack([mfcc, delta, delta2], axis=-1)


# ===============================
# STREAMLIT UI
# ===============================
st.title("🎙️ Deteksi Aksen, Usia, Gender & Provinsi")
st.write("Upload audio `.wav` → sistem akan membaca metadata & mendeteksi aksen")

uploaded_file = st.file_uploader("Upload Audio WAV", type=["wav"])

if uploaded_file:
    temp_path = "temp.wav"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    st.audio(temp_path)

    model, le_aksen, le_gender, le_provinsi, scaler_usia, ohe, metadata = load_assets()

    file_name = uploaded_file.name

    # ===============================
    # METADATA LOOKUP
    # ===============================
    row = metadata[metadata["file_name"] == file_name]

    if row.empty:
        st.error("❌ File tidak ditemukan di metadata.csv")
        st.stop()

    usia = row["usia"].values[0]
    gender = row["gender"].values[0]
    provinsi = row["provinsi"].values[0]

    # ===============================
    # FEATURE BUILDING
    # ===============================
    mfcc_feat = extract_mfcc(temp_path)

    usia_scaled = scaler_usia.transform([[usia]])
    cat_encoded = ohe.transform([[gender, provinsi]])

    meta = np.hstack([usia_scaled, cat_encoded]).astype(np.float32)
    meta = meta[:, None, None, :]
    meta = np.repeat(meta, mfcc_feat.shape[0], axis=1)
    meta = np.repeat(meta, mfcc_feat.shape[1], axis=2)

    X = np.concatenate([mfcc_feat[None, ...], meta], axis=-1)
    X = tf.convert_to_tensor(X, dtype=tf.float32)

    # ===============================
    # SUPPORT SET (STATIC SAMPLING)
    # ===============================
    support_idx = metadata.sample(15).index
    support_files = metadata.loc[support_idx, "file_name"]

    support_features = []
    support_labels = []

    for i, f in zip(support_idx, support_files):
        path = f
        if not os.path.exists(path):
            continue

        feat = extract_mfcc(path)
        usia_i = metadata.loc[i, "usia"]
        gender_i = metadata.loc[i, "gender"]
        prov_i = metadata.loc[i, "provinsi"]

        usia_s = scaler_usia.transform([[usia_i]])
        cat_s = ohe.transform([[gender_i, prov_i]])
        meta_s = np.hstack([usia_s, cat_s]).astype(np.float32)
        meta_s = meta_s[:, None, None, :]
        meta_s = np.repeat(meta_s, feat.shape[0], axis=1)
        meta_s = np.repeat(meta_s, feat.shape[1], axis=2)

        final = np.concatenate([feat[None, ...], meta_s], axis=-1)
        support_features.append(final[0])
        support_labels.append(le_aksen.transform([metadata.loc[i, "label_aksen"]])[0])

    support_tensor = tf.convert_to_tensor(np.array(support_features), dtype=tf.float32)
    support_labels = tf.convert_to_tensor(support_labels, dtype=tf.int32)

    # ===============================
    # PREDICTION
    # ===============================
    logits = model.call(
        support_tensor,
        X,
        support_labels,
        n_way=len(le_aksen.classes_)
    )

    pred_idx = tf.argmax(logits, axis=1).numpy()[0]
    pred_aksen = le_aksen.inverse_transform([pred_idx])[0]

    # ===============================
    # OUTPUT
    # ===============================
    st.success("✅ Prediksi Berhasil")

    st.markdown(f"""
    ### 🧾 Hasil Deteksi
    - **Aksen** : **{pred_aksen}**
    - **Usia** : {usia}
    - **Gender** : {gender}
    - **Provinsi** : {provinsi}
    """)
