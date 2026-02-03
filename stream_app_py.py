# streamlit_app.py
import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tempfile
import os
import sys

# Set page config FIRST
st.set_page_config(
    page_title="Few-Shot Accent Detection",
    page_icon="🎙️",
    layout="wide"
)

# Title
st.title("🎙️ Few-Shot Accent Detection")
st.markdown("Deteksi aksen, gender, dan provinsi dari file audio")

# Check if we're in deployment environment
@st.cache_resource
def check_environment():
    """Check if we're in Streamlit Cloud"""
    try:
        import socket
        hostname = socket.gethostname()
        return "Streamlit" in hostname or "share.streamlit.io" in hostname
    except:
        return False

# Load encoders and model (simplified for deployment)
@st.cache_resource
def load_encoders():
    """Load or create encoders"""
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.preprocessing import OneHotEncoder
    
    # Create dummy encoders for demo
    le_y = LabelEncoder()
    le_gender = LabelEncoder()
    le_provinsi = LabelEncoder()
    scaler_usia = StandardScaler()
    
    # Fit with dummy data
    le_y.fit(["Sunda", "Jawa_Tengah", "Jawa_Timur", "YogyaKarta", "Betawi])
    le_gender.fit(["Male", "Female"])
    le_provinsi.fit(["Jawa Barat", "Jawa Tengah", "Jawa Timur", "Yogyakarta", "DKI Jakarta"])
    
    return le_y, le_gender, le_provinsi, scaler_usia

# Feature extraction
def extract_mfcc_features(audio_path, sr=22050, n_mfcc=40, max_len=174):
    """Extract MFCC features from audio file"""
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=sr)
        
        # Normalize
        y = librosa.util.normalize(y)
        
        # Extract MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        
        # Pad or truncate
        if mfcc.shape[1] < max_len:
            pad_width = max_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_len]
            
        # Add channel dimension
        mfcc = mfcc[np.newaxis, ..., np.newaxis]
        
        return mfcc
        
    except Exception as e:
        st.error(f"Error extracting features: {str(e)}")
        return None

# Main app
def main():
    # Initialize session state
    if 'prediction_made' not in st.session_state:
        st.session_state.prediction_made = False
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Demo mode selection
        demo_mode = st.radio(
            "Select Mode:",
            ["Upload Audio", "Use Sample Audio"]
        )
        
        # Metadata inputs
        st.subheader("📋 Metadata")
        usia = st.number_input("Usia", 0, 100, 25)
        gender = st.selectbox("Gender", ["Male", "Female"])
        provinsi = st.selectbox("Provinsi", [
            "Jawa Barat", "Jawa Tengah", "Jawa Timur", 
            "Yogyakarta", "Jakarta"
        ])
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🎵 Audio Input")
        
        audio_file = None
        
        if demo_mode == "Upload Audio":
            audio_file = st.file_uploader(
                "Upload file audio (.wav, .mp3)",
                type=["wav", "mp3"],
                help="Upload audio untuk deteksi aksen"
            )
        else:
            # Create a sample audio option
            st.info("Using sample audio for demonstration")
            # You can add actual sample audio files here
    
    with col2:
        st.header("Metadata Summary")
        st.metric("Usia", usia)
        st.metric("Gender", gender)
        st.metric("Provinsi", provinsi)
    
    # Process audio if available
    if audio_file is not None or demo_mode == "Use Sample Audio":
        st.divider()
        
        # Display audio player
        if audio_file is not None:
            st.audio(audio_file, format="audio/wav")
        
        # Feature extraction button
        if st.button("🚀 Extract Features and Predict", type="primary"):
            with st.spinner("Processing audio..."):
                try:
                    # Save uploaded file temporarily
                    if audio_file is not None:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                            tmp_file.write(audio_file.getbuffer())
                            tmp_path = tmp_file.name
                    else:
                        # Use a dummy path for sample mode
                        tmp_path = "sample.wav"
                    
                    # Extract features
                    features = extract_mfcc_features(tmp_path)
                    
                    if features is not None:
                        # Load encoders
                        le_y, le_gender, le_provinsi, _ = load_encoders()
                        
                        # Generate predictions (demo - replace with actual model)
                        accent_classes = le_y.classes_
                        accent_probs = np.random.dirichlet(np.ones(len(accent_classes)))
                        predicted_accent = accent_classes[np.argmax(accent_probs)]
                        
                        gender_pred = gender  # Using input for demo
                        gender_conf = 0.85
                        
                        province_pred = provinsi  # Using input for demo
                        province_conf = 0.78
                        
                        # Store in session state
                        st.session_state.prediction_made = True
                        st.session_state.results = {
                            'features': features,
                            'accent': predicted_accent,
                            'accent_probs': accent_probs,
                            'gender': gender_pred,
                            'gender_conf': gender_conf,
                            'province': province_pred,
                            'province_conf': province_conf,
                            'metadata': {
                                'usia': usia,
                                'gender_input': gender,
                                'provinsi_input': provinsi
                            }
                        }
                        
                        # Clean up temp file
                        if audio_file is not None and os.path.exists(tmp_path):
                            os.unlink(tmp_path)
                            
                        st.success("✅ Prediction completed!")
                
                except Exception as e:
                    st.error(f"Error during processing: {str(e)}")
    
    # Display results if prediction was made
    if st.session_state.prediction_made and 'results' in st.session_state:
        results = st.session_state.results
        
        st.divider()
        st.header("📈 Prediction Results")
        
        # Create three columns for results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("🎭 Accent Prediction")
            st.metric(
                "Predicted Accent",
                results['accent'],
                f"{max(results['accent_probs'])*100:.1f}%"
            )
            
            # Display probabilities
            st.write("**Probabilities:**")
            for accent, prob in zip(le_y.classes_, results['accent_probs']):
                st.progress(float(prob), text=f"{accent}: {prob*100:.1f}%")
        
        with col2:
            st.subheader("👤 Gender Prediction")
            st.metric(
                "Predicted Gender",
                results['gender'],
                f"{results['gender_conf']*100:.1f}%"
            )
            
            # Gender confidence
            st.write("**Confidence:**")
            st.progress(float(results['gender_conf']))
        
        with col3:
            st.subheader("📍 Province Prediction")
            st.metric(
                "Predicted Province",
                results['province'],
                f"{results['province_conf']*100:.1f}%"
            )
            
            # Province confidence
            st.write("**Confidence:**")
            st.progress(float(results['province_conf']))
        
        # Feature information
        st.divider()
        st.subheader("📊 Feature Information")
        
        feat_col1, feat_col2 = st.columns(2)
        with feat_col1:
            st.write(f"**Feature Shape:** {results['features'].shape}")
            st.write(f"**MFCC Coefficients:** {results['features'].shape[1]}")
            st.write(f"**Time Frames:** {results['features'].shape[2]}")
        
        with feat_col2:
            st.write("**Input Metadata:**")
            st.write(f"- Usia: {results['metadata']['usia']}")
            st.write(f"- Gender Input: {results['metadata']['gender_input']}")
            st.write(f"- Provinsi Input: {results['metadata']['provinsi_input']}")
    
    # Add deployment instructions
    if check_environment():
        st.sidebar.info("🌐 Running on Streamlit Cloud")
    else:
        st.sidebar.info("💻 Running locally")

# Run the app
if __name__ == "__main__":
    # Add try-except for better error handling
    try:
        main()
    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        st.info("Please check the terminal for detailed error messages.")
