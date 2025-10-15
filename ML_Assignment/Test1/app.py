import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
import joblib
from tensorflow.keras.models import load_model
import pandas as pd

# --- CONFIGURATION ---
MODEL_PATH = 'cnn_1d_sequence_model.h5'
SCALER_PATH = 'scaler.pkl'            # saved scaler
ENCODER_PATH = 'encoder_classes.npy'  # Path to saved encoder classes

# Feature extraction parameters (must match your training script)
N_MFCC = 13
N_CHROMA = 12
MAX_PAD_LEN = 174

# --- LOAD MODELS AND PREPROCESSORS ---
# Use st.cache_resource to load these only once and speed up the app
@st.cache_resource
def load_essentials():
    """Loads the trained model, scaler, and label encoder."""
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoder_classes = np.load(ENCODER_PATH, allow_pickle=True)
    return model, scaler, encoder_classes

model, scaler, encoder_classes = load_essentials()

# --- FEATURE EXTRACTION FUNCTION ---
# This function must be IDENTICAL to the one used for training
def extract_features(file_path, max_pad_len=MAX_PAD_LEN):
    """Extracts the same set of features as used in training."""
    try:
        audio, sr = librosa.load(file_path, res_type='kaiser_fast')
        
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_chroma=N_CHROMA)
        zcr = librosa.feature.zero_crossing_rate(y=audio)
        rms = librosa.feature.rms(y=audio)
        spec_cent = librosa.feature.spectral_centroid(y=audio, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
        spec_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        
        combined = np.vstack([mfccs, chroma, zcr, rms, spec_cent, spec_bw, spec_rolloff])
        
        if combined.shape[1] > max_pad_len: combined = combined[:, :max_pad_len]
        else: combined = np.pad(combined, ((0, 0), (0, max_pad_len - combined.shape[1])), mode='constant')
        
        return combined.T
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None

# --- STREAMLIT UI ---
st.set_page_config(layout="wide")

st.title("🎤 Speech Emotion Recognition")
st.write(
    "Welcome! This application uses a deep learning model to predict the emotion from a speech audio file. "
    "Upload a **.wav** file and click 'Predict Emotion' to see the result."
)
st.markdown("---")

# File uploader
uploaded_file = st.file_uploader("Choose a WAV file", type="wav")

if uploaded_file is not None:
    # Display the audio player
    st.audio(uploaded_file, format='audio/wav')
    
    # "Predict" button
    if st.button("Predict Emotion", type="primary"):
        with st.spinner("Analyzing the audio..."):
            # 1. Extract features from the uploaded file
            features = extract_features(uploaded_file)
            
            if features is not None:
                # 2. Reshape and scale the features
                # Reshape for the scaler (which expects 2D)
                features_reshaped_for_scaler = features.reshape(1, -1)
                
                # We need to reshape the features to match the shape the scaler was fit on.
                # The scaler was fit on (num_samples * time_steps, num_features).
                # Since we have 1 sample, we reshape to (time_steps, num_features).
                scaled_features_reshaped = scaler.transform(features.reshape(-1, features.shape[-1]))

                # Reshape back to the 3D shape the model expects: (1, time_steps, num_features)
                scaled_features = scaled_features_reshaped.reshape(1, features.shape[0], features.shape[1])

                # 3. Make a prediction
                prediction_probs = model.predict(scaled_features)
                
                # 4. Decode the prediction
                predicted_index = np.argmax(prediction_probs)
                predicted_emotion = encoder_classes[predicted_index]
                confidence = np.max(prediction_probs) * 100
                
                # Display the result
                st.success(f"**Predicted Emotion: {predicted_emotion.capitalize()}**")
                st.info(f"Confidence: {confidence:.2f}%")

                # Display probabilities for all classes
                st.write("### Prediction Probabilities")
                probs_df = pd.DataFrame({'Emotion': encoder_classes, 'Probability': prediction_probs.flatten()})
                st.bar_chart(probs_df.set_index('Emotion'))

st.markdown("---")