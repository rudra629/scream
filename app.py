import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
from audio_recorder_streamlit import audio_recorder
import os

# --- CONFIGURATION ---
MODEL_PATH = "updated_audio_model.h5" 
CLASSES = [
    'Background_Noise', 'Sendhelp', 'bachao', 'call police', 
    'help me', 'i need help', 'madat karo', 'mujhe_bachao', 'palice call martini'
]

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model '{MODEL_PATH}' not found!")
        return None
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# --- HELPER: PROCESS AUDIO ---
def predict_audio(audio_path):
    # Load and Preprocess
    audio, sr = librosa.load(audio_path, sr=22050)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0).reshape(1, -1)
    
    # Predict
    prediction = model.predict(mfccs_scaled)
    predicted_index = np.argmax(prediction)
    confidence = prediction[0][predicted_index] * 100
    label = CLASSES[predicted_index]
    
    return label, confidence

# --- UI LAYOUT ---
st.title("🚨 Scream Detection AI")
st.write("Click the microphone button below to speak.")

# 🎤 THE WEB MICROPHONE WIDGET
audio_bytes = audio_recorder(
    text="Click to Record",
    recording_color="#e8b62c",
    neutral_color="#6aa36f",
    icon_size="2x",
)

if audio_bytes:
    # 1. Save the audio from the browser to a temp file
    temp_filename = "temp_audio.wav"
    with open(temp_filename, "wb") as f:
        f.write(audio_bytes)
    
    st.audio(audio_bytes, format="audio/wav")
    
    # 2. Run AI Prediction
    with st.spinner("Analyzing sound..."):
        label, confidence = predict_audio(temp_filename)
    
    # 3. Show Results
    if label == "Background_Noise":
        st.info(f"🔊 Status: Safe (Just Noise)")
    elif confidence > 60:
        st.error(f"🚨 DETECTED: {label.upper()} ({confidence:.1f}%)")
        st.markdown("### 📞 Action: Sending Alert to Authorities...")
    else:
        st.warning(f"🤔 Heard: {label} ({confidence:.1f}%) - Not sure.")
