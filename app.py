import streamlit as st
import sounddevice as sd
import numpy as np
import librosa
import tensorflow as tf
import requests
import time
import os

# --- ⚙️ CONFIGURATION ---
MODEL_PATH = "audio_classification_model_200.h5"
WEBHOOK_URL = "https://your-api-endpoint.com/alert"  # <--- REPLACE THIS WITH YOUR REAL URL
CONFIDENCE_THRESHOLD = 80.0  # Alert only if > 80% sure
VOLUME_THRESHOLD = 0.01      # Sensitivity (Calibrated from your previous tests)

# ⚠️ YOUR EXACT CLASS LIST (Must match training)
CLASSES = [
    'Background_Noise', 
    'Sendhelp', 
    'bachao', 
    'call police', 
    'help me', 
    'i need help', 
    'madat karo', 
    'mujhe_bachao', 
    'palice call martini'
]

# --- 1. SETUP ---
st.set_page_config(page_title="AI Audio Guardian", layout="centered")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file '{MODEL_PATH}' not found!")
        return None
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# --- 2. AUDIO & API FUNCTIONS ---
def process_audio(audio_data):
    # Convert raw audio to MFCCs (The language of the AI)
    mfccs = librosa.feature.mfcc(y=audio_data, sr=22050, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    return mfccs_scaled.reshape(1, -1)

def send_post_request(command, confidence):
    """Sends the alert to your server"""
    payload = {
        "alert": "CRITICAL AUDIO DETECTED",
        "command": command,
        "confidence": f"{confidence:.2f}%",
        "timestamp": time.ctime()
    }
    
    try:
        # 🚀 SENDING REQUEST
        # Uncomment the next line to actually send it!
        # requests.post(WEBHOOK_URL, json=payload, timeout=2) 
        
        print(f"🚀 POST SENT: {payload}") # Print to terminal for debugging
        return True
    except Exception as e:
        print(f"❌ API Error: {e}")
        return False

# --- 3. UI LAYOUT ---
st.title("🚨 AI Audio Guardian")
st.markdown("---")

# Session State for Start/Stop
if 'run' not in st.session_state:
    st.session_state.run = False

# Buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("▶️ START LISTENING", type="primary"):
        st.session_state.run = True
with col2:
    if st.button("⏹️ STOP"):
        st.session_state.run = False

# Status Indicators
status_text = st.empty()
visual_alert = st.empty()

# --- 4. MAIN LISTENING LOOP ---
if st.session_state.run:
    if model is None:
        st.error("Model not loaded.")
    else:
        status_text.info(f"🎧 System Active... (Threshold: >{CONFIDENCE_THRESHOLD}%)")
        
        # CONTINUOUS LOOP
        while st.session_state.run:
            try:
                # 1. Record 2 Seconds
                duration = 2.0
                sr = 22050
                # Using device=None uses Windows Default. 
                # If it doesn't hear you, add device=X here (e.g. device=1)
                audio_chunk = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
                sd.wait() # Wait until recording is finished
                
                # 2. Check Volume (Gate)
                audio_flat = audio_chunk.flatten()
                vol = np.sqrt(np.mean(audio_flat**2))
                
                if vol < VOLUME_THRESHOLD:
                    visual_alert.markdown(f"💤 *Silence... (Vol: {vol:.4f})*")
                    time.sleep(0.1)
                    continue # Loop back immediately

                # 3. AI Prediction
                features = process_audio(audio_flat)
                predictions = model.predict(features, verbose=0)
                
                p_index = np.argmax(predictions)
                confidence = predictions[0][p_index] * 100
                result = CLASSES[p_index]

                # 4. DECISION LOGIC
                if result == "Background_Noise":
                    visual_alert.info(f"👂 Hearing Noise... ({confidence:.1f}%)")
                
                elif confidence > CONFIDENCE_THRESHOLD:
                    # 🔥 CRITICAL ALERT 🔥
                    visual_alert.error(f"🚨 **DETECTED: {result.upper()}** ({confidence:.1f}%)")
                    
                    # Send Post Request
                    send_post_request(result, confidence)
                    
                    # Optional: Play a beep locally or sleep briefly to avoid spamming 100 requests
                    time.sleep(1) 
                
                else:
                    # Heard something, but confidence too low (< 80)
                    visual_alert.warning(f"🤔 Unsure: {result} ({confidence:.1f}%)")

            except Exception as e:
                st.error(f"Loop Error: {e}")
                st.session_state.run = False
                break
else:
    status_text.markdown("**Status:** 🛑 System Stopped")