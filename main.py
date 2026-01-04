import os
import numpy as np
import librosa
import tensorflow as tf
import sounddevice as sd
import queue

# --- CONFIGURATION ---
# Make sure this matches the name of the file you downloaded!
MODEL_PATH = "updated_audio_model_new.h5" 

# ⚠️ AUTOMATICALLY SORTED ALPHABETICALLY (Required for correct predictions)
# I re-ordered your list based on how Python/AI sorts them:
# ⚠️ REPLACE THE 'CLASSES' LIST IN YOUR main.py WITH THIS:
CLASSES = [
    'Background_Noise', 
    'Sendhelp', 
    'call police', 
    'help me', 
    'i need help', 
    'madat karo', 
    'mujhe_bachao', 
    'palice call martini'
]
# Audio Settings
DURATION = 2.0        # Seconds to listen per chunk
SAMPLE_RATE = 22050   # Must match training
CONFIDENCE_THRESHOLD = 85.0  # % Confidence required to trigger alert
VOLUME_THRESHOLD = 0.01 # Minimum volume to even bother listening

def load_model():
    if not os.path.exists(MODEL_PATH):
        # Try finding *any* .h5 file if the specific name is wrong
        files = [f for f in os.listdir('.') if f.endswith('.h5')]
        if files:
            print(f"⚠️ '{MODEL_PATH}' not found. Using '{files[0]}' instead...")
            return tf.keras.models.load_model(files[0])
        else:
            raise FileNotFoundError(f"❌ No .h5 model file found in this folder!")
    
    print("Loading AI model... (this may take a moment)")
    return tf.keras.models.load_model(MODEL_PATH)

def preprocess_audio(audio_data):
    """Convert raw audio to MFCCs exactly like training."""
    mfccs = librosa.feature.mfcc(y=audio_data, sr=SAMPLE_RATE, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    return mfccs_scaled.reshape(1, -1)

def listen_and_detect():
    model = load_model()
    print("\n" + "="*50)
    print(f"🎧 SYSTEM ACTIVE: Listening for {len(CLASSES)-1} commands...")
    print(f"   (Ignoring 'Background_Noise')")
    print(f"   Threshold: {CONFIDENCE_THRESHOLD}%")
    print("="*50)

    try:
        while True:
            # 1. Record Audio Chunk
            audio_chunk = sd.rec(int(DURATION * SAMPLE_RATE), 
                                 samplerate=SAMPLE_RATE, 
                                 channels=1, 
                                 dtype='float32')
            sd.wait() # Wait for recording to finish
            
            # Flatten to 1D array
            audio_data = audio_chunk.flatten()

            # 2. Volume Gate (Save CPU if it's dead silent)
            volume = np.sqrt(np.mean(audio_data**2))
            if volume < VOLUME_THRESHOLD:
                print(".", end="", flush=True) # Silence dots
                continue

            # 3. Predict
            features = preprocess_audio(audio_data)
            predictions = model.predict(features, verbose=0)
            
            # Get result
            predicted_index = np.argmax(predictions)
            confidence = predictions[0][predicted_index] * 100
            result = CLASSES[predicted_index]

            # 4. Filter Results
            if result == "Background_Noise":
                # AI thinks it's just noise -> Do nothing (or print weak indicator)
                print("_", end="", flush=True) 
            
            elif confidence > CONFIDENCE_THRESHOLD:
                # REAL DETECTION!
                print(f"\n\n🚨 DETECTED: {result.upper()} ({confidence:.1f}%)")
                print("   Action: Sending Alert... (Placeholder)\n")
            
            else:
                # It heard something, but wasn't sure
                print(f"\n   [Heard: {result} ({confidence:.1f}%)]")

    except KeyboardInterrupt:
        print("\n\n🛑 Stopping...")

if __name__ == "__main__":
    listen_and_detect()