import os
import numpy as np
import librosa
import tensorflow as tf
import sounddevice as sd

# --- CONFIGURATION ---
MODEL_PATH = "audio_classification_model_200.h5" 
SAMPLE_RATE = 22050
DURATION = 1.0 # Shorter chunks for faster calibration

def listen_volume():
    print("\n" + "="*50)
    print("🎤 CALIBRATION MODE: Stay Silent!")
    print("   Watch the 'Volume' number below.")
    print("="*50)

    try:
        while True:
            # Record 1 second
            audio_chunk = sd.rec(int(DURATION * SAMPLE_RATE), 
                                 samplerate=SAMPLE_RATE, 
                                 channels=1, 
                                 dtype='float32')
            sd.wait()
            
            # Calculate Volume (RMS)
            audio_data = audio_chunk.flatten()
            volume = np.sqrt(np.mean(audio_data**2))
            
            # Print the volume level
            print(f"🔊 Current Volume: {volume:.5f}")

    except KeyboardInterrupt:
        print("\nStopped.")

if __name__ == "__main__":
    listen_volume()