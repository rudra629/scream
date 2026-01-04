import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import time
import os

# Settings
SAMPLE_RATE = 22050
DURATION = 2.0  # 2 seconds per clip
NUM_CLIPS = 20  # Record 20 clips
OUTPUT_FOLDER = "new_noise_data"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print(f"🛑 FINE-TUNING PREP")
print(f"We will record {NUM_CLIPS} clips of your room's SILENCE.")
print("Don't speak. Let it capture the fan/static/hum.")
print("-" * 40)

for i in range(NUM_CLIPS):
    print(f"Recording clip {i+1}/{NUM_CLIPS}...", end="\r")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    
    # Save file
    filename = os.path.join(OUTPUT_FOLDER, f"real_static_{i}.wav")
    wav.write(filename, SAMPLE_RATE, audio)
    time.sleep(0.5)

print(f"\n✅ Done! You have {NUM_CLIPS} files in '{OUTPUT_FOLDER}'.")
print("👉 ACTION: Upload these files to your Colab now.")