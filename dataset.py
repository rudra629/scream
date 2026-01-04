import os
import time
import shutil
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
import librosa.effects

# --- CONFIGURATION ---
RAW_FOLDER = "raw_bachao"       # Where we save your voice
FINAL_FOLDER = "bachao"         # Where the AI data goes
ZIP_NAME = "bachao"             # Name of the final zip file
SAMPLE_RATE = 22050
DURATION = 2.0                  # Seconds per recording
TARGET_COUNT = 500              # Total files to generate

def record_user_voice():
    if os.path.exists(RAW_FOLDER): shutil.rmtree(RAW_FOLDER)
    os.makedirs(RAW_FOLDER)

    print("\n" + "="*50)
    print("🎙️  STEP 1: RECORDING MASTER SAMPLES")
    print("   We need 10 recordings of you saying 'BACHAO'")
    print("   Tip: Vary your tone! (Scream, Whisper, Panic, Fast)")
    print("="*50)

    for i in range(1, 11):
        input(f"\nPress ENTER to record sample {i}/10...")
        print("🔴 Recording... SAY IT NOW!")
        
        # Record
        audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait() # Wait for recording to finish
        print("✅ Captured.")

        # Save
        filename = os.path.join(RAW_FOLDER, f"original_{i}.wav")
        sf.write(filename, audio, SAMPLE_RATE)
        time.sleep(0.5)

    print(f"\n✨ Recording complete! Saved 10 files in '{RAW_FOLDER}'")

def augment_audio(y, sr):
    """Apply random effects to create new variations"""
    # 1. Random Pitch Shift (Make voice deeper or higher)
    # Range: -3 (Deep) to +3 (Chipmunk)
    steps = np.random.uniform(-3, 3)
    y_aug = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)

    # 2. Random Speed Change (Slower or Faster)
    # Range: 0.8x (Slow) to 1.2x (Fast)
    if np.random.random() > 0.3: # 70% chance to change speed
        rate = np.random.uniform(0.8, 1.2)
        y_aug = librosa.effects.time_stretch(y_aug, rate=rate)

    # 3. Add Noise (Simulate bad mic or distance)
    if np.random.random() > 0.4: # 60% chance to add noise
        noise_amp = 0.005 * np.random.uniform() * np.amax(y_aug)
        y_aug = y_aug + noise_amp * np.random.normal(size=y_aug.shape[0])

    # 4. Fix Length (Ensure it stays exactly 2 seconds)
    target_len = int(SAMPLE_RATE * DURATION)
    if len(y_aug) > target_len:
        y_aug = y_aug[:target_len]
    else:
        y_aug = np.pad(y_aug, (0, target_len - len(y_aug)))

    return y_aug

def generate_dataset():
    print("\n" + "="*50)
    print(f"⚙️  STEP 2: GENERATING {TARGET_COUNT} SAMPLES")
    print("   (This multiplies your voice using AI magic)")
    print("="*50)

    if os.path.exists(FINAL_FOLDER): shutil.rmtree(FINAL_FOLDER)
    os.makedirs(FINAL_FOLDER)

    # Load master files
    master_files = [os.path.join(RAW_FOLDER, f) for f in os.listdir(RAW_FOLDER) if f.endswith('.wav')]
    
    count = 0
    while count < TARGET_COUNT:
        for file_path in master_files:
            if count >= TARGET_COUNT: break

            try:
                # Load Original
                y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                
                # Augment
                y_new = augment_audio(y, sr)

                # Save
                out_name = os.path.join(FINAL_FOLDER, f"bachao_aug_{count}.wav")
                sf.write(out_name, y_new, SAMPLE_RATE)
                
                count += 1
                if count % 50 == 0: print(f"   Generated {count}/{TARGET_COUNT} files...", end="\r")
            except Exception as e:
                print(f"Error: {e}")

    print(f"\n✅ Dataset Generation Complete: {count} files ready.")

def zip_dataset():
    print("\n📦 STEP 3: ZIPPING FILES")
    shutil.make_archive(ZIP_NAME, 'zip', FINAL_FOLDER)
    print(f"🎉 SUCCESS! created '{ZIP_NAME}.zip'")
    print("👉 ACTION: Upload this zip file to your Google Drive 'dataset' folder.")

if __name__ == "__main__":
    try:
        record_user_voice()
        generate_dataset()
        zip_dataset()
    except KeyboardInterrupt:
        print("\n❌ Stopped by user.")