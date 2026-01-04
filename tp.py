import sounddevice as sd
import numpy as np
import time

# Settings from your main.py
SAMPLE_RATE = 22050
DURATION = 1.0 # 1 second chunks

def calibrate_rms():
    print("\n📊 RMS CALIBRATOR (Matches your AI's eyes)")
    print("------------------------------------------")
    print("Speak normally into your mic now...")
    print("------------------------------------------")

    try:
        while True:
            # Record exactly like main.py
            audio_chunk = sd.rec(int(DURATION * SAMPLE_RATE), 
                                 samplerate=SAMPLE_RATE, 
                                 channels=1, 
                                 dtype='float32')
            sd.wait()
            
            # Calculate RMS (Root Mean Square) - The real volume number
            audio_data = audio_chunk.flatten()
            rms_volume = np.sqrt(np.mean(audio_data**2))
            
            # Visual Bar
            bar_len = int(rms_volume * 500) 
            bar = "█" * bar_len
            
            print(f"Volume: {rms_volume:.5f}  {bar}")
            
    except KeyboardInterrupt:
        print("\nStopped.")

if __name__ == "__main__":
    calibrate_rms()