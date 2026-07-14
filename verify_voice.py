import requests
import base64
import numpy as np
import io
import soundfile as sf

def test_voice_predict():
    url = "http://127.0.0.1:5000/voice-predict"
    
    # Generate a dummy 2-second sine wave audio
    sr = 22050
    t = np.linspace(0, 2, 2 * sr)
    # A shaky-ish low frequency wave
    y = 0.5 * np.sin(2 * np.pi * 100 * t) + 0.1 * np.random.randn(len(t))
    
    buffer = io.BytesIO()
    sf.write(buffer, y, sr, format='WAV')
    audio_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    try:
        resp = requests.post(url, json={"audio": audio_b64})
        print(f"Status: {resp.status_code}")
        print(f"Response: {resp.json()}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_voice_predict()
