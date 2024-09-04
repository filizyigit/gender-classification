import os
import torch
import torch.nn as nn
import numpy as np
import pyaudio
import wave
import librosa
import joblib
import time

# Model sınıfını tanımlama
class SVM(nn.Module):
    def __init__(self, input_dim):
        super(SVM, self).__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.fc(x)

# Özellik çıkarma fonksiyonu
def extract_features(y, sr):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean

# Cinsiyet tahmin fonksiyonu
def predict_gender(y, sr, model, scaler, silence_threshold=0.001):
    if np.mean(np.abs(y)) < silence_threshold:
        return None  # Ses algılanmadı
    
    features = extract_features(y, sr)
    features = features.reshape(1, -1)
    
    # Pad işlemi yaparak özellik sayısını eğitimdeki gibi 20'ye tamamla
    if features.shape[1] < 20:
        pad_width = 20 - features.shape[1]
        features = np.pad(features, ((0, 0), (0, pad_width)), mode='constant')
    
    features_scaled = scaler.transform(features)
    features_tensor = torch.tensor(features_scaled, dtype=torch.float32)
    with torch.no_grad():
        output = model(features_tensor)
        prediction = torch.round(torch.sigmoid(output))
        gender = 'female' if prediction.item() == 1 else 'male'
        return gender

def main():
    model_path = 'voice_model_weights.pth'
    scaler_path = 'scaler.pkl'
    save_path = "C:/Users/filizyigit/Desktop/Bitirme Ödevi/bitirme/classification_voice"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    input_dim = 20  # Eğitim sırasında kullanılan özellik sayısı
    model = SVM(input_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    scaler = joblib.load(scaler_path)

    rate = 44100
    chunk = 4096
    channels = 1
    duration = 5  # 5 saniye boyunca ses kaydı

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=channels, rate=rate, input=True, frames_per_buffer=chunk)

    print("Ses kaydediliyor...")

    frames = []
    for _ in range(0, int(rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
    y = audio_data.astype(np.float32) / np.iinfo(np.int16).max
    sr = rate

    predicted_gender = predict_gender(y, sr, model, scaler)
    
    if predicted_gender is None:
        print("Ses algılanamadı.")
    else:
        print(f'Tahmin edilen cinsiyet: {predicted_gender}')
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"{predicted_gender}_{timestamp}.wav"
        file_path = os.path.join(save_path, filename)
        
        wf = wave.open(file_path, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        print(f'Ses kaydedildi: {file_path}')

    stream.stop_stream()
    stream.close()
    p.terminate()
    print("Mikrofon kapatıldı.")

if __name__ == "__main__":
    main()

