import numpy as np
import librosa

def mel_from_wav(y, sr=22050, n_mels=128, hop_length=512, n_fft=2048, fmin=20, fmax=None):
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, fmin=fmin, fmax=fmax
    )
    S = librosa.power_to_db(S, ref=np.max)
    Smin, Smax = S.min(), S.max()
    Sm = (S - Smin) / (Smax - Smin + 1e-8)
    return Sm.astype("float32")
