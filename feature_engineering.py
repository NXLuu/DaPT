import librosa
import numpy as np
from config import CreateDataset

sr = CreateDataset.sr
fs = CreateDataset.fs
# fs =2
# hs = 1
hs = CreateDataset.hs


def extract_feature(samples):
    result = []
    features = []

    # Timbre features
    energy = rms(samples)
    l_zc = zc(samples)
    silient_ratio1 = silent_ratio(samples)
    spectral_centroid = librosa.feature.spectral_centroid(y=samples, sr=sr, n_fft=fs, hop_length=hs)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=samples, sr=sr, n_fft=fs, hop_length=hs)
    spectral_contrast = librosa.feature.spectral_contrast(y=samples, sr=sr, n_fft=fs, hop_length=hs)
    spectral_rollof = librosa.feature.spectral_rolloff(y=samples, sr=sr, n_fft=fs, hop_length=hs)
    zero_crossing = librosa.feature.zero_crossing_rate(y=samples, frame_length=fs, hop_length=hs)

    features.append(spectral_contrast)
    features.append(spectral_bandwidth)
    features.append(spectral_centroid)
    features.append(spectral_rollof)
    features.append(l_zc)
    features.append(energy)
    features.append(silient_ratio1)

    for feature in features:
        result.append(np.mean(feature))
        result.append(np.std(feature))

    return result

def rms(y):
    energy = np.array([
        sum(abs(y[i:i + fs] ** 2))/fs
        for i in range(0, len(y), hs)
    ])
    return  energy

def zc(y):
    sgn_array = [sgn(y[i:i + fs])  for i in range(0, len(y), hs)]
    zc = np.array([
        sum( [abs(sgn_array[i][j] - sgn_array[i][j-1]) for j in range(0, len(sgn_array[i])-1)]) / (2 * len(sgn_array[i]))
        for i in range(0, len(sgn_array), 1)
    ])
    return zc

def sgn(x):
    x[x > 0] = 1
    x[x == 0] = 0
    x[x < 0] = -1
    return x

def silent_ratio(y):
    threshold = np.mean(abs(y)) * 0.06
    y[abs(y) < threshold] = 0
    ratio = np.array([
        (np.count_nonzero(y[i:i + fs]==0) / fs)
        for i in range(0, len(y), hs)
    ])


    return ratio
# arr = np.array([1.0,-2.0,3.0,4.0,-1.0])
# zc1 = zc(arr)
# zero_crossing = librosa.feature.zero_crossing_rate(y=arr, frame_length=fs, hop_length=hs, center=False)
# spectral_centroid = librosa.feature.spectral_centroid(y=arr, sr=sr, n_fft=fs, hop_length=hs)
# print(silient_ratio(arr))