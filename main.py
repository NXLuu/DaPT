import matplotlib.pyplot as plt
import  scipy
from scipy.io import wavfile as wav
from scipy.fftpack import fft
from os import walk
import  pydub
import os
from pydub import AudioSegment
from python_speech_features import mfcc
from python_speech_features import logfbank
import librosa
import librosa.display
import pandas as pd
import openpyxl
# import IPython.display as idp
# import IPython.display as ipd

duoPath = "D:\Kì 4\Đa phương tiện\Dataset\Song tấu"
soloPath = "D:\Kì 4\Đa phương tiện\Dataset\Đơn tấu"
concertPath = "D:\Kì 4\Đa phương tiện\Dataset\Hòa tấu"

mypath = duoPath;



link = os.getcwd()
print(link)

pydub.AudioSegment.converter = "C:\Program Files\\ffmpeg\\bin" + "\\ffmpeg.exe"
pydub.AudioSegment.ffprobe   = "C:\Program Files\\ffmpeg\\bin" + "\\ffprobe.exe"

def convertMp3ToWav(filename):
    src = mypath + "\\" + filename
    dst = mypath + "\wav\\" + filename.split(".mp3")[0] + ".wav";
    sound = AudioSegment.from_mp3(src)
    sound.export(dst, format="wav")
    print(src)

def ftt(path, filePath, filename):
    fs, data = wav.read(filePath) # load the data
    a = data.T[0] # this is a two channel soundtrack, I get the first track
    b=[(ele/2**8.)*2-1 for ele in a] # this is 8-bit track, b is now normalized on [-1,1)
    c = fft(b) # create a list of complex number
    d = int(len(c)/2)  # you only need half of the fft list
    plt.plot(np.abs(c[:(d-1)]),'r')
    plt.savefig(path+'\\fig\\' + filename + '.png' ,bbox_inches='tight')
    plt.clf()

def ftt2(path, filePath, filename):
    fs_rate, signal = wav.read(filePath)
    # print("Frequency sampling", fs_rate)
    l_audio = len(signal.shape)
    # print("Channels", l_audio)
    if l_audio == 2:
        signal = signal.sum(axis=1) / 2
    N = signal.shape[0]
    # print("Complete Samplings N", N)
    secs = N / float(fs_rate)
    # print("secs", secs)
    Ts = 1.0 / fs_rate  # sampling interval in time
    # print("Timestep between samples Ts", Ts)
    t = np.arange(0, secs, Ts)  # time vector as scipy arange field / numpy.ndarray
    FFT = abs(fft(signal))
    FFT_side = FFT[range(int(N / 2))]  # one side FFT range
    freqs = scipy.fftpack.fftfreq(signal.size, t[1] - t[0])
    fft_freqs = np.array(freqs)
    freqs_side = freqs[range(int(N / 2))]  # one side frequency range
    fft_freqs_side = np.array(freqs_side)
    plt.subplot(311)
    p1 = plt.plot(t, signal, "g")  # plotting the signal
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.subplot(312)
    p2 = plt.plot(freqs, FFT, "r")  # plotting the complete fft spectrum
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Count dbl-sided')
    plt.subplot(313)
    p3 = plt.plot(freqs_side, abs(FFT_side), "b")  # plotting the positive fft spectrum
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Count single-sided')
    plt.savefig(path + '\\fig\\' + filename + '.png', bbox_inches='tight')
    plt.clf()

import numpy as np
#
# for (filename) in f:
#     # src = mypath + "\\" + filename
#     dst = mypath + "\wav\\" + filename.split(".mp3")[0] + ".wav";
#     # sound = AudioSegment.from_mp3(src)
#     # sound.export(dst, format="wav")
#     # print(src)
#     # convertMp3ToWav(mypath, filename)
#     ftt2(mypath, dst, filename)



def getMfcc(path, filePath, filename):
    fs_rate, signal = wav.read(filePath)
    mfcc_feat1 = mfcc(signal, fs_rate)
    print(mfcc_feat1)
    fbank_feat = logfbank(sig, rate)
    print(fbank_feat)

def loadAllWav():
    f = []
    for (dirpath, dirnames, filenames) in walk(mypath):
        f.extend(filenames)
        break
    # print(f)
    # ZCR = [];
    # e = [];
    for (filename) in f:
        # dst = mypath + "\wav\\" + filename.split(".mp3")[0] + ".wav";
        # # getMfcc(mypath, dst, filename)
        #
        # ZCR = np.append(ZCR, getZCR(dst))
        # e = np.append(e, energy(dst))
        convertMp3ToWav(filename)
    # print(ZCR)
    # df = pd.DataFrame(ZCR,
    #                   index=f, columns=['ZCR'])
    # book = openpyxl.load_workbook('feature.xlsx')
    # writer = pd.ExcelWriter('feature.xlsx', engine='openpyxl')
    # writer.book = book
    # writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
    # df.to_excel(writer, sheet_name='Sheet1', startrow=50, startcol=10)
    # writer.save()

def getZCR(filePath):
    FRAME_SIZE = 1024
    HOP_LENGTH = 512
    x, sr = librosa.load(filePath)
    # fs_rate, audioData = wav.read(filePath)
    # print(audioData.shape)
    # zero_crosses = np.nonzero(np.diff( audioData > 0))[0]
    # print(zero_crosses.size);
    # print(zero_crosses.size/(audioData.size))

    music = librosa.load(filePath)
    # zcr = librosa.zero_crossings(x, pad=False)
    # print(np.mean(zcr, axis=-2, keepdims=True))
    zcrs = librosa.feature.zero_crossing_rate(x)
    # print(zcrs.mean())
    # print(np.mean(zcr, axis=0))
    return zcrs.mean();

def energy(filePath):
    y, sr = librosa.load(filePath)
    e = librosa.feature.rms(y=y)
    return e.mean()


def pitch(filePath):
    x, sr = librosa.load(filePath)
    hop_length = 100
    onset_env = librosa.onset.onset_strength(x)
    plt.plot(onset_env)
    plt.xlim(0, len(onset_env))

# pitch('D:\Kì 4\Đa phương tiện\BTL\data\Hòa tấu\wav\piano +violin+ clarinet.wav');
# loadAllWav()

loadAllWav()