import sys
import os
import numpy as np
import librosa
import pandas as pd
import sklearn
from os import walk
import joblib


from feature_engineering import *
from config import CreateDataset, Normal

data_path = CreateDataset.data_paths
csv_name = CreateDataset.Name
labels = CreateDataset.labels


# get file path
def getFilesInFolder(path):
    files = [];
    for (dirpath, dirnames, filenames) in walk(path):
        files.extend(filenames)
        break
    return files


# file load
def get_sampels():
    audios = []
    data_labels = []

    i = 0;
    for path in data_path:
        files = getFilesInFolder(path)
        for file in files:
            path_audio = path + "\\" + file
            y, sr = librosa.load(path_audio)
            audios.append(y)
            data_labels.append(labels[i])
        i+=1
    audios_numpy = np.array(audios)
    return audios_numpy, data_labels


def main():
    is_created = False
    audios_numpy, data_labels = get_sampels()
    for samples in audios_numpy:
        row = extract_feature(samples);
        if not is_created:
            dataset_numpy = np.array(row)
            is_created = True
        elif is_created:
            dataset_numpy = np.vstack((dataset_numpy, row))

    scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1,1))
    scaler.fit(dataset_numpy)
    joblib.dump(scaler, 'scaler1.save')
    dataset_numpy = scaler.transform(dataset_numpy)
    print(dataset_numpy.shape)
    dataset_pandas = pd.DataFrame(dataset_numpy)
    dataset_pandas["labels"] = data_labels
    dataset_pandas.to_csv(csv_name, index=False)

if __name__ == '__main__':
    main()