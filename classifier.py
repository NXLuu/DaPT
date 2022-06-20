import sys
import os
import numpy as np
import pandas as pd
import librosa
import sklearn
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
import joblib
from feature_engineering import *
from config import *
from config import Model, CreateDataset

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
plt.tight_layout()
plt.rcParams["figure.figsize"] = (10,7.5)

def save_model():
    # Load data
    data_set = pd.read_csv(CreateDataset.Name, index_col=False)
    data_set = np.array(data_set)

    # Cacluate Shape
    row, col = data_set.shape
    # Get X and Y
    x = data_set[:, :col-1]
    y = data_set[:, col-1]

    neigh = KNeighborsClassifier(n_neighbors=7)
    neigh.fit(x, y)

    joblib.dump(neigh, "model.sav")


    # # Cross-validation score
    # print("Cross-validation score: ")
    # print(cross_val_score(neigh, x, y, scoring="accuracy"))
    #
    # # Cross_validation predictions
    # predictions = cross_val_predict(neigh, x, y, cv=3)
    # # A better figure representation of the confusion matrix
    # print_confusion_matrix(confusion_matrix(y, predictions), classNames)
    #
    #
    # # return res

def classify(fileName):
    loaded_model = joblib.load("model.sav")

    y, sr = librosa.load(fileName)
    librosa.get_duration(y, sr)
    data = np.array([extract_feature(y)])
    scaler = joblib.load("scaler1.save")
    data = scaler.transform(data)
    res = loaded_model.predict(data)

    return res



#Names of the classes
classNames = ["đơn tấu", "song tấu", "hòa tấu"]

def print_confusion_matrix(confusion_matrix,class_names,figsize=(5,3),fontsize=10):
    df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names, )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    return fig

save_model()