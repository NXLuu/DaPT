import numpy as np
from sklearn import preprocessing
duoPath = "D:\Kì 4\Đa phương tiện\BTL\data\Song tấu"
soloPath = "D:\Kì 4\Đa phương tiện\BTL\data\Đơn tấu"
concertPath = "D:\Kì 4\Đa phương tiện\BTL\data\Hòa tấu"

input_labels = ['đơn tấu','song tấu','hòa tấu']
encoder = preprocessing.LabelEncoder()
encoder.fit(input_labels)
encoded_labels = encoder.transform(input_labels)
print(encoded_labels)
input_path = [
    'D:\Kì 4\Đa phương tiện\BTL\data\Song tấu\wav',
    "D:\Kì 4\Đa phương tiện\BTL\data\Đơn tấu\wav",
    'D:\Kì 4\Đa phương tiện\BTL\data\Hòa tấu\wav'
]

def loadAllFile():
    return  1
