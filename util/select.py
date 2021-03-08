import pandas as pd
import os
import cv2
from sklearn.utils import shuffle
import numpy as np

classified_data = ["dr7objid",
                   'Error',
                   'CNN2classes1stClass', 'CNN2classes1stClassPerc',
                   'CNN2classes2ndClass', 'CNN2Classes2ndClassPerc', 'CNN3classes1stClass',
                   'CNN3Classes1stClassPerc', 'CNN3Classes2ndClass',
                   'CNN3Classes2ndClassPerc', 'CNN3Classes3rdClass',
                   'CNN3Classes3rdClassPerc']


def read(url):
    # remove missing data and redundant classified labels
    galaxies_csv = url
    data_frame = pd.read_csv(galaxies_csv, low_memory=False)
    data_frame = data_frame[data_frame["Error"] != 2]
    train_csv = data_frame.rename(columns={'ML2classes': "Class"}).drop(columns=classified_data)
    train_csv["Class"] = train_csv["Class"].map({"1": int(1), "0": int(0)})
    return train_csv[~train_csv["Class"].isin(['U', '--'])]


def concat(raw):
    return [i for i in raw]


def balanced_select(raw_data, size):
    one_feature = raw_data[raw_data["Class"] == 1][:size]
    zero_feature = raw_data[raw_data['Class'] == 0][:size]
    piece = 5
    res = shuffle(pd.concat([one_feature[:piece], zero_feature[:piece]]))
    for i in range(1, size // piece):
        temp = shuffle(pd.concat([one_feature[piece * i:piece * (i + 1)], zero_feature[piece * i:piece * (i + 1)]]))
        res = pd.concat([res, temp])
    return res


def select(base_url):
    data_frame = read(base_url)
    data_frame = data_frame.sample(frac=1)
    data_frame = balanced_select(data_frame, 20000)
    data_frame.to_csv("train.csv")
