import pandas as pd
import os
import numpy as np

classified_data = ['Error',
                   # 'ML2classes',
                   'CNN2classes1stClass', 'CNN2classes1stClassPerc',
                   'CNN2classes2ndClass', 'CNN2Classes2ndClassPerc', 'CNN3classes1stClass',
                   'CNN3Classes1stClassPerc', 'CNN3Classes2ndClass',
                   'CNN3Classes2ndClassPerc', 'CNN3Classes3rdClass',
                   'CNN3Classes3rdClassPerc']


def read(base_url):
    # remove missing data and redundant classified labels
    galaxies_csv = os.path.join(base_url, "galaxies.csv")
    data_frame = pd.read_csv(galaxies_csv, low_memory=False)
    train_csv = data_frame.rename(columns={'ML2classes': "Class"}).drop(columns=classified_data)
    return train_csv[~train_csv["Class"].isin(['U', '--'])]


def concat(raw):
    return np.array([i for i in raw.values])


def balanced_select(raw_data, size):
    one_featured = raw_data.loc[raw_data['Class'] == 1][:size]
    zero_feature = raw_data.loc[raw_data['Class'] == 0][:size]
    return pd.concat([one_featured, zero_feature])


def select(base_url):
    data_frame = read(base_url)
    # print(data_frame.head())
    data_frame = data_frame.sample(frac=1)
    train_num, val_num = int(0.7 * len(data_frame)), int(0.2 * len(data_frame))
    train_csv = data_frame[:train_num]
    val_csv = data_frame[train_num:(train_num + val_num)]
    test_csv = data_frame.iloc[(train_num + val_num):]
    train_csv.to_csv("train.csv", index=False)
    val_csv.to_csv("val.csv", index=False)
    test_csv.to_csv("test.csv", index=False)
