import pandas as pd
import os
import cv2

import numpy as np

classified_data = ['Error',
                   # 'ML2classes',
                   'CNN2classes1stClass', 'CNN2classes1stClassPerc',
                   'CNN2classes2ndClass', 'CNN2Classes2ndClassPerc', 'CNN3classes1stClass',
                   'CNN3Classes1stClassPerc', 'CNN3Classes2ndClass',
                   'CNN3Classes2ndClassPerc', 'CNN3Classes3rdClass',
                   'CNN3Classes3rdClassPerc']

drop_list = ['Class2.1', 'Class2.2',
             'Class3.1', 'Class3.2', 'Class4.1', 'Class4.2', 'Class5.1', 'Class5.2',
             'Class5.3', 'Class5.4', 'Class6.1', 'Class6.2',
             # 'Class7.1', 'Class7.2','Class7.3',
             'Class8.1', 'Class8.2', 'Class8.3', 'Class8.4', 'Class8.5',
             'Class8.6', 'Class8.7', 'Class9.1', 'Class9.2', 'Class9.3', 'Class10.1',
             'Class10.2', 'Class10.3', 'Class11.1', 'Class11.2', 'Class11.3',
             'Class11.4', 'Class11.5', 'Class11.6']


def read(url):
    # remove missing data and redundant classified labels
    galaxies_csv = url
    data_frame = pd.read_csv(galaxies_csv, low_memory=False)
    train_csv = data_frame.rename(columns={'ML2classes': "Class"}).drop(columns=classified_data)
    return train_csv[~train_csv["Class"].isin(['U', '--'])]


def process(url):
    train_data = pd.read_csv(url, low_memory=False).drop(columns=drop_list)
    # print(train_data[["Class1.1","Class1.2","Class1.3"]].values)
    # print(np.argmax(train_data[["Class1.1","Class1.2","Class1.3"]].values,axis=1))
    para = train_data[["Class1.1", "Class1.2", "Class1.3"]].values
    train_data["Category"] = np.argmax(para, axis=1)
    train_data["Confidence"] = np.max(para, axis=1) * 2 + np.min(para, axis=1) - 1
    train_data = train_data[train_data["Confidence"] > 0.9]
    print(train_data["Category"].value_counts())
    return train_data


def concat(raw):
    return np.array([i for i in raw.values])


def balanced_select(raw_data, size):
    one_featured = raw_data.loc[raw_data['Class'] == 1][:size]
    zero_feature = raw_data.loc[raw_data['Class'] == 0][:size]
    return pd.concat([one_featured, zero_feature])


def combine(base_url, csv_url):
    classes = process(csv_url)
    combined_data = {}
    column_name = ["GalaxyID"]
    column_name.extend("Pixel%d" % i for i in range(64*64))
    j = 0
    for image in sorted(os.listdir(base_url)):
        Id = int(image[:len(image) - 4])
        if Id not in classes["GalaxyID"].values:
            continue
        img_path = os.path.join(base_url, image)
        img_data = [Id]
        src = cv2.resize(cv2.imread(img_path),(32,32))
        img = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        img = img.flatten() / 255.0
        # cv2.namedWindow("gray-image")
        # cv2.imshow("gray-image",img)
        # cv2.waitKey()
        img_data.extend(img)
        j += 1
        print("Round {}".format(j))
        # print(len(img_data))
        for i in range(len(img_data)):
            if column_name[i] not in combined_data.keys():
                combined_data.setdefault(column_name[i], [])
            combined_data[column_name[i]].append(img_data[i])
        if "Category" not in combined_data.keys():
            combined_data.setdefault("Category",[])
        if "Confidence" not in combined_data.keys():
            combined_data.setdefault("Confidence",[])
        current=classes[classes["GalaxyID"].values == Id]
        category=current["Category"].values[0]
        confidence=current["Confidence"].values[0]
        combined_data["Category"].append(category)
        combined_data["Confidence"].append(confidence)
    data_frame=pd.DataFrame(combined_data)
    data_frame.to_csv("images.csv")
    print(pd.read_csv("solutions.csv")["Class3.1"].head())
    print(pd.read_csv("solutions.csv")["Class3.2"].head())


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
