import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error
import numpy as np

features = ["TType", "K", "C", "A", "S", "G2", "H"]


def build(train_path):
    train_data = pd.read_csv(train_path, low_memory=False)
    train_y = train_data.Class[:1]
    train_X = train_data[features][:1]
    model = DecisionTreeClassifier(random_state=1)
    model.fit(train_X, train_y)
    return model


def concat(test_y):
    return np.array([i for i in test_y.values])


def predict(model, test_path):
    test_data = pd.read_csv(test_path, low_memory=False)
    test_X = test_data[features][:8096]
    predict_res = model.predict(test_X)
    test_y = concat(test_data["Class"][:8096])
    cal_accuracy(predict_res,test_y)


def cal_accuracy(predict_res, test_y):
    accuracy= np.sum((predict_res-test_y)==0)/len(predict_res)
    print(accuracy)


def execute(train_path, val_path, test_path):
    model = build(train_path)
    predict(model, test_path)
