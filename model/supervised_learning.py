import pandas as pd
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn import metrics
from util.select import concat
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import joblib
import numpy as np
import matplotlib.pyplot as plt

features = ["TType", "K", "C", "A", "S", "G2", "H"]
labels = "Class"


def build(train_data, model):
    train_y, train_x = train_data[labels], train_data[features]
    model.fit(train_x, train_y)
    # model.partial_fit(train_x,train_y,classes=np.array([0,1]))


def predict(val_data, model):
    val_y, val_x = val_data[labels], val_data[features]
    prediction_y = model.predict_proba(val_x)
    accuracy = model.score(val_x, val_y)
    return accuracy, prediction_y, val_y


def k_cross_validation(train_path):
    total_data = pd.read_csv(train_path)[:5000]
    K = 10
    mode = ["DecisionTree", "SVM", "RandomForest"]
    piece = len(total_data) // K
    accuracy = []
    model = SGDClassifier()
    for i in range(K):
        print("Round {}".format(i + 1))
        val_data = total_data[(piece * i):(piece * (i + 1))]
        train_data = pd.concat([total_data[:piece * i], total_data[piece * (i + 1):]])
        build(train_data, model)
        acc, pre_y, val_y = predict(val_data, model)
        accuracy.append(acc)
        pre_y = np.array(pre_y)
        val_y = concat(val_y)
        # roc_plot(val_y, pre_y, mode[2], i + 1)
    accuracy = np.array(accuracy)
    # print("Each Round Accuracy of the {} model is {}".format(accuracy))
    # print("The Average Accuracy of K-cross Validation for {} Model is {}".format(np.average(accuracy)))


def roc_plot(val_y, pre_y, mode, cross):
    one_fpr, one_tpr, _ = metrics.roc_curve(val_y, pre_y[:, 1])
    one_auc = metrics.auc(one_fpr, one_tpr)
    plt.plot(one_fpr, one_tpr, color='red', label='Round %d: Positive AUC = %.2f' % (cross, one_auc))
    plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Binary Classification (Model:{})'.format(mode))
    plt.legend()
    plt.show()


def s_execute(train_path, val_path=None, test_path=None):
    k_cross_validation(train_path)


def standard(raw_data):
    raw_data = raw_data.copy()
    X = raw_data[features]
    X = (X - X.mean()) / X.std()
    raw_data.drop(columns=features)
    raw_data[features] = X
    return raw_data
