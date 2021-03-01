import pandas as pd
from model.supervised_learning import build, predict, standard
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from util.select import balanced_select

import matplotlib.pyplot as plt
import numpy as np

batch = 100
accuracy = []
threshold = 0.96


def train(train_data, val_data):
    # train_data = standard(train_data)
    next_data = train_data[:batch]
    res_data = train_data[batch:]
    # model = svm.SVC(probability=True, class_weight="balanced")
    model= RandomForestClassifier(class_weight='balanced',max_depth=4)
    strategy = "Least Confident"
    for k in range(1, len(train_data) // batch):
        print("Round {} is training!".format(k))
        build(next_data, model)
        acc, pre_y, val_y = predict(res_data, model)
        score, _, _ = predict(val_data, model)
        accuracy.append(score)
        if score > threshold:
            break
        print("Round {} finished training!".format(k))
        print("Accuracy on the test data is %.4f" % score)
        print('-------------------------\n\n')
        next_data, res_data = query(res_data, pre_y, val_y, strategy)
    acc_plot()


def query(res_data, pre_y=None, val_y=None, strategy="random"):
    if strategy == "random":
        # print(len(res_data))
        shuffled_data = shuffle(res_data)
        return shuffled_data[:batch], shuffled_data[batch:]
    elif strategy == "Least Confident":
        return least_confident(res_data, pre_y, val_y)


def least_confident(res_data, pre_y, val_y):
    confident = np.max(pre_y, axis=1).reshape(len(pre_y), 1)
    res_data = res_data.copy()
    res_data["temp"] = confident
    res_data = res_data.sort_values(by="temp", ascending=True)
    return res_data[:batch], res_data[batch:]


def acc_plot(strategy="random"):
    epoch = [i for i in range(len(accuracy))]
    plt.plot(epoch, accuracy, color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve (Strategy:{})'.format(strategy))
    plt.legend()
    plt.show()


def plot_dis(x):
    pass


def a_execute(train_data_path, val_data_path=None):
    raw_train_data = pd.read_csv(train_data_path)
    train_data = raw_train_data
    # train_data = shuffle(balanced_select(raw_train_data, 5000 // 2))
    # raw_val_data = pd.read_csv(val_data_path)
    train_data, val_data = train_data[:8000], train_data[8000:8100]
    train(train_data, val_data)
