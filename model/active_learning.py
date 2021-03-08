import pandas as pd
from model.supervised_learning import build, predict, features, labels
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt
import numpy as np
import time

batch = 250
accuracies = {}
acc_threshold = 0.96
query_threshold = 15

query_strategies = [
    "Density Weighted",
    "random",
    "Least Confident"
]
colors = {
    "random": "red",
    "Least Confident": "blue",
    "Density Weighted": "green"
}


def train(train_data, val_data):
    current_mode = "Decision Tree"
    for s in query_strategies:
        query_times=0
        print("Strategy: {}".format(s))
        model=DecisionTreeClassifier(max_depth=4,class_weight='balanced')
        # model = RandomForestClassifier(max_depth=4, class_weight='balanced')
        # model = SVC(probability=True, class_weight='balanced', C=1)
        next_data = train_data[:batch]
        res_data = train_data[batch:]
        for k in range(1, len(train_data) // batch):
            print("Round {} is training!".format(k))
            build(next_data, model)
            pre_y, val_y = None, None
            if s != "random":
                acc, pre_y, val_y = predict(res_data, model)
            score, _, _ = predict(val_data, model)
            if s not in accuracies.keys():
                accuracies.setdefault(s, [])
            accuracies[s].append(score)
            if score >= acc_threshold or query_times>=query_threshold:
                break
            print("Round {} finished training!".format(k))
            print("Accuracy on the test data is %.4f" % score)
            print('-------------------------\n\n')
            next_data, res_data = query(next_data, res_data, pre_y, val_y, s)
            if s!="random":
                query_times += 1
    for s in query_strategies:
        acc_plot(s, accuracies[s])
    plt.title('Accuracy Curve ( Model: {})'.format(current_mode))
    plt.legend()
    plt.show()


def query(next_data, res_data, pre_y=None, val_y=None, strategy="random"):
    if strategy == "random":
        return pd.concat([next_data, res_data[:batch]]), res_data[batch:]
    elif strategy == "Least Confident":
        return least_confident(next_data, res_data, pre_y, val_y)
    elif strategy == "Density Weighted":
        return density_weighted(next_data, res_data, pre_y, val_y)


def least_confident(next_data, res_data, pre_y, val_y):
    confident = np.max(pre_y, axis=1).reshape(len(pre_y), 1)
    res_data = res_data.copy()
    res_data["temp"] = confident
    res_data = res_data.sort_values(by="temp", ascending=True)
    # print(res_data[:batch])
    res_data.drop(columns=["temp"])
    return pd.concat([next_data, res_data[:batch]]), res_data[batch:]


def density_weighted(next_data, res_data, pre_y, val_y):
    # calculate confidence
    confident = np.max(pre_y, axis=1).reshape(len(pre_y), 1)
    res_data = res_data.copy()
    # calculate similarity
    left = np.array(next_data[features])
    right = np.array(res_data[features])
    right_sq, left_sq = np.sum(np.square(right), axis=1), np.sum(np.square(left).T, axis=0)
    right_sq = right_sq.reshape((len(right_sq), 1))
    left_sq = left_sq.reshape((1, len(left_sq)))
    mul = np.sqrt(np.dot(right_sq, left_sq))
    similarity = np.average(np.dot(right, left.T) / mul, axis=1)
    similarity = np.reshape(similarity, (len(similarity), 1))
    # weighted density = confidence * similarity
    res_data["temp"] = confident * similarity
    res_data = res_data.sort_values(by="temp", ascending=True)
    res_data.drop(columns=["temp"])
    return pd.concat([next_data, res_data[:batch]]), res_data[batch:]


def acc_plot(strategy, accuracy):
    epoch = [i for i in range(len(accuracy))]
    plt.plot(epoch, accuracy, 'o-', color=colors[strategy], label=strategy + " Accuracy: %.3f" % np.mean(accuracy))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')


def a_execute(train_data_path, val_data_path=None):
    raw_train_data = pd.read_csv(train_data_path)
    train_data = raw_train_data
    train_data, val_data = train_data[:5000], train_data[5000:6000]
    train(train_data, val_data)
