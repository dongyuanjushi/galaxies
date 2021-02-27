import pandas as pd
from model.supervised_learning import build, predict
from sklearn.utils import shuffle
from util.select import balanced_select
import matplotlib.pyplot as plt

batch = 100
accuracy = []
threshold = 0.90


def train(train_data, val_data):
    next_data = train_data[:batch]
    res_data = train_data[batch:]
    for k in range(1, len(train_data) // batch):
        print("Round {} is training!".format(k))
        model = build(next_data, "SVM")
        acc, pre_y, val_y = predict(res_data, model)
        score, _, _ = predict(val_data, model)
        accuracy.append(score)
        if score > threshold:
            break
        next_data, res_data = query(res_data, pre_y, val_y)
        print("Round {} finished training!".format(k))
        print("Accuracy on the test data is %.4f" % score)
        print('-------------------------\n\n')
        # print(next_data.head())
        # next_data, res_data = query(res_data)
    acc_plot()


def query(res_data, pre_y=None, val_y=None, strategy="random"):
    if strategy == "random":
        shuffled_data = shuffle(res_data)
        return shuffled_data[:batch], shuffled_data[batch:]
    else:
        return None


def least_confident(res_data):
    pass


def acc_plot(strategy="random"):
    epoch = [i for i in range(len(accuracy))]
    plt.plot(epoch, accuracy, color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve (Strategy:{})'.format(strategy))
    plt.legend()
    plt.show()


def execute(train_data_path, val_data_path):
    raw_train_data = pd.read_csv(train_data_path)
    train_data = shuffle(balanced_select(raw_train_data, 5000 // 2))
    # raw_val_data = pd.read_csv(val_data_path)
    val_data = shuffle(balanced_select(raw_train_data, 2000 // 2))
    train(train_data, val_data)
