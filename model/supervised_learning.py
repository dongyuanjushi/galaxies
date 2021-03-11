import pandas as pd
from sklearn.model_selection import cross_validate, train_test_split
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
scoring = ["accuracy"]
colors = {
    "Decision Tree": "red",
    "Random Forest": "blue",
    "SVM": "green",
}
accuracies = {}


# build prediction model
def build(train_data, model):
    train_y, train_x = train_data[labels], train_data[features]
    model.fit(train_x, train_y)


# predict result
def predict(val_data, model):
    val_y, val_x = val_data[labels], val_data[features]
    prediction_y = model.predict_proba(val_x)
    accuracy = model.score(val_x, val_y)
    return accuracy, prediction_y, val_y


def train(train_path):
    total_data = pd.read_csv(train_path)
    train_set, test_set = total_data[:5000],total_data[5000:6000]
    train_x, train_y, test_x, test_y = train_set[features], train_set[labels], test_set[features], test_set[labels]
    modes = ["Decision Tree", "Random Forest", "SVM"]
    K = 10
    for current_mode in modes:
        if current_mode == "Decision Tree":
            model = DecisionTreeClassifier(max_depth=4)
        elif current_mode == "Random Forest":
            model = RandomForestClassifier(max_depth=4)
        else:
            model = svm.SVC(kernel='rbf', probability=True)
        # K cross validation
        score = cross_validate(model, train_x, train_y, cv=K)
        accuracy = score["test_score"]
        if current_mode not in accuracies.keys():
            accuracies.setdefault(current_mode, accuracy)
        model.fit(train_x, train_y)
        pre_y = model.predict_proba(test_x)
        test_y = concat(test_y)
        roc_plot(test_y, pre_y, current_mode)
    for accuracy in accuracies:
        plot_accuracy(accuracies[accuracy], accuracy)
    plt.xlabel('Cross')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.show()


# plot accuracy curve
def plot_accuracy(accuracy, mode):
    epoch = np.array([i for i in range(len(accuracy))])
    plt.plot(epoch, accuracy, 'o-', color=colors[mode], label=mode + ' Accuracy: %.3f' % (np.mean(accuracy)))


# plot ROC Curve
def roc_plot(val_y, pre_y, mode):
    one_fpr, one_tpr, _ = metrics.roc_curve(val_y, pre_y[:, 1])
    one_auc = metrics.auc(one_fpr, one_tpr)
    plt.plot(one_fpr, one_tpr, color='red', label='Positive AUC = %.3f' % one_auc)
    plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Model:{})'.format(mode))
    plt.legend()
    plt.show()


def s_execute(train_path, val_path=None, test_path=None):
    train(train_path)


def plot_feature(data):
    plt.scatter(data["S"], data["G2"], c=data["Class"])
    plt.show()
