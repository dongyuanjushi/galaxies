import pandas as pd
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn import metrics
from util.select import concat
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import joblib
import numpy as np
import matplotlib.pyplot as plt

features = ["TType", "K", "C", "A", "S", "G2", "H"]


def build(train_data, mode):
    train_y, train_x = train_data["Class"], train_data[features]
    model_path = "weights/" + mode + ".model"
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    elif mode == "SVM":
        model = svm.SVC(kernel='linear', probability=True, class_weight='balanced')
    # elif mode == "SGD":
    #     model = SGDClassifier(penalty='l2', class_weight='balanced')
    elif mode == "RandomForest":
        model = RandomForestClassifier(class_weight='balanced', max_features='sqrt', max_depth=4)
    elif mode == "MLP":
        model = MLPClassifier(max_iter=1000,learning_rate_init=0.01)
    # default use Decision Tree Models
    else:
        model = DecisionTreeClassifier(class_weight='balanced', max_depth=4)
    model.fit(train_x, train_y)
    joblib.dump(model, model_path)
    return model


def predict(val_data, model):
    val_y, val_x = val_data["Class"], val_data[features]
    prediction_y = model.predict_proba(val_x)
    accuracy = model.score(val_x, val_y)
    return accuracy, prediction_y, val_y


def k_cross_validation(train_path):
    total_data = pd.read_csv(train_path)[:5000]
    K = 10
    mode = ["DecisionTree", "SVM", "RandomForest"]
    piece = len(total_data) // K
    accuracy = []
    for i in range(K):
        val_data = total_data[(piece * i):(piece * (i + 1))]
        train_data = pd.concat([total_data[:piece * i], total_data[piece * (i + 1):]])
        model = build(train_data, mode[2])
        acc, pre_y, val_y = predict(val_data, model)
        accuracy.append(acc)
        pre_y = np.array(pre_y)
        val_y = concat(val_y)
        # roc_plot(val_y, pre_y, mode[2], i + 1)
    accuracy = np.array(accuracy)
    print("Each Round Accuracy of the {} model is {}".format(mode[2], accuracy))
    print("The Average Accuracy of K-cross Validation for {} Model is {}".format(mode[2], np.average(accuracy)))


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


def execute(train_path, val_path, test_path):
    k_cross_validation(train_path)
