import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
import sklearn.tree as tree
import numpy as np
import graphviz

features = ["TType", "K", "C", "A", "S", "G2", "H"]


def build(train_path):
    train_data = pd.read_csv(train_path, low_memory=False)
    # train_y = train_data.Class[0:5]
    # train_X = train_data[features][0:5]
    train_y = train_data.Class[0:200]
    train_X = train_data[features][0:200]
    model = DecisionTreeClassifier()
    # model=svm.SVC(gamma='scale')
    model.fit(train_X, train_y)
    return model


def validate(model,val_path):
    val_data=pd.read_csv(val_path,low_memory=False)
    # val_x=val_data[features][:1000]
    # val_y=val_data["Class"][:1000]
    val_x = val_data[features][:100]
    val_y = val_data["Class"][:100]
    score=model.score(val_x,val_y)
    print("Accuracy is {}".format(score))


def concat(test_y):
    return np.array([i for i in test_y.values])


def predict(model, test_path):
    dot_data=tree.export_graphviz(model)
    graph=graphviz.Source(dot_data)
    graph.render("galaxy")
    # test_data = pd.read_csv(test_path, low_memory=False)
    # test_X = test_data[features][:10]
    # predict_res = model.predict(test_X)
    # print(predict_res)
    # test_y = concat(test_data["Class"][:8096])
    # cal_accuracy(predict_res,test_y)


def cal_accuracy(predict_res, test_y):
    accuracy= np.sum((predict_res-test_y)==0)/len(predict_res)
    print(accuracy)


def execute(train_path, val_path, test_path):
    # print(float("-9.72747802734e-05"))
    model = build(train_path)
    # validate(model,val_path)
    predict(model, test_path)
