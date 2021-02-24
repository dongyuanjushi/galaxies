## Introduction

The dataset about Morphological Classification of Galaxies can be found on https://www.kaggle.com/saurabhshahane/galaxy-classification and can be downloaded freely. The data contains 670560 galaxies from  Sloan Digital Sky Survey Data Release 7 (SDSS-DR7). In this data, independent variables are provided as following: C (Concentration), A (Asymmetry), S (Smoothness), G2 (Gradient Pattern Analysis), H (Entropy). And independent variable is the class of galaxies, either 2 classes (0: elliptical, 1: spiral galaxy) or 3 classes (0: elliptical, 1: non-barred spiral, or 2: barred spiral galaxy). 

## Methods

### Supervised Learning
In this part, three supervised learning approaches are introduced to do prediction work.  

Before modeling, raw data (galaxies.csv) is preprocessed to remove missing data and get shuffled. Train,validation and test data is generated from raw data, according to the ratio of 7:2:1. 

In the evalution stage, k-cross-validation method is used. The main idea of this method is to split data into K pieces, then use (K-1) pieces to train and the rest piece to validate. The validation process for each model will generate K different prediction accuracy and the mean accuracy will be the final evaluation result. ROC curves for each model are plotted and images will be shown for just one round of K. In the following models, K is set to 10, the total number of samples for training and validating is 5000 and class_weight is set as balanced considering uneven distribution of data.  

#### Decision Tree

The main idea of decision tree is to generate different conditions from the data and then to classify. All the conditions learned will finally generate a tree-like model (usually binary tree). In this instance, trial and error experiments have been done to discover that 4 may be appropriate for max depth of decision tree.

The accuracy of each round of K-cross-validation is [0.932 0.918 0.912 0.904 0.942 0.914 0.924 0.924 0.898 0.914] and final accuracy is 0.9182 and ROC curve is shown below. 

![](/home/mark/project/galaxies/Decision Tree.png)

#### Random Forest

Random forest is a extensive method of decision tree. It adopts different decision tree and combine each tree's prediction probability into final one (usually by voting). In this instance, the number of tree estimators is 100 (default) and max depth of each tree is still 4. 

The accuracy of each round of K-cross-validation is [0.972 0.952 0.974 0.968 0.966 0.962 0.964 0.964 0.958 0.968] and final accuracy is 0.9648 and ROC curve is shown below. 

![](/home/mark/project/galaxies/Random Forest.png)

#### SVM (Supporting Vector Machine)

SVM is aimed to find a largest-distanced Hyperplane to make classfications, which means the distance between data point and the Hyperplane should be maximized. SVM model needs kernel function to decide the Hyperplane and linear function is applied in the instance to solve a binary classfication problem. 

The accuracy of each round of K-cross-validation is [0.894 0.898 0.892 0.898 0.908 0.892 0.902 0.872 0.882 0.886] and final accuracy is 0.8924 and ROC curve is shown below. 

![](/home/mark/project/galaxies/SVM.png)