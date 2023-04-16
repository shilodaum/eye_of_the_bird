from sklearn.linear import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets
import matplotlib.pyplot as plt
import numpy as np

from create_dataset import get_dataset

def svm_model():
    dataset = get_dataset()

    X = dataset['data'] # input, features.
    Y = dataset['target'] # output, laybel.

    # split the dataset to train and test. 0.8/0.2.
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, Z, train_size=0.8)

    C_const = 1 # const that determines

    linear_model = svm.SVC(kernel='linear', C=C_const, decision_function_shape='ovo').fit(X_train, Y_train)
    rbf_model = svm.SVC(kernel='rbf', gamma=1, C=C_const, decision_function_shape='ovo').fit(X_train, Y_train)
    poly_model = svm.SVC(kernel='poly', degree=3, C=C_const, decision_function_shape='ovo').fit(X_train, Y_train)
    sig_model = svm.SVC(kernel='sigmoid', C=C_const, decision_function_shape='ovo').fit(X_train, Y_train)

    h = 0.1

    #create the meshgrid
    X_min, X_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    Y_min, Y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    XX, YY = np.meshgrid(np.arange(X_min, X_max, h),np.arange(Y_min, Y_max, h))

    linear_pred = linear_model.predict(X_test)
    poly_pred = poly_model.predict(X_test)
    rbf_pred = rbf_model.predict(X_test)
    sig_pred = sig_model.predict(X_test)


    accuracy_lin = linear_model.score(X_test, Y_test)
    accuracy_poly = poly_model.score(X_test, Y_test)
    accuracy_rbf = rbf_model.score(X_test, Y_test)
    accuracy_sig = sig_model.score(X_test, Y_test)

    # print accuracy
    print('Accuracy Linear Kernel:', accuracy_lin)
    print('Accuracy Polynomial Kernel:', accuracy_poly)
    print('Accuracy Radial Basis Kernel:', accuracy_rbf)
    print('Accuracy Sigmoid Kernel:', accuracy_sig)

    # creating a confusion matrix
    cm_lin = confusion_matrix(Y_test, linear_pred)
    cm_poly = confusion_matrix(Y_test, poly_pred)
    cm_rbf = confusion_matrix(Y_test, rbf_pred)
    cm_sig = confusion_matrix(Y_test, sig_pred)
    print(cm_lin)
    print(cm_poly)
    print(cm_rbf)
    print(cm_sig)