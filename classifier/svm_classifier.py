from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets
import matplotlib.pyplot as plt
import numpy as np
from create_dataset import get_dataset


def svm_model():
    """
    This function creates an SVM ML model.
    The function gets a dataset from the function get_dataset.

    :return: nothing at this time.
    """
    dataset = get_dataset() # returns the dataset. the data is saved under 'data', the labels are saved under 'target'

    X = dataset['data']  # input, features.
    Y = dataset['target']  # output, label.

    # split the dataset to train and test. 0.8/0.2.
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8)

    C_const = 1.5  # const that determines how much you want to avoid misclassification.
                 # For large values of C, the optimization will choose a smaller-margin hyperplane

    linear_model = svm.SVC(kernel='linear', C=C_const, decision_function_shape='ovo').fit(X_train, Y_train)
    rbf_model = svm.SVC(kernel='rbf', gamma=1, C=C_const, decision_function_shape='ovo').fit(X_train, Y_train)
    poly_model = svm.SVC(kernel='poly', degree=3, C=C_const, decision_function_shape='ovo').fit(X_train, Y_train)
    sig_model = svm.SVC(kernel='sigmoid', C=C_const, decision_function_shape='ovo').fit(X_train, Y_train)

    """
    h = 0.1

    # create the meshgrid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
    # create the title that will be shown on the plot
    titles = ['Linear kernel','RBF kernel','Polynomial kernel','Sigmoid kernel']
    """

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
