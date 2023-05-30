from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from create_dataset import get_dataset
import seaborn as sns
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def svm_model():
    """
    This function creates an SVM ML model.
    The function gets a dataset from the function get_dataset.

    :return: nothing at this time.
    """
    dataset = get_dataset()  # returns the dataset. the data is saved under 'data', the labels are saved under 'target'
    print(dataset)
    labels = ['person', 'cylinder_on_tripod', 'barrel', 'car', 'french_bed', 'chair', 'box',
              'table', 'gazebo']

    X = dataset['data']  # input, features.
    Y = dataset['target']  # output, label.

    # split the dataset to train and test. 0.8/0.2.
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.5, stratify=Y)

    C_const = 1.0  # const that determines how much you want to avoid misclassification.
    # For large values of C, the optimization will choose a smaller-margin hyperplane
    print(np.unique(Y_train))
    labels = list(np.unique(Y_train))
    sample_weight = compute_class_weight('balanced', y=Y_train, classes=np.unique(Y_train))
    print(sample_weight)
    print(Y_train)
    Y_train_weights = [sample_weight[labels.index(i)] for i in Y_train]
    print(Y_train_weights)
    model = svm.SVC(kernel='linear').fit(X_train, Y_train)
    # rbf_model = svm.SVC(kernel='rbf', gamma=1, C=C_const, decision_function_shape='ovo').fit(X_train, Y_train)
    # poly_model = svm.SVC(kernel='poly', degree=3, C=C_const, decision_function_shape='ovo').fit(X_train, Y_train)
    # sig_model = svm.SVC(kernel='sigmoid', C=C_const, decision_function_shape='ovo').fit(X_train, Y_train)


    # calculate sample weight on test set

    # rf_model = GradientBoostingClassifier(max_depth=4).fit(X_train, Y_train, sample_weight=Y_train_weights)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(Y_test, y_pred)

    # print accuracy
    print('Accuracy:', accuracy)

    # creating a confusion matrix
    print(Counter(Y))
    # print(set(Y_test))
    # print(len(set(Y_test)))

    cm = confusion_matrix(Y_test, y_pred, labels=labels)
    # cm_poly = confusion_matrix(Y_test, poly_pred)
    # cm_rbf = confusion_matrix(Y_test, rbf_pred, labels=labels)
    # cm_rf = confusion_matrix(Y_test, rf_pred, labels=labels)
    # cm_sig = confusion_matrix(Y_test, sig_pred)
    ax = sns.heatmap(cm, annot=True, xticklabels=labels, yticklabels=labels, cmap='Blues')

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

    print(cm)
    # print(cm_poly)
    # print(cm_rf)
    # print(cm_sig)
