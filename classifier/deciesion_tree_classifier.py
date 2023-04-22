# importing necessary libraries
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from create_dataset import get_dataset

# TODO - fix error
def decision_tree_model():
    """

    :return:
    """

    # loading dataset
    dataset = get_dataset()

    X = dataset['data'] # input, features.
    Y = dataset['target'] # output, laybel.

    # dividing X, y into train and test data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 0)

    # training a DescisionTreeClassifier

    dtree_model = DecisionTreeClassifier(max_depth = 2).fit(X_train, Y_train)
    dtree_predictions = dtree_model.predict(X_test)

    # creating a confusion matrix
    cm = confusion_matrix(Y_test, dtree_predictions)
