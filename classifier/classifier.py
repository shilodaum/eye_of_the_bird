from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from joblib import dump, load
from create_dataset import get_dataset
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, accuracy_score


def train_model():
    """
    This function trains a random forest model on the dataset.
    :return: None
    """
    dataset = get_dataset()  # returns the dataset. the data is saved under 'data', the labels are saved under 'target'

    X = dataset['data']  # input, features
    y = dataset['target']  # output, labels
    print(Counter(y))

    # split the dataset to train and test. 0.3/0.7
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3, stratify=y)

    # calculate sample weight on train set
    labels = list(np.unique(y_train))
    sample_weight = compute_class_weight('balanced', y=y_train, classes=np.unique(y_train))
    train_weights = [sample_weight[labels.index(i)] for i in y_train]

    # train random forest model. you can change the model here, we tried linear SVM and it also worked well
    model = RandomForestClassifier(max_depth=5)
    model.fit(X_train, y_train, sample_weight=train_weights)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    # print accuracy
    print('Accuracy:', accuracy)

    # show confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm, annot=True, xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

    dump(model, r'model\model.joblib')


def predict():
    """
    This function predicts the label of a new sample.
    """
    model = load(r'model\model.joblib')
    data = get_dataset()
    prediction = model.predict(data['data'])
    print(prediction)
