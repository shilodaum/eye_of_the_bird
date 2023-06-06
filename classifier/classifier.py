from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import json
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, accuracy_score


FEATURE_X = "feature_x"
FEATURE_Y = "feature_y"
FEATURE_Z = "feature_z"
FEATURE_VOLUME = "feature_volume"
FEATURE_AREA = "feature_area"
FEATURE_AREA_RATIO = "feature_a_ratio"
FEATURE_HEIGHT_RATIO = "feature_h_ratio"
FEATURE_DENSITY = "feature_density"
FEATURE_AERIAL_DENSITY = "feature_a_density"
FEATURES = [FEATURE_X, FEATURE_Y, FEATURE_Z, FEATURE_VOLUME, FEATURE_AREA, FEATURE_AREA_RATIO, FEATURE_HEIGHT_RATIO, FEATURE_DENSITY, FEATURE_AERIAL_DENSITY]


def get_dataset():
    """
    This function creates a dataset from the json files in the folder 'features'.
    :return: a dictionary with the data and the labels.
    """

    dataset = dict()
    dataset['data'] = list()
    dataset['target'] = list()
    path = os.path.join(os.path.dirname(os.getcwd()), r'data\features')

    for file_name in glob.glob(os.path.join(path, '*.json')):
        with open(file_name, encoding='utf-8', mode='r') as curr_file:
            json_file = json.load(curr_file)
            curr_data = list()
            dataset['target'].append(json_file['label'])
            for feature in FEATURES:
                curr_data.append(json_file[feature])
            dataset['data'].append(curr_data)

    return dataset


def train_model():
    """
    This function trains a random forest model on the dataset.
    :return: None
    """
    dataset = get_dataset()  # returns the dataset. the data is saved under 'data', the labels are saved under 'target'

    X = dataset['data']  # input, features
    y = dataset['target']  # output, labels

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


def main():
    train_model()


if __name__ == '__main__':
    main()
