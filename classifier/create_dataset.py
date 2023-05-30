import json
import os
import glob


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
            curr_data.append(json_file['feature_x'])
            curr_data.append(json_file['feature_y'])
            curr_data.append(json_file['feature_z'])
            dataset['data'].append(curr_data)

    return dataset
