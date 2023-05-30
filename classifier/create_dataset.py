import json
import os
import glob

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
