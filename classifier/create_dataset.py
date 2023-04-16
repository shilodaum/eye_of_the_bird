import json
import os


def get_dataset():
    dataset = dict()
    dataset['data'] = list()
    dataset['target'] = list()

    path = 'C:\magdad\eye_of_the_bird\classifier\Features_1'
    for json_file in os.walk(path):
        curr_file = json.load(json_file)
        curr_data = list()
        for feature in curr_file:
            curr_data.append(feature)
        dataset['data'].append(curr_data)

    return dataset






