import json
import os
import glob

def get_dataset():  # TODO - add docstring
    """

    :return: The complete dataset in the form of a dictionary.
     the data is saved under 'data', the labels are saved under 'target'
    """

    dataset = dict()
    dataset['data'] = list()
    dataset['target'] = list()

    path = r"C:\magdad\eye_of_the_bird\classifier\labeled_data"  # TODO - change to relative path, and add r"" to the path
    # path = input("Enter path to get  json files from")


    for file_name in glob.glob(os.path.join(path, '*.json')):
        with open(file_name, encoding='utf-8', mode='r') as curr_file:
            json_file = json.load(curr_file)
            curr_data = list()
            first = True
            for feature in json_file:
                if first:
                    dataset['target'].append(json_file[feature])
                    first = False
                else:
                    curr_data.append(json_file[feature])
            dataset['data'].append(curr_data)

    return dataset






