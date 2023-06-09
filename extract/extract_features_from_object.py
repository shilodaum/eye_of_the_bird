import os

import laspy
import numpy as np
from extract import geometry
import json
from os.path import join as pjoin, isdir, isfile

FEATURE_X = "feature_x"
FEATURE_Y = "feature_y"
FEATURE_Z = "feature_z"
FEATURE_VOLUME = "feature_volume"
FEATURE_AREA = "feature_area"
FEATURE_AREA_RATIO = "feature_a_ratio"
FEATURE_HEIGHT_RATIO = "feature_h_ratio"
FEATURE_DENSITY = "feature_density"
FEATURE_AERIAL_DENSITY = "feature_a_density"
LABEL = "label"


def extract_features_from_object(object_file_path, output_file_path, label=None):
    """
    Extract features from the object file and save them to the output file
    :param object_file_path: the path to the object file
    :param output_file_path: the path to the output file
    :param label: the label of the object
    :return: None
    """
    laspy_file = laspy.read(object_file_path)

    points = laspy_file.xyz

    intensities = laspy_file.intensity
    num_points = len(points)
    points = geometry.get_center(points)

    points = geometry.remove_outliers(points)
    x_diff, y_diff, angle = geometry.find_min_bounding_rectangle(points)
    z_diff = geometry.get_z_diff(points)
    points = geometry.rotate(points, angle)
    area = x_diff * y_diff
    volume = x_diff * y_diff * z_diff
    data = {
        FEATURE_X: x_diff,
        FEATURE_Y: y_diff,
        FEATURE_Z: z_diff,
        FEATURE_AREA: area,
        FEATURE_VOLUME: volume,
        FEATURE_AREA_RATIO: y_diff / x_diff,
        FEATURE_HEIGHT_RATIO: z_diff / (x_diff * y_diff),
        FEATURE_DENSITY: num_points / volume,
        FEATURE_AERIAL_DENSITY: num_points / area,
    }
    if label is not None:
        data[LABEL] = label

    with open(output_file_path, "w") as f:
        json.dump(data, f, indent=4)


def extract_all_objects(input_folder, output_folder):
    """
    Extract features from all the objects in the input folder and save them to the output folder
    """
    # counter
    file_counter = 0
    for obj_type in os.listdir(input_folder):
        # iterate all folders
        if isdir(pjoin(input_folder, obj_type)):
            obj_type_path = pjoin(input_folder, obj_type)
            # iterate all files
            for obj in os.listdir(obj_type_path):
                if obj.endswith(".las"):
                    # extract features from the object
                    obj_path = pjoin(obj_type_path, obj)
                    print(obj_path)
                    json_path = f"obj{str(file_counter).zfill(3)}.json"
                    extract_features_from_object(obj_path, pjoin(output_folder, json_path), obj_type)
                    file_counter += 1


if __name__ == '__main__':
    extract_all_objects()
