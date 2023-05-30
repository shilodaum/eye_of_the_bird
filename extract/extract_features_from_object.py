import os

import laspy
import numpy as np
from extract import geometry
import json
from os.path import join as pjoin, isdir, isfile

IN_FOLDER = r"../data/objects"
OUT_FOLDER = r"../data/features"

FEATURE_X = "feature_x"
FEATURE_Y = "feature_y"
FEATURE_Z = "feature_z"
FEATURE_AVG_INTENSITY = "feature_avg_intensity"
LABEL = "label"


def extract_features_from_object(object_file_path, output_file_path, label=None):
    laspy_file = laspy.read(object_file_path)
    points = laspy_file.xyz
    average_intensity = np.mean(laspy_file.intensity)
    points = geometry.get_center(points)

    points = geometry.remove_outliers(points)
    x_diff, y_diff, angle = geometry.find_min_bounding_rectangle(points)
    points = geometry.rotate(points, angle)

    data = {
        FEATURE_X: x_diff,
        FEATURE_Y: y_diff,
        FEATURE_Z: geometry.get_z_diff(points),
        FEATURE_AVG_INTENSITY: average_intensity,
        LABEL: label
    }

    with open(output_file_path, "w") as f:
        json.dump(data, f, indent=4)


def main():
    i = 0
    for obj_type in os.listdir(IN_FOLDER):
        if isdir(pjoin(IN_FOLDER, obj_type)):
            obj_type_path = pjoin(IN_FOLDER, obj_type)
            for obj in os.listdir(obj_type_path):
                if obj.endswith(".las"):
                    obj_path = pjoin(obj_type_path, obj)
                    print(obj_path)
                    json_path = f"obj{str(i).zfill(3)}.json"
                    extract_features_from_object(obj_path, pjoin(OUT_FOLDER, json_path), obj_type)
                    i += 1


if __name__ == '__main__':
    main()
