import os

import laspy
import numpy as np
from extract import geometry
import json

FEATURE_X = "feature_x"
FEATURE_Y = "feature_y"
FEATURE_Z = "feature_z"
LABEL = "label"


def extract_features_from_object(object_file_path, output_file_path, label=None):
    laspy_file = laspy.read(object_file_path)
    points = laspy_file.xyz
    points = geometry.get_center(points)

    points = geometry.remove_outliers(points)
    x_diff, y_diff, angle = geometry.find_min_bounding_rectangle(points)
    points = geometry.rotate(points, angle)

    data = {
        FEATURE_X: x_diff,
        FEATURE_Y: y_diff,
        FEATURE_Z: geometry.get_z_diff(points),
        LABEL: label
    }

    with open(output_file_path, "w") as f:
        json.dump(data, f, indent=4)


def main():
    filepath = r"..\..\objects"
    i = 0
    for obj_type in os.listdir(filepath):
        if os.path.isdir(os.path.join(filepath, obj_type)):
            obj_type_path = os.path.join(filepath, obj_type)
            for obj in os.listdir(obj_type_path):
                if obj.endswith(".las"):
                    obj_path = os.path.join(obj_type_path, obj)
                    print(obj_path)
                    extract_features_from_object(obj_path, "..\\..\\labeled_data\\obj" + str(i).zfill(3) + ".json",
                                                 obj_type)
                    i += 1


if __name__ == '__main__':
    main()
