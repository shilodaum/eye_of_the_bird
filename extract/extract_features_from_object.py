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
FEATURE_VOLUME = "feature_volume"
FEATURE_AREA = "feature_area"
FEATURE_AREA_RATIO = "feature_a_ratio"
FEATURE_HEIGHT_RATIO = "feature_h_ratio"
FEATURE_DENSITY = "feature_density"
FEATURE_AERIAL_DENSITY = "feature_a_density"
LABEL = "label"


def extract_features_from_object(object_file_path, output_file_path, label=None):
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
