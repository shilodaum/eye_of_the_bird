import laspy
import numpy as np
from extract import geometry
import json

FEATURE_X = "feature_x"
FEATURE_Y = "feature_y"
FEATURE_Z = "feature_z"


def extract_features_from_object(object_file_path, output_file_path):
    laspy_file = laspy.read(object_file_path)
    points = laspy_file.xyz
    points = geometry.get_center(points)

    points = geometry.remove_outliers(points)
    x_diff, y_diff, angle = geometry.find_min_bounding_rectangle(points)
    points = geometry.rotate(points, angle)

    data = {
        FEATURE_X: x_diff,
        FEATURE_Y: y_diff,
        FEATURE_Z: geometry.get_z_diff(points)
    }

    with open(output_file_path, "w") as f:
        json.dump(data, f, indent=4)


def main():
    filepath = r"..\..\objects\box\CC-9.las"
    extract_features_from_object(filepath, "box.json")


if __name__ == '__main__':
    main()
