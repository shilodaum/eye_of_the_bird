from classifier.classifier import train_model
from extract.extract_features_from_object import extract_all_objects
from subtract.CLI_subtract_to_cc import subtract_clouds
from os.path import join as pjoin
import os

SCANS_PATH = r"data\scans"
OBJECTS_PATH = r"data\objects"
FEATURES_PATH = r"data\features"
REFERENCE_CLOUD_PATH = pjoin(SCANS_PATH, "reference.las")
NEW_CLOUD_PATH = pjoin(SCANS_PATH, "input.las")
MIN_POINTS_PER_CLOUD = 500
SHIFT = [-692300, -3616100, 0]
BOX = [692297, 3616248, 300, 692335, 3616288, 400]
TRAIN_SIZE = 0.5


def main():
    print("Subtracting clouds, this may take a while...")
    # subtract_clouds(REFERENCE_CLOUD_PATH, NEW_CLOUD_PATH, OBJECTS_PATH, SHIFT, MIN_POINTS_PER_CLOUD, BOX)
    print("Label the objects in the folder 'data/objects' by moving them to the folders 'data/objects/<label>'")
    print("Press enter when done")
    input()
    print("Extracting features")
    extract_all_objects(r"data\objects", r"data\features")
    print("Training model")
    train_model(r"data\features", TRAIN_SIZE)
    print("Done")


if __name__ == "__main__":    # Usable functions:
    main()