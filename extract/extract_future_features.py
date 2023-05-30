import numpy as np
from sklearn import linear_model
import laspy
from extract import geometry
from matplotlib import pyplot as plt

THRESHOLD = 0.03


def find_plane(points, n_iteration=1000, threshold=THRESHOLD):
    """
    Finds the planes in the points
    :param points: the points
    :param n_iteration: the number of iterations
    :param threshold: the threshold for the distance
    :return: the plane
    """
    # create the model
    ransac = linear_model.RANSACRegressor(residual_threshold=threshold, max_trials=n_iteration)
    # fit the model
    ransac.fit(points[:, :2], points[:, 2])
    # get the plane
    plane = np.array([ransac.estimator_.coef_[0], ransac.estimator_.coef_[1], -1, ransac.estimator_.intercept_])
    # normalize the plane
    plane = plane / np.linalg.norm(plane[:3])

    return plane


def find_plane_inliers(points, plane, threshold=THRESHOLD):
    """
    Finds the inliers of the points
    :param points: the points
    :param plane: the plane
    :param threshold: the threshold for the distance
    :return: the inliers
    """
    # find the distance of the points to the plane
    dist = np.abs(np.dot(points, plane[:3]) + plane[3])
    # find the inliers
    inliers = points[dist < threshold, :]
    return inliers


def find_plane_outliers(points, plane, threshold=THRESHOLD):
    """
    Finds the outliers of the points
    :param points: the points
    :param plane: the plane
    :param threshold: the threshold for the distance
    :return: the outliers
    """
    # find the distance of the points to the plane
    dist = np.abs(np.dot(points, plane[:3]) + plane[3])
    # find the outliers
    outliers = points[dist > threshold, :]
    return outliers

import laspy
import numpy as np
from extract import geometry
import json

FEATURE_X = "feature_x"
FEATURE_Y = "feature_y"
FEATURE_Z = "feature_z"
import matplotlib.pyplot as plt


def find_cluster_centers(points, max_angle_range, num_of_clusters):
    angle_min = np.min(points)
    angle_max = np.max(points)
    angle_range = np.arange(angle_min, angle_max)

    # find center that contains most points in range
    cluster_centers = []
    num_points = np.zeros((2, len(angle_range)))
    for i, angle in enumerate(angle_range):
        num_points[0][i] = np.sum(np.logical_and(points >= angle - max_angle_range, points <= angle + max_angle_range))
        num_points[1][i] = angle

    # find multiple clusters
    for num_of_clusters in range(num_of_clusters):
        most_common_range_index = np.argmax(num_points[0])
        most_common_angle = int(num_points[1][most_common_range_index])
        cluster_centers.append(most_common_angle)

        # remove the current cluster from the angle range
        for i, angle in enumerate(angle_range):
            if i in range(most_common_range_index - max_angle_range, most_common_range_index + max_angle_range):
                num_points[0][i] = 0

    print(cluster_centers)
    return cluster_centers



def seperate_by_scan_angle(object_file_path, max_angle_range):
    laspy_file = laspy.read(object_file_path)
    points = laspy_file.scan_angle
    points_norm = np.asarray([p*180/30000 for p in points])
    # Find the range of scan angles to consider
    angle_min = np.min(points_norm)
    angle_max = np.max(points_norm)
    angle_range = np.arange(angle_min, angle_max)


    # Find the number of points within each scan angle range
    num_points = np.zeros(len(angle_range))
    for i, angle in enumerate(angle_range):
        num_points[i] = np.sum(np.logical_and(points_norm >= angle - max_angle_range, points_norm <= angle + max_angle_range))
    # find the 3 most dominent clusters
    cluster_centers = find_cluster_centers(points_norm, max_angle_range,3)
    # Calculate the center of the most common range
    i = 0
    for angle in cluster_centers:
        most_common_angle = angle

        new_points = laspy_file.points[
            (points_norm >= most_common_angle - max_angle_range) & (points_norm <= most_common_angle + max_angle_range)]
        new_laspy_file = laspy.create(point_format=laspy_file.header.point_format,
                                      file_version=laspy_file.header.version)
        new_laspy_file.points = new_points
        new_laspy_file.filename = "points_within_range"+str(i)+".las"
        new_laspy_file.write(new_laspy_file.filename)
        i += 1
    print(new_laspy_file)


def show_by_scan_angle(object_file_path, output_file_path):
    laspy_file = laspy.read(object_file_path)
    points = laspy_file.scan_angle
    points_norm = [p*180/30000 for p in points]
    plt.hist(points_norm, bins=200)
    plt.show()


def main():
    filepath = r"..\objects\box\CC-9.las"

    print(seperate_by_scan_angle(filepath, 5))
    #extract_features_from_object(filepath, "box.json")
