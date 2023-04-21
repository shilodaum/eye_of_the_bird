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

