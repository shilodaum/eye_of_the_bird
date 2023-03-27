import numpy as np
from scipy.spatial import KDTree


def find_min_bounding_rectangle(points, num_angles=180, quartile=0.01):
    """
    finds the minimum bounding rectangle of the points
    :param quartile: the allowed number of outliers
    :param num_angles: the number of angles to consider
    :param points: the point cloud
    :return: width, height, angle
    """
    # iterate over angles, and find angle such that the width is minimal
    min_width = np.inf
    min_angle = 0
    for angle in np.arange(0, np.pi / 2, np.pi / num_angles):
        # rotate the points
        rotated_points = rotate(points, angle)
        # find the width and height
        min_x = np.quantile(rotated_points[:, 0], quartile)
        max_x = np.quantile(rotated_points[:, 0], 1 - quartile)
        width = max_x - min_x
        # if the width is smaller than the current minimum, update the minimum
        if width < min_width:
            min_width = width
            min_angle = angle
    # rotate the points back
    points = rotate(points, min_angle)
    # find the width and height
    width = get_x_diff(points)
    height = get_y_diff(points)
    return width, height, min_angle


def get_center(points):
    """Return the center of the points"""
    mean = np.array([points[:, 0].mean(), points[:, 1].mean(), points[:, 2].mean()]).reshape(1, 3)
    return points - mean


def get_z_diff(points):
    """Return the height of the points"""
    return points[:, 2].max() - points[:, 2].min()


def get_x_diff(points):
    """Return the x difference of the points"""
    return points[:, 0].max() - points[:, 0].min()


def get_y_diff(points):
    """Return the y difference of the points"""
    return points[:, 1].max() - points[:, 1].min()


def remove_outliers(points, threshold=1, k=6):
    """
    Remove outliers from the points
    :param points: the points
    :param threshold: the threshold for the distance
    :param k: the number of neighbors to consider
    :return: the points without outliers
    """
    # find the k nearest neighbors of each point
    tree = KDTree(points)
    distances, indices = tree.query(points, k=k)
    # find the mean distance to the k nearest neighbors
    mean_distances = np.mean(distances, axis=1)
    std_distance = np.std(mean_distances)
    average_distance = np.mean(mean_distances)
    # find the points that are within the threshold
    points = points[mean_distances < average_distance + threshold * std_distance]
    return points


def rotate(points, angle, center_of_rotation=None):
    """
    Rotate the points around the center of rotation around the z axis
    :param points: the points
    :param angle: the angle in radians
    :param center_of_rotation: the point around which to rotate. if set to None, (0,0,0) is used
    :return: the rotated points
    """
    if center_of_rotation is None:
        center_of_rotation = np.array([0, 0, 0])
    # translate the points
    points = points - center_of_rotation
    # rotate the points
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                                [np.sin(angle), np.cos(angle), 0],
                                [0, 0, 1]])
    # apply the rotation matrix to all points
    points = np.dot(points, rotation_matrix)
    # translate the points back
    points = points + center_of_rotation
    return points


def get_pca_main_axis(points):
    """finds the orientation vector of the cloud of points"""
    # find the covariance matrix between x and y
    covariance_matrix = np.cov(points[:, 0], points[:, 1])
    # find the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    # find the eigenvector with the maximum eigenvalue
    pca_main_component = eigenvectors[:, np.argmax(eigenvalues)]
    return pca_main_component


def get_angle_pca(points):
    """finds the rotation of the cloud of points"""
    pca_main_component = get_pca_main_axis(points)
    angle = np.arctan2(pca_main_component[1], pca_main_component[0])
    return angle
