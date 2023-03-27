import laspy
import numpy as np


def plot_cloud_points_2d(points):
    pass

def center(points):
    """Return the center of the points"""
    mean = np.array([points[:, 0].mean(), points[:, 1].mean(), points[:, 2].mean()]).reshape(1, 3)
    return points - mean


def get_height(points):
    """Return the height of the points"""
    return points[:, 2].max() - points[:, 2].min()


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


def find_min_bounding_box(points):
    """
    find the minimal rectangle that contains all points
    :param points: the points
    :return: width, height, angle
    """
    # get the 2d covariance matrix between x and y
    cov = np.cov(points[:, 0], points[:, 1])
    # use the convariance matrix to find the angle
    angle = np.arctan2(cov[0, 1], cov[0, 0])
    # rotate the points
    rotated_points = rotate(points, angle)
    # find the width and height
    width = rotated_points[:, 0].max() - rotated_points[:, 0].min()
    height = rotated_points[:, 1].max() - rotated_points[:, 1].min()
    return width, height, angle


def main():
    filepath = r"..\..\objects\9 - car.las"
    # load the file using laspy
    laspy_file = laspy.read(filepath)
    # get the points
    points = laspy_file.xyz
    points = center(points)
    print(find_min_bounding_box(points))


if __name__ == '__main__':
    main()
