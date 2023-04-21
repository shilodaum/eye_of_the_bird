import numpy as np
from sklearn import linear_model
import laspy
from extract import geometry
from matplotlib import pyplot as plt


def find_plane(points, n_iteration=200, threshold=0.05):
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


def main():
    laspy_file = laspy.read(
        r"C:\Users\TLP-241\Desktop\talpiot\magdad\code\extract\subtracted_9_10\filtered.las_COMPONENT_1.las"
    )
    points = laspy_file.xyz
    points = geometry.get_center(points)

    # remove outliers
    points = geometry.remove_outliers(points)

    # get 10% of the points randomly
    points = points[np.random.choice(points.shape[0], int(points.shape[0] * 0.1), replace=False), :]

    points = geometry.remove_outliers(points)

    plane = find_plane(points)

    # plot the points and the plane in 3d space
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # set aspect ratio to 1:1:1
    ax.set_aspect('equal', 'box')
    # plot a plane
    x = np.linspace(-1, 1, 10)
    y = np.linspace(-1, 1, 10)
    X, Y = np.meshgrid(x, y)
    Z = (-plane[0] * X - plane[1] * Y - plane[3]) * 1. / plane[2]
    ax.plot_surface(X, Y, Z, alpha=0.2)
    # plot the points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], s=0.3)


    plt.show()









if __name__ == '__main__':
    main()