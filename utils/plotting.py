from matplotlib import pyplot as plt


def plot_cloud_2d(points):
    """plots the points on x-y plane with z plane as colorbar and ratio 1:1."""
    plt.scatter(points[:, 0], points[:, 1], c=points[:, 2], s=0.3)
    plt.colorbar()
    plt.axis("equal")
    plt.show()
