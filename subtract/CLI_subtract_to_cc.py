from subtract import pyCloudCompare as pycc
from typing import List
import os
import laspy
import glob
from os.path import join as pjoin

SHIFT = [-692300, -3616100, 0]
BOX = [692297, 3616248, 300, 692335, 3616288, 400]


def get_intersection_bounds(cloud1_path: str, cloud2_path: str) -> List[float]:
    """
    Get the intersection bounds of 2 clouds.
    :param cloud1_path: path to the first cloud
    :param cloud2_path: path to the second cloud
    :return: the intersection bounds of the 2 clouds
    """
    # Load the first LAS file
    las1 = laspy.read(cloud1_path)

    # Load the second LAS file
    las2 = laspy.read(cloud2_path)

    # Get the bounds of the first cloud
    min_x_1 = las1.header.x_min
    min_y_1 = las1.header.y_min
    min_z_1 = las1.header.z_min
    max_x_1 = las1.header.x_max
    max_y_1 = las1.header.y_max
    max_z_1 = las1.header.z_max

    # Get the bounds of the second cloud
    min_x_2 = las2.header.x_min
    min_y_2 = las2.header.y_min
    min_z_2 = las2.header.z_min
    max_x_2 = las2.header.x_max
    max_y_2 = las2.header.y_max
    max_z_2 = las2.header.z_max

    # Get the intersection bounds
    min_x = max(min_x_1, min_x_2)
    min_y = max(min_y_1, min_y_2)
    min_z = max(min_z_1, min_z_2)
    max_x = min(max_x_1, max_x_2)
    max_y = min(max_y_1, max_y_2)
    max_z = min(max_z_1, max_z_2)

    return [min_x, min_y, min_z, max_x, max_y, max_z]


def subtract_clouds(input_path: str = r"..\data\scans", output_path: str = r"..\data\objects",
                    min_points_per_cloud: int = 500, global_shift: List[int] = SHIFT,
                    bounds: List[int] = None) -> None:
    """
    Subtract 2 clouds, filter the result, and extract connected components from the result.
    :param input_path: path to the folder containing the 2 clouds to subtract
    :param output_path: path to the folder to save the result in
    :param min_points_per_cloud: minimum number of points in a connected component to be saved
    :param global_shift: shift to apply to the clouds before subtracting
    :param bounds: bounds of the area to crop the clouds to. If None, the intersection bounds of the 2 clouds are used.
    :return: None. The result is saved in the output folder.
    """

    # Get the paths to the 2 clouds
    clouds = glob.glob(pjoin(input_path, '*.las'))
    cloud1_path = clouds[0]
    cloud2_path = clouds[1]

    # Initialize
    cli = pycc.CloudCompareCLI()
    cmd = cli.new_command()
    cmd.silent()  # Disable console
    cmd.log_file(output_path + r"\log.txt")
    cmd.auto_save(False)
    cmd.cloud_export_format("LAS")

    if bounds is None:
        bounds = get_intersection_bounds(cloud1_path, cloud2_path)

    shifted_bounds = [bounds[0] + global_shift[0], bounds[1] + global_shift[1], bounds[2] + global_shift[2],
                      bounds[3] + global_shift[0], bounds[4] + global_shift[1], bounds[5] + global_shift[2]]
    # Open 2 clouds
    cmd.open(cloud1_path, global_shift=global_shift)
    cmd.open(cloud2_path, global_shift=global_shift)

    cmd.crop(*shifted_bounds)  # Crop the 2 clouds to the same area
    cmd.c2c_dist()  # Subtract the 2 clouds
    cmd.save_clouds(pjoin(output_path, r"\subtracted.las"),
                    pjoin(output_path, r"\reference.las"))  # Save the subtracted cloud
    cmd.clear_clouds()  # Close all opened clouds

    cmd.open(pjoin(output_path, r"\subtracted.las"), global_shift=global_shift)  # Open only the subtracted cloud
    cmd.filter_sf(0.1, 10)  # Filter the cloud by its calculated distance from its neighbors in the other scan
    cmd.save_clouds(pjoin(output_path, r"\filtered.las"))  # Save filtered cloud
    cmd.auto_save(True)
    cmd.extract_cc(8, min_points_per_cloud)  # Extract Connected Components from the filtered cloud, and save them
    cmd.execute()

    # Delete temporary files
    os.remove(pjoin(output_path, r"\subtracted.las"))
    os.remove(pjoin(output_path, r"\reference.las"))
    os.remove(pjoin(output_path, r"\filtered.las"))


def merge_clouds(clouds_path: str, output_path: str, merged_cloud_name: str,
                 global_shift: List[int] = SHIFT) -> None:
    """
    Merge multiple clouds into one.
    :param clouds_path: path to the folder containing the clouds to merge
    :param output_path: path to the output folder
    :param merged_cloud_name: name of the merged cloud
    :param global_shift: shift to apply to the clouds before merging
    :return: None. The result is saved in the output folder.
    """
    # Initialize
    cli = pycc.CloudCompareCLI()
    cmd = cli.new_command()
    cmd.silent()  # Disable console
    cmd.log_file(pjoin(output_path, r"\log.txt"))
    cmd.auto_save(False)
    cmd.cloud_export_format("LAS")

    # Open all clouds
    i = 0
    for cloud in os.listdir(clouds_path):
        if cloud.endswith(".las"):
            cmd.open(pjoin(clouds_path, cloud), global_shift=global_shift)
        i += 1
        print("Opened " + str(i) + " clouds")

    # Merge clouds
    cmd.merge_clouds()
    # Save merged cloud
    cmd.save_clouds(pjoin(clouds_path, merged_cloud_name + ".las"))

    cmd.execute()
