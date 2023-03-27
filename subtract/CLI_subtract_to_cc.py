from subtract import pyCloudCompare as pycc
from typing import List
import os


def subtract_clouds(cloud1_path: str, cloud2_path: str, output_path: str,
                    bounds: List[int], global_shift: List[int]) -> None:
    """
    Subtract 2 clouds, filter the result, and extract connected components from the result.
    :param cloud1_path: path to the first cloud
    :param cloud2_path: path to the second cloud
    :param output_path: path to the output folder
    :param bounds: bounds of the area to crop the clouds to
    :param global_shift: shift to apply to the clouds before cropping
    :return: None. The result is saved in the output folder.
    """
    # Initialize
    cli = pycc.CloudCompareCLI()
    cmd = cli.new_command()
    cmd.silent()  # Disable console
    cmd.log_file(output_path + r"\log.txt")
    cmd.auto_save(False)
    cmd.cloud_export_format("LAS")
    shifted_bounds = [bounds[0] + global_shift[0], bounds[1] + global_shift[1], bounds[2] + global_shift[2],
                      bounds[3] + global_shift[0], bounds[4] + global_shift[1], bounds[5] + global_shift[2]]
    # Open 2 clouds
    cmd.open(cloud1_path, global_shift=global_shift)
    cmd.open(cloud2_path, global_shift=global_shift)

    cmd.crop(*shifted_bounds)  # Crop the 2 clouds to the same area
    cmd.c2c_dist()  # Subtract the 2 clouds
    cmd.save_clouds(output_path + r"\subtracted.las", output_path + r"\reference.las")  # Save the subtracted cloud
    cmd.clear_clouds()  # Close all opened clouds

    cmd.open(output_path + r"\subtracted.las", global_shift=global_shift)  # Open only the subtracted cloud
    cmd.filter_sf(0.1, 10)  # Filter the cloud by its calculated distance from its neighbors in the other scan
    cmd.save_clouds(output_path + r"\filtered.las")  # Save filtered cloud
    cmd.auto_save(True)
    cmd.extract_cc(8, 100)  # Extract Connected Components from the filtered cloud, and save them
    cmd.execute()


def merge_clouds(clouds_path: str, output_path: str, merged_cloud_name: str, global_shift: List[int]) -> None:
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
    cmd.log_file(output_path + r"\log.txt")
    cmd.auto_save(False)
    cmd.cloud_export_format("LAS")

    # Open all clouds
    i = 0
    for cloud in os.listdir(clouds_path):
        if cloud.endswith(".las"):
            cmd.open(clouds_path + "\\" + cloud, global_shift=global_shift)
        i += 1
        print("Opened " + str(i) + " clouds")
    # Merge clouds
    cmd.merge_clouds()
    # Save merged cloud
    cmd.save_clouds(output_path + "\\" + merged_cloud_name + ".las")
    print(cmd)
    cmd.execute()
