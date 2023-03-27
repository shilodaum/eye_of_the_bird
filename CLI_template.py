import pyCloudCompare as cc

cli = cc.CloudCompareCLI()
cmd = cli.new_command()
cmd.silent()  # Disable console
cmd.open("pointcloud.ply")  # Read file
cmd.cloud_export_format(cc.CLOUD_EXPORT_FORMAT.ASCII, extension="xyz")
cmd.save_clouds("newPointcloud.xyz")
print(cmd)
cmd.execute()