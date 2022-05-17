from argparse import RawDescriptionHelpFormatter
import open3d as o3d
import numpy as np
from glob import glob
from scipy.spatial.transform import Rotation as R
import os
import pandas as pd



# lidar_ext = [-2.502, -0.004, 2.033, 3.5, -0.2, 0 ]
# for 20220118
# lidar_ext = [0, 0, 0, -2.5, 0.2, 0]
# radar_ext = [0.06, -0.2, 0.7, -2, 0, 180]

# lidar_ext = [0, 0, -0.3, -2.5, 0, 0] #-0.2
# radar_ext = [0.06, -0.2, 0.7, -3.5, 2, 180]

# for 20220126
lidar_ext = [0, 0, 0, -1, 2, 0]
radar_ext = [0.06, -0.2, 0.2, -1, 2, 180]

def csv2geometry(fname):
    
    pts = pd.read_table(fname,sep=",", header=None).values[1:,1:4]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd

def get_matrix_from_ext(ext):
    rot = R.from_euler('ZYX', ext[3:], degrees=True)
    rot_m = rot.as_matrix()
    x, y, z = ext[:3]
    tr = np.eye(4)
    tr[:3,:3] = rot_m
    tr[:3, 3] = np.array([x, y, z]).T
    return tr

def get_rotation(arr):
    x,y,_ = arr[:3]
    yaw = np.arctan(y/x)
    angle = np.array([0, 0, yaw])
    r = R.from_euler('XYZ', angle)
    return r.as_matrix()

def get_bbx_param(obj_info):
    center = obj_info[2:5] #+ np.array([-2.5, 0, 0])
    
    extent = obj_info[5:8]
    
    
    angle = obj_info[8:]
    # angle[0] = -angle[0]
    rot_m = get_rotation(angle)
    # rot_m = np.eye(3)
    
    obbx = o3d.geometry.OrientedBoundingBox(center.T, rot_m, extent.T)
    return obbx

root_path = "../20220126-14-52-23/"

lidar_path = "/media/toytiny/Data2/20220126-14-52-23/lidar/20220126-14-52-23_C/"

lidar_files = sorted(glob(lidar_path + "*.pcd"))

gt_files = sorted(glob(root_path + "sync_gt/*.csv"))

radar_files = sorted(glob(root_path + 'sync_radar/*.csv'))

img_path = "/home/dj/Downloads/lidar/shangqi/inhouse_format/img_vis"

save_img = False
if save_img:
    if not os.path.exists(img_path):
        os.mkdir(img_path)
    else:
        os.system("rm -r " + img_path)
        os.mkdir(img_path)


start = 4500
gt_fname = gt_files[0]
lidar_tr = get_matrix_from_ext(lidar_ext)
radar_tr = get_matrix_from_ext(radar_ext)
# lidar_pcd = o3d.io.read_point_cloud(lidar_files[3])
# lidar_pcd.transform(lidar_tr)
# lidar_pcd.paint_uniform_color([1, 0, 0])

radar_pcd = o3d.geometry.PointCloud()
radar_temp_pcd = csv2geometry(radar_files[0])
radar_pcd.points = radar_temp_pcd.points
radar_pcd.paint_uniform_color([0, 0, 1])
radar_pcd.transform(radar_tr)


vis = o3d.visualization.Visualizer()
vis.create_window(width=1920,height=1080)
vis.add_geometry(lidar_pcd)
vis.add_geometry(radar_pcd)

# print(gt_fname)
gt_data = np.loadtxt(gt_fname)

box_list = []
if gt_data.size != 0:
    if len(gt_data.shape) == 1:
        gt_data = gt_data.reshape(1, -1)
    for obj_info in gt_data:
        obj_bbx = get_bbx_param(obj_info)
        box_list += [obj_bbx]
        vis.add_geometry(obj_bbx)


for idx in range(start, len(lidar_files)):
    # temp_pcd = o3d.io.read_point_cloud(lidar_files[idx])
    # lidar_pcd.points = temp_pcd.points
    # lidar_pcd.paint_uniform_color([0, 0, 1])
    # lidar_pcd.transform(lidar_tr)

    temp_pcd = csv2geometry(radar_files[idx])
    radar_pcd.points = temp_pcd.points
    radar_pcd.paint_uniform_color([1, 0, 0])
    radar_pcd.transform(radar_tr)

    # vis.update_geometry(lidar_pcd)
    vis.update_geometry(radar_pcd)

    for box in box_list:
        vis.remove_geometry(box, reset_bounding_box=False)
    box_list = []
    gt_data = np.loadtxt(gt_files[idx])
    if gt_data.size != 0:
        if len(gt_data.shape) == 1:
            gt_data = gt_data.reshape(1, -1)
        for obj_info in gt_data:
            obj_bbx = get_bbx_param(obj_info)
            box_list += [obj_bbx]
            vis.add_geometry(obj_bbx, reset_bounding_box=False)
    
    vis.poll_events()
    vis.update_renderer()
    # fname = os.path.join(img_path, str(idx).zfill(9) + '.png')
    # vis.capture_screen_image(fname)