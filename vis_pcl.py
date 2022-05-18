from argparse import RawDescriptionHelpFormatter
import open3d as o3d
import numpy as np
from glob import glob
from scipy.spatial.transform import Rotation as R
import os
import pandas as pd


lidar_ext = [0, 0, -0.3, -2.5, 0, 0] #-0.2
radar_ext = [0.06, -0.2, 0.7, -3.5, 2, 180]

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
    x,y,_ = arr
    yaw = np.arctan(y/x)
    angle = np.array([0, 0, yaw])
    r = R.from_euler('XYZ', angle)
    return r.as_matrix()

def get_bbx_param(obj_info):
    
    center = obj_info[2:5] #+ np.array([-2.5, 0, 0])
    
    extent = obj_info[5:8] 
    
    
    angle = obj_info[8:-1]
    # angle[0] = -angle[0]
    rot_m = get_rotation(angle)
    # rot_m = np.eye(3)
    
    obbx = o3d.geometry.OrientedBoundingBox(center.T, rot_m, extent.T)
    return obbx

root_path = "/mnt/12T/fangqiang/inhouse/20220126-14-52-23/"
lidar_path = "/mnt/12T/public/inhouse/20220126-14-52-23/input/lidar/20220126-14-52-23_C/"

lidar_files = sorted(glob(lidar_path + "*.pcd"))

gt_files = sorted(glob(root_path + "sync_gt/*.csv"))

radar_files = sorted(glob(root_path + "sync_radar/*.csv"))

img_path = root_path + "vis_lidar"

save_img = True
if save_img:
    if not os.path.exists(img_path):
        os.mkdir(img_path)
    

read_view= False
start = 7150 #2200 #500 #500 #2200 #5300 # 7150
gt_fname = gt_files[3]
lidar_tr = get_matrix_from_ext(lidar_ext)
radar_tr = get_matrix_from_ext(radar_ext)
lidar_pcd = o3d.io.read_point_cloud(lidar_files[3])
lidar_pcd.transform(lidar_tr)
lidar_pcd.paint_uniform_color([1, 0, 0])

# radar_pcd = o3d.geometry.PointCloud()
# radar_temp_pcd = csv2geometry(radar_files[3])
# radar_pcd.points = radar_temp_pcd.points
# radar_pcd.paint_uniform_color([0, 0, 1])
# radar_pcd.transform(radar_tr)

radar_temp_pcd = csv2geometry(radar_files[3])
radar_pcd = o3d.geometry.PointCloud()
radar_pcd.points = radar_temp_pcd.points
radar_pcd.transform(radar_tr)
radar_temp = np.asarray(radar_pcd.points)

radar_ls = []

for i in range(np.size(radar_temp,0)):
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
    mesh_sphere.compute_vertex_normals()
    mesh_sphere.paint_uniform_color([255/255, 0/255, 191/255])
    mesh_sphere.translate(radar_temp[i,:])
    radar_ls.append(mesh_sphere)
    
vis = o3d.visualization.Visualizer()
vis.create_window(width=1280,height=720)
vis.add_geometry(lidar_pcd)
for radar in radar_ls:
    vis.add_geometry(radar)
    
ctr = vis.get_view_control()
if read_view:
    param = o3d.io.read_pinhole_camera_parameters('origin.json')
    ctr.convert_from_pinhole_camera_parameters(param)
# print(gt_fname)
# gt_data = np.loadtxt(gt_fname)
# box_list = []
# for obj_info in gt_data:
#     obj_bbx = get_bbx_param(obj_info)
#     box_list += [obj_bbx]
#     vis.add_geometry(obj_bbx)

for idx in range(start, len(lidar_files)):
    temp_pcd = o3d.io.read_point_cloud(lidar_files[idx])
    lidar_pcd.points = temp_pcd.points
    lidar_pcd.paint_uniform_color([0, 191/255, 1])
    lidar_pcd.transform(lidar_tr)

    for radar in radar_ls:
        vis.remove_geometry(radar)
    # temp_pcd = csv2geometry(radar_files[idx])
    # radar_pcd.points = temp_pcd.points
    # radar_pcd.paint_uniform_color([255/255, 0/255, 191/255])
    # radar_pcd.transform(radar_tr)
    radar_temp_pcd = csv2geometry(radar_files[idx])
    radar_pcd = o3d.geometry.PointCloud()
    radar_pcd.points = radar_temp_pcd.points
    radar_pcd.transform(radar_tr)
    radar_temp = np.asarray(radar_pcd.points)
    
    radar_ls = []
    
    for i in range(np.size(radar_temp,0)):
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        mesh_sphere.compute_vertex_normals()
        mesh_sphere.paint_uniform_color([255/255, 0/255, 191/255])
        mesh_sphere.translate(radar_temp[i,:])
        radar_ls.append(mesh_sphere)
        
    vis.update_geometry(lidar_pcd)
    #vis.update_geometry(radar_pcd)
    for radar in radar_ls:
        vis.add_geometry(radar)
    # for box in box_list:
    #     vis.remove_geometry(box, reset_bounding_box=False)
    # box_list = []
    # gt_data = np.loadtxt(gt_files[idx])
    # for obj_info in gt_data:
    #     obj_bbx = get_bbx_param(obj_info)
    #     box_list += [obj_bbx]
    #     vis.add_geometry(obj_bbx, reset_bounding_box=False)
    ctr = vis.get_view_control()
    if read_view:
        param = o3d.io.read_pinhole_camera_parameters('origin.json')
        ctr.convert_from_pinhole_camera_parameters(param)
    vis.poll_events()
    vis.update_renderer()
    if save_img:
        fname = os.path.join(img_path, str(idx).zfill(9) + '.png')
        vis.capture_screen_image(fname)


