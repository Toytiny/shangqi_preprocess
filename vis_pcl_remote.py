from argparse import RawDescriptionHelpFormatter
from matplotlib import pyplot as plt
import numpy as np
from glob import glob
from scipy.spatial.transform import Rotation as R
import os
import cv2
import open3d as o3d
import pandas as pd
 
SIDE_RANGE = (-20, 20)
FWD_RANGE = (-20, 80)
HEIGHT_RANGE = (0,3)
RES = 0.15625/2

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

def filt_points_in_range(x,y,z):

    f_filt = np.logical_and((x > FWD_RANGE[0]), (x < FWD_RANGE[1]))
    s_filt = np.logical_and((y > SIDE_RANGE[0]), (y < SIDE_RANGE[1]))
    h_filt = np.logical_and((z > HEIGHT_RANGE[0]), (z < HEIGHT_RANGE[1]))
    filt = np.logical_and(np.logical_and(f_filt, s_filt), h_filt)
    indices = np.argwhere(filt).flatten()

    return indices

def show_bev_lidar(pcs,im,color):

    x_img = np.floor((pcs[:,0]-FWD_RANGE[0]) / RES).astype(np.int32)
    y_img = np.floor(-(pcs[:,1] + SIDE_RANGE[0])/RES).astype(np.int32)
    im[y_img,x_img] = color
    return im

def show_bev_radar(pcs,im,color):

    x_img = np.floor((pcs[:,0]-FWD_RANGE[0]) / RES).astype(np.int32)
    y_img = np.floor(-(pcs[:,1] + SIDE_RANGE[0])/RES).astype(np.int32)
    N = x_img.shape[0]
    for j in range(N):
        im=cv2.circle(im,(x_img[j],y_img[j]),2, color=color, thickness=-1)
    return im

def show_bev_box(box, im, color):

    x_points = box[:,0]
    y_points = box[:,1]
    x_img = np.floor((x_points - FWD_RANGE[0]) / RES).astype(np.int32)
    y_img = np.floor(-(y_points + SIDE_RANGE[0])/RES).astype(np.int32)
    lines = [[0, 1], [1, 2], [2, 3], [3, 0]]
    for line in lines:
        im = cv2.line(im, (x_img[line[0]], y_img[line[0]]), (x_img[line[1]], y_img[line[1]]), color, 1)
    return im

def vis_with_opencv(lidar_pc, radar_pc, box_list, fname):

    x_max = int((FWD_RANGE[1] - FWD_RANGE[0]) / RES)
    y_max = int((SIDE_RANGE[1] - SIDE_RANGE[0]) / RES)
    im = np.zeros([y_max, x_max,3], dtype=np.uint8) + 255
    im = show_bev_lidar(lidar_pc,im,(255,0,0))
    im = show_bev_radar(radar_pc,im,(0,0,255))
    for box in box_list:
        show_bev_box(box,im,(0,255,0))
    cv2.imwrite(fname,im)

def vis_with_plt(lidar_pc,radar_pc,fname):

    fig=plt.figure()
    ax1 = plt.axes(projection='3d')
    ax1.scatter3D(lidar_pc[:,0],lidar_pc[:,1],lidar_pc[:,2], s=0.1, marker='.')
    ax1.scatter3D(radar_pc[:,0],radar_pc[:,1],radar_pc[:,2], s=0.1, marker='.')
    ax1.grid(False)
    plt.savefig(fname, dpi=600)

def get_rotation(arr):
    x,y,_ = arr
    yaw = np.arctan(y/x)
    angle = np.array([0, 0, yaw])
    r = R.from_euler('XYZ', angle)
    return r.as_matrix()

def get_bbx_param(obj_info, lidar_tr):
    
    center = obj_info[2:5] #+ np.array([-2.5, 0, 0])
    extent = obj_info[5:8] 
    angle = obj_info[8:11]
    # angle[0] = -angle[0]
    rot_m = get_rotation(angle)
    # rot_m = np.eye(3)
    obbx = o3d.geometry.OrientedBoundingBox(center.T, rot_m, extent.T)
    #obbx.rotate(lidar_tr[:3,:3])
    #obbx.translate(lidar_tr[:3,3])
    return obbx

def get_bbx_corner(obbx):
    l = obbx.extent[0]
    w = obbx.extent[1]
    h = obbx.extent[2]
    rot_m = obbx.R
    center = obbx.center
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    z_corners = [0, 0, 0, 0, h, h, h, h]
    corners_3d = np.vstack([x_corners, y_corners, z_corners])
    new_corners_3d = np.dot(rot_m, corners_3d).T + center
    return new_corners_3d


lidar_ext = [-2.50, 0, 2.03, 4.9, -1.5, 0 ] # maybe zero
radar_ext = [0.06, -0.2, 0.7, 0, 3.19, 180] # pitch 3.19

root_path = "/mnt/12T/fangqiang/inhouse/20220222-10-32-36/"
lidar_path = "/mnt/12T/fangqiang/20220222-10-32-36/input/lidar/20220222-10-32-36_C/"

radar_files = sorted(glob(root_path + "sync_radar/*.csv"))
lidar_files = sorted(glob(lidar_path + "*.pcd"))
gt_files = sorted(glob(root_path + "sync_gt/*.csv"))

img_path = root_path + "vis_lidar"
save_img = True
if save_img:
    if not os.path.exists(img_path):
        os.mkdir(img_path)
lidar_tr = get_matrix_from_ext(lidar_ext)
radar_tr = get_matrix_from_ext(radar_ext)

for idx in range(0, len(lidar_files)):#len(lidar_files)):
    lidar_pcd = o3d.io.read_point_cloud(lidar_files[idx])
    lidar_pcd = o3d.geometry.PointCloud.uniform_down_sample(lidar_pcd, 10)
    radar_pcd = csv2geometry(radar_files[idx])
    lidar_pcd.transform(lidar_tr)
    radar_pcd.transform(radar_tr)
    radar_pc = np.asarray(radar_pcd.points)
    lidar_pc = np.asarray(lidar_pcd.points)

    indices = filt_points_in_range(lidar_pc[:,0],lidar_pc[:,1],lidar_pc[:,2])
    lidar_pc = lidar_pc[indices]
    indices = filt_points_in_range(radar_pc[:,0],radar_pc[:,1],radar_pc[:,2])
    radar_pc = radar_pc[indices]

    gt_data = np.loadtxt(gt_files[idx+1])
    box_list = []
    for obj_info in gt_data:
         obj_bbx = get_bbx_param(obj_info, lidar_tr)
         corners = get_bbx_corner(obj_bbx)
         box_list.append(corners)
         

    fname = os.path.join(img_path, str(idx).zfill(9) + '.png')
    vis_with_opencv(lidar_pc,radar_pc,box_list,fname)
    #vis_with_plt(lidar_pc,radar_pc,fname)
    print('save image-{}'.format(fname))

