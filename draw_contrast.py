
from argparse import RawDescriptionHelpFormatter
import open3d as o3d
import numpy as np
from glob import glob
from scipy.spatial.transform import Rotation as R
import os
import pandas as pd


# lidar_ext = [-2.502, -0.004, 2.033, 3.5, -0.2, 0 ]
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
    # obbx = o3d.geometry.TriangleMesh.create_box(extent[0],extent[1],extent[2])
    # obbx.rotate(rot_m)
    # obbx.translate(center-extent/2)
    
    obbx = o3d.geometry.OrientedBoundingBox(center.T, rot_m, extent.T)
    obbx = o3d.geometry.LineSet.create_from_oriented_bounding_box(obbx)
    
    return obbx

def c2p(pcs):
    
    x = pcs[:,0]
    y = pcs[:,1]
    z = pcs[:,2]
    
    r = np.sqrt(x**2+y**2+z**2)
    theta = np.arcsin(z/r)
    phi = np.arctan2(y,x)
    
    return r, theta, phi

root_path = "../20220118-13-43-20/"
lidar_path = "/media/toytiny/Data/20220118-13-43-20/lidar/20220118-13-43-20_C/"

lidar_files = sorted(glob(lidar_path + "*.pcd"))

gt_files = sorted(glob(root_path + "sync_gt/*.csv"))

radar_files = sorted(glob(root_path + "sync_radar/*.csv"))

img_path = root_path + "img_vis_contrast/"

save_img = True
if save_img:
    if not os.path.exists(img_path):
        os.mkdir(img_path)
   

lidar_tr = get_matrix_from_ext(lidar_ext)
radar_tr = get_matrix_from_ext(radar_ext)

read_view= False
save_view= False

st_idx = 5485 #5485 #6765  #7187 #2228
for idx in range(st_idx,len(radar_files)):
    
    gt_fname = gt_files[idx]
    # lidar_pcd = o3d.io.read_point_cloud(lidar_files[idx])
    # lidar_pcd.transform(lidar_tr)
    #temp_pcs = np.asarray(lidar_pcd.points)
    #r, theta, phi = c2p(temp_pcs)
    
    #temp_idx = np.logical_and(r<75, np.logical_and(np.logical_and(theta>-(10*np.pi/180),theta<(10*np.pi/180)),temp_pcs[:,0]>0))
    
    #front_pcs = temp_pcs[temp_idx,:]
    #lidar_pcd.points = o3d.utility.Vector3dVector(front_pcs)
    #lidar_pcd.paint_uniform_color([0, 191/255, 1])
    
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
        
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1920,height=1080)   
    #vis.get_render_option().load_from_json('option.json')
    
    vis.add_geometry(lidar_pcd)
    for radar in radar_ls:
        vis.add_geometry(radar)
    
    ctr = vis.get_view_control()
    if read_view:
        param = o3d.io.read_pinhole_camera_parameters('origin.json')
        ctr.convert_from_pinhole_camera_parameters(param)
        
    print(gt_fname)
    gt_data = np.loadtxt(gt_fname)
    box_list = []
    for obj_info in gt_data:
        if obj_info[2]>0 and np.linalg.norm(obj_info[2:4])<75 and obj_info[5]>2:
            obj_bbx = get_bbx_param(obj_info)
            box_list += [obj_bbx]
            #vis.get_render_option().line_width = 100
        
            vis.add_geometry(obj_bbx)
            vis.add_geometry(obj_bbx)
    
    
    vis.run()
    #vis.poll_events()
    #vis.update_renderer()
    #if save_img:
    fname = os.path.join(img_path, str(idx).zfill(9) + '.png')
    vis.capture_screen_image(fname)
    if save_view:
        param = ctr.convert_to_pinhole_camera_parameters()
        o3d.io.write_pinhole_camera_parameters('origin.json', param)
        
    vis.destroy_window()