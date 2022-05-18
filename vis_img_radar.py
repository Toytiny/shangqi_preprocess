import open3d as o3d
import numpy as np
from glob import glob
from scipy.spatial.transform import Rotation as R
import os
import pandas as pd
import cv2
from tqdm import tqdm

def get_matrix_from_ext(ext):
    rot = R.from_euler('ZYX', ext[3:], degrees=True)
    rot_m = rot.as_matrix()
    x, y, z = ext[:3]
    tr = np.eye(4)
    tr[:3,:3] = rot_m
    tr[:3, 3] = np.array([x, y, z]).T
    return tr

def get_radar_target(data):

    ## use the original right-hand coordinate systen, front is x, left is y, up is z
    x_points = data[0, :]
    y_points = data[1, :]
    z_points = data[2, :]
    vel_r = data[7, :]
    rcs = data[6,:]
    power = data[5,:]
    x = -y_points
    y = -z_points
    z = x_points
    points = np.vstack((x, y, z)).T
    N = points.shape[0]


    targets = {
        "coord" : points,
        "vel_r": vel_r,
        "rcs": rcs,
        "power": power
    }

    return targets

def save_img_radar(img,rd_target,radar_T, cam_T, cam_K, img_file, save_path):
                
    radar_p = rd_target['coord'].T
    radar_p = np.concatenate((radar_p,np.ones((1,len(radar_p[0])))),axis=0)
    ego_p = radar_T @ radar_p
    cam_p = np.linalg.inv(cam_T) @ ego_p
    cam_uvz = cam_K @ cam_p[:3,:]
    cam_u = cam_uvz[0]/cam_uvz[2]
    cam_v = cam_uvz[1]/cam_uvz[2]
    depth = cam_uvz[2]

    npoints = cam_u.shape[0]

    for i in range(npoints):
                    if cam_v[i]>0 and cam_v[i]<img.shape[0] and cam_u[i]>0 and cam_u[i]<img.shape[1]:
                                    cv2.circle(img,(round(cam_u[i]),round(cam_v[i])),4,(255,0,0))
    img_save_path = save_path + img_file.split('/')[-1] 
    cv2.imwrite(img_save_path,img)



def main():

    sync_path = "/mnt/12T/fangqiang/inhouse/"
    root_path_ls = ["/20220118-13-43-20/",
                    "/20220126-14-52-23/",
                    "/20220126-15-02-25/",
                    "/20220126-15-12-26/",
                    "/20220126-15-22-27/",
                    "/20220126-15-32-28/",
                    "/20220126-15-42-29/",
                    ]
    fx = 1146.501
    fy = 1146.589
    cx = 971.982
    cy = 647.093 
    cam_K = np.array([[fx, 0, cx],
            [0, fy, cy],
            [0, 0,  1]])
    lidar_ext = [0, 0, -0.3, -2.5, 0, 0] 
    radar_ext = [0.06, -0.2, 0.7, -3.5, 2, 180]
    cam_ext =   [-1.793, -0.036, 1.520, -1.66, -0.09, -0.9] 

    radar_T = get_matrix_from_ext(radar_ext)
    lidar_T = get_matrix_from_ext(lidar_ext)
    cam_T = get_matrix_from_ext(cam_ext)

    for i in range(1,len(root_path_ls)):
        root_path = sync_path + root_path_ls[i]
        img_path = root_path + 'sync_img/'
        radar_path = root_path + 'sync_radar/'
        save_path = root_path + 'vis_radar_img/'

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        img_files = sorted(glob(img_path + '*.png'))
        radar_files = sorted(glob(radar_path + '*.csv'))

        for i in tqdm(range(len(img_files))):
            img = cv2.imread(img_files[i])
            radar_data = pd.read_table(radar_files[i], sep=",", header=None)
            rd_data =  radar_data.values[1:,1:].T.astype(np.float32)
            rd_target = get_radar_target(rd_data)
            save_img_radar(img, rd_target, radar_T, cam_T, cam_K, img_files[i], save_path)


if __name__ == '__main__':
    main()