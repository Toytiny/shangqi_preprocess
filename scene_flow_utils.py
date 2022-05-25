import os
import numpy as np
from time import *
import pandas as pd
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from RAFT.core.utils.flow_viz import flow_to_image
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm


SIDE_RANGE = (-50, 50)
FWD_RANGE = (0, 100)
HEIGHT_RANGE = (-10,10) 
RES = 0.15625
RADAR_EXT = np.array([0.06, -0.2, 0.7,-3.5, 2, 180])
RESIZE_SCALE = 1/2

def mask_show(mask,target,num_pcs,seq, img_path):
    
    img_path = img_path + "/" + seq + "/" + "mask/"
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    x_img = np.array(target["bev_loc_x"]).astype(np.int32)
    y_img = np.array(target["bev_loc_y"]).astype(np.int32)
    x_max = int((FWD_RANGE[1] - FWD_RANGE[0]) / RES)
    y_max = int((SIDE_RANGE[1] - SIDE_RANGE[0]) / RES)
    im = np.zeros([y_max, x_max,3], dtype=np.uint8)+255
    for i in range(len(x_img)):
        if mask[i]==1:
            im=cv2.circle(im,(x_img[i],y_img[i]),3,(255,0,0))
        else:
            im=cv2.circle(im,(x_img[i],y_img[i]),3,(0,0,255))
            
    im=cv2.putText(im, 'Static', (300,45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)
    im=cv2.putText(im, 'Moving', (300,70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
    path = img_path + "{}.jpg".format(num_pcs)
    cv2.imwrite(path, im)

def align_show(radar1, radar2, tran, num_pcs, seq, img_path):
    
    tran = np.eye(4)
    img_path = img_path + "/" +  seq + "/" + "align/"
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    nps_2 = len(radar2['car_loc_x'])
    nps_1 = len(radar1['car_loc_x'])
    pnts_2 = np.vstack((radar2['car_loc_x'],radar2['car_loc_y'],radar2['car_loc_z'])).T
    h_2 = np.hstack((pnts_2, np.ones((nps_2, 1))))
    a_2 = np.dot(tran, h_2.T)[:3].T 
    x_img_2 = np.floor((a_2[:,0]) / RES).astype(np.int32)
    y_img_2 = np.floor(-(a_2[:,1] + SIDE_RANGE[0])/RES).astype(np.int32)
    x_img_1 = np.array(radar1["bev_loc_x"]).astype(np.int32)
    y_img_1 = np.array(radar1["bev_loc_y"]).astype(np.int32)
    
    x_max = int((FWD_RANGE[1] - FWD_RANGE[0]) / RES)
    y_max = int((SIDE_RANGE[1] - SIDE_RANGE[0]) / RES)
    im = np.zeros([y_max, x_max,3], dtype=np.uint8)+255
    for i in range(nps_1):
        im=cv2.circle(im,(x_img_1[i],y_img_1[i]),2,(255,0,0))
    for i in range(nps_2):
        im=cv2.circle(im,(x_img_2[i],y_img_2[i]),2,(0,255,0))
    im=cv2.putText(im, 'PC1', (300,45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)
    im=cv2.putText(im, 'PC2_Aligned', (280,70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
    path = img_path + "{}.jpg".format(num_pcs)
    cv2.imwrite(path, im)
    

def route_plot(poses,seq):
    
    poses = np.array(poses)

    x_ego = poses[:,0,3]
    y_ego = poses[:,1,3]
    z_ego = poses[:,2,3]
    
    plt.figure()
    plt.winter()
    plt.scatter(-y_ego,x_ego, s=0.5, c=z_ego)
  
    plt.xlabel('X [m]',fontdict={'fontsize': 12, 'fontweight': 'medium'})
    plt.ylabel('Y [m]',fontdict={'fontsize': 12, 'fontweight': 'medium'})
    cb=plt.colorbar()
    plt.tight_layout()
    plt.xlim([-4000,4000])
    plt.ylim([-4000,4000])
    plt.tick_params(labelsize=12)
    cb.set_label('Z [m]',fontdict={'fontsize': 12, 'fontweight': 'medium'})
    cb.ax.tick_params(labelsize=12)
    # cv2.imwrite(path, im)
    path = seq + "route.jpg"
    #plt.savefig(path, dpi=600)
    delta_x = x_ego[1:]-x_ego[:-1]
    delta_y = y_ego[1:]-y_ego[:-1]
    drive_dis = np.sum(np.sqrt(delta_x**2+delta_y**2))
    
    print(drive_dis)
    
    return drive_dis

def show_optical_flow(img1, img2, opt_flow, seq, img_path, num_pcs):

    flow_img = flow_to_image(opt_flow)
    vis_img = np.concatenate((img1,img2,flow_img),axis=0)
    img_path = img_path + "/" + seq + "/" + "opt_flow/"
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    path = img_path + "{}.jpg".format(num_pcs)
    cv2.imwrite(path, vis_img)