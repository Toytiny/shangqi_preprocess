import os
import ujson
import numpy as np
from time import *
import argparse
import pandas as pd
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import cv2
from shutil import copyfile
import torch
from tqdm import tqdm
from pose_extract import get_trans_from_gnssimu, get_matrix_from_ext, get_interpolate_pose


ROOT_PATH = "/mnt/12T/fangqiang/inhouse/"
SAVE_PATH = "/mnt/12T/fangqiang/inhouse_comp/"
RADAR_EXT = np.array([0.06, -0.2, 0.7,-3.5, 2, 180])
# utc local
BASE_TS_LS = {'20220118-13-43-20': [1642484600284,1642484600826],
              '20220126-14-52-23': [1643179942119,1643179944003],
              '20220126-15-02-25': [1643180543397,1643180545286],
              '20220126-15-12-26': [1643181144484,1643181146376],
              '20220126-15-22-27': [1643181745461,1643181747357],
              '20220126-15-32-28': [1643182346486,1643182348386],
              '20220126-15-42-29': [1643182947438,1643182949343]
              }

def main():

    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    
    seqs = sorted(os.listdir(ROOT_PATH))
    ## extrinsic parameters of radar
    ego_to_radar = get_matrix_from_ext(RADAR_EXT)
    radar_to_ego = np.linalg.inv(ego_to_radar)

    for seq in seqs:
        print("Start to process the seq {}".format(seq))
        base_ts = BASE_TS_LS[seq][0] ## utc base timestamp, for lidar and robosense
        base_ts_local = BASE_TS_LS[seq][1] ## local base timestamp, for radar and pose

        ## some used paths
        pose_path = ROOT_PATH + seq + "/" + "gnssimu-sample-v6@2.csv"
        data_path = ROOT_PATH + seq + "/" + "sync_radar/"
        gt_path = ROOT_PATH + seq + "/" + "sync_gt/"
        img_path = ROOT_PATH + seq + "/" + "sync_img/"
        pose_save_path = SAVE_PATH + seq + "/" + "gnssimu-sample-v6@2.csv"
        data_save_path = SAVE_PATH + seq + "/" + "sync_radar/"
        gt_save_path = SAVE_PATH + seq + "/" + "sync_gt/"
        img_save_path = SAVE_PATH + seq + "/" + "sync_img/"
        
        ## Getting data list 
        pcs_ls = sorted(os.listdir(data_path))
        gts_ls = sorted(os.listdir(gt_path))
        img_ls = sorted(os.listdir(img_path))
        pcs_len = len(pcs_ls)

        # copy all gt, pose and image files
        if not os.path.exists(os.path.join(ROOT_PATH, seq)):
            os.makedirs(os.path.join(ROOT_PATH, seq))
        if not os.path.exists(gt_save_path):
            os.makedirs(gt_save_path)
        if not os.path.exists(img_save_path):
            os.makedirs(img_save_path)
        if not os.path.exists(data_save_path):
            os.makedirs(data_save_path)
        copyfile(pose_path, pose_save_path)
        print("Success to save the pose files")
        for gts in tqdm(gts_ls):
            # construct full file path
            source_file = os.path.join(gt_path + gts)
            target_file = os.path.join(gt_save_path + gts)
            copyfile(source_file, target_file)
        print("Success to save the gt files")
        # for img in tqdm(img_ls):
        #     # construct full file path
        #     source_file = os.path.join(img_path + img)
        #     target_file = os.path.join(img_save_path + img)
        #     copyfile(source_file, target_file)
        # print("Success to save the image files")

        ## extract the pose data
        ego_poses, pose_ts = get_interpolate_pose(pose_path,scale=1)
        radar_poses =  ego_poses @ ego_to_radar 
        pose_ts = (pose_ts-base_ts_local)/1e3

        for i in tqdm(range(pcs_len)):
            pd_data = pd.read_table(data_path+pcs_ls[i], sep=",", header=None)
            data = pd_data.values[1:,1:].T.astype(np.float32)
            pc = data[:3,:]
            vel_r = data[7, :]
            ts = (int(pcs_ls[i].split('.')[0])-base_ts)/1e3
            # match radar and pose timestamps
            diff = abs(ts - pose_ts)
            idx = diff.argmin()
            if (idx+3)<len(radar_poses):
                pose = radar_poses[idx-3]
                pose_next = radar_poses[idx+3]
            else:
                pose = radar_poses[idx-6]
                pose_next = radar_poses[idx]
            ## transformation of the radar sensor in two pose interval
            tran = np.dot(np.linalg.inv(pose), pose_next)
            pc_o3d = o3d.utility.Vector3dVector(pc.T)
            pc_geo = o3d.geometry.PointCloud()
            pc_geo.points = pc_o3d
            pc_tran = pc_geo.transform(np.linalg.inv(tran)).points
            flow_rigid = np.asarray(pc_tran).T-np.asarray(pc_o3d).T
            # project the rigid flow onto the radial direction
            flow_rigid_proj = np.sum(flow_rigid*pc,axis=0)/(np.linalg.norm(pc,axis=0))
            # use flow to approximate the radial velocity induced by ego-motion
            if (idx+3)<len(radar_poses):
                vel_rigid_proj = flow_rigid_proj / (pose_ts[idx+3] - pose_ts[idx-3])
            else:
                vel_rigid_proj = flow_rigid_proj / (pose_ts[idx] - pose_ts[idx-6])
            # compenated vel_r 
            vel_comp = vel_r -  vel_rigid_proj
            # add compenated velocity to data
            new_data = np.vstack((data, vel_comp))
            target_path = os.path.join(data_save_path, str(pcs_ls[i]).zfill(13))
            save_data = pd.DataFrame(new_data.T)
            save_data.to_csv(target_path)
        print("Success to save the new radar data files")

if __name__ == "__main__":
    main()


