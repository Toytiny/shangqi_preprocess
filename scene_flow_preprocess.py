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
from tqdm import tqdm
from pose_extract import get_trans_from_gnssimu, get_matrix_from_ext, get_interpolate_pose
from RAFT.core.raft import RAFT
from RAFT.core.utils.flow_viz import flow_to_image
from scene_flow_utils import mask_show, align_show, route_plot, show_optical_flow
import torch


ROOT_PATH = "/mnt/12T/fangqiang/inhouse/"
SAM_PATH = "/mnt/12T/fangqiang/scene_flow_samples/"
SHOW_PATH = "/mnt/12T/fangqiang/scene_flow_imgs/"

SIDE_RANGE = (-50, 50)
FWD_RANGE = (0, 100)
HEIGHT_RANGE = (-10,10) 
RES = 0.15625
RADAR_EXT = np.array([0.06, -0.2, 0.7,-3.5, 2, 180])
CAM_EXT =  np.array([-1.793, -0.036, 1.520, -1.66, -0.09, -0.9])
RESIZE_SCALE = 1/2
# utc local
BASE_TS_LS = {'20220118-13-43-20': [1642484600284,1642484600826],
              '20220126-14-52-23': [1643179942119,1643179944003],
              '20220126-15-02-25': [1643180543397,1643180545286],
              '20220126-15-12-26': [1643181144484,1643181146376],
              '20220126-15-22-27': [1643181745461,1643181747357],
              '20220126-15-32-28': [1643182346486,1643182348386],
              '20220126-15-42-29': [1643182947438,1643182949343]
              }
fx = 1146.501
fy = 1146.589
cx = 971.982
cy = 647.093 
   

CAM_K = np.array([[fx, 0, cx],
                [0, fy, cy],
                [0, 0,  1]])
RADAR_T = get_matrix_from_ext(RADAR_EXT)
CAM_T = get_matrix_from_ext(CAM_EXT)


def get_rotation(arr):
    x,y,_ = arr
    yaw = np.arctan(y/x)
    angle = np.array([0, 0, yaw])
    r = R.from_euler('XYZ', angle)
    return r.as_matrix()

def get_bbx_param(obj_info):
    
    center = obj_info[2:5] #+ np.array([-2.5, 0, 0])
    # enlarge the box field to include points with meansure errors
    extent = obj_info[5:8] + 1.0
    angle = obj_info[8:11]
    rot_m = get_rotation(angle)
    obbx = o3d.geometry.OrientedBoundingBox(center.T, rot_m, extent.T)
    
    return obbx

def transform_bbx(obj_bbx,trans):
    
    # eight corner points 
    bbx_pnts = obj_bbx.get_box_points()
    o3d_pnts = o3d.geometry.PointCloud()
    o3d_pnts.points = bbx_pnts
    # transform eight points
    o3d_pnts.transform(trans)
    obj_bbx = o3d.geometry.OrientedBoundingBox.create_from_points(o3d_pnts.points)
    
    return obj_bbx
    
def get_flow_label(target1,tran,gt1,gt2,radar_to_ego,mode):
    
    pc1 = np.vstack((target1["car_loc_x"], target1["car_loc_y"], target1["car_loc_z"]))
    num_pnts = np.size(pc1,1)
    pc1 = o3d.utility.Vector3dVector(pc1.T)
    num_obj = np.size(gt1,0)
    labels = np.zeros((num_pnts,3),dtype=np.float32)
    mask = np.zeros(num_pnts,dtype=np.float32)
    in_idx_ls = []
    in_confs = np.zeros(num_pnts,dtype=np.float32)
    in_labels = np.zeros((num_pnts,3),dtype=np.float32)
    
    # get flow labels for points within objects 
    for i in range(num_obj):
        if gt1.ndim==2 and gt2.ndim==2:
            track_id1 = gt1[i,-2]
            next_idx = np.where(gt2[:,-2] == track_id1)[0]
            if len(next_idx)!=0 and not (gt1[i,5:8]<0.1).any(): # avoid too small boxes 
                # object in the first frame
                obj1 = gt1[i,:]
                obj_bbx1 = get_bbx_param(obj1)
                bbx1 = transform_bbx(obj_bbx1,radar_to_ego)
                # object in the second frame
                obj2 = gt2[next_idx[0],:]
                obj_bbx2 = get_bbx_param(obj2)
                bbx2 = transform_bbx(obj_bbx2,radar_to_ego)
                # select radar points within the bounding box in the first frame
                in_idx = bbx1.get_point_indices_within_bounding_box(pc1)
                if len(in_idx)>0:
                    in_labels[in_idx] = bbx2.center-bbx1.center   
                    in_confs[in_idx] =  obj1[-1]
                    in_idx_ls.extend(in_idx)
            else: 
                continue
        else:
            continue
        
    if mode=='test':
        # get rigid flow labels for all points
        pc1_rg = o3d.utility.Vector3dVector(np.asarray(pc1))
        pc1_geo = o3d.geometry.PointCloud()
        pc1_geo.points = pc1_rg
        pc1_tran = pc1_geo.transform(np.linalg.inv(tran)).points
        flow_r = np.asarray(pc1_tran)-np.asarray(pc1_rg)
        
        # get non-rigid components for inbox points
        flow_nr = in_labels[in_idx_ls] - flow_r[in_idx_ls]
    
        # obtain the index for foreground (dynamic) points 
        fg_idx = np.array(in_idx_ls)[np.linalg.norm(flow_nr,axis=1)>0.05]
        
    else:
        fg_idx = in_idx_ls
        flow_r = np.zeros((np.size(np.asarray(pc1),0),np.size(np.asarray(pc1),1)))
        
        
    if len(fg_idx)>0:
        bg_idx = np.delete(np.arange(0,num_pnts),fg_idx)
    else:
        bg_idx = np.arange(0,num_pnts)

    # fill the labels of foreground and background, obtain the mask
    mask[bg_idx] = 1
    labels[bg_idx] = flow_r[bg_idx]
    if len(fg_idx)>0:
        labels[fg_idx] = in_labels[fg_idx]
        mask[fg_idx] = 1-in_confs[fg_idx]
    
 
   
    return labels, mask
        
    
def get_radar_target(data,ts,trans,pose_ts):

    ## use the original right-hand coordinate systen, front is x, left is y, up is z
    x_points = data[0, :]
    y_points = data[1, :]
    z_points = data[2, :]
    vel_r = data[7, :]
    rcs = data[6,:]
    power = data[5,:]
    
    # dis = np.sqrt(x_points**2+y_points**2+z_points**2)
    # if dis.max()<100:
    #     state = 'near'
    # else:
    #     state = 'mid'
    f_filt = np.logical_and((x_points > FWD_RANGE[0]), (x_points < FWD_RANGE[1]))
    s_filt = np.logical_and((y_points > SIDE_RANGE[0]), (y_points < SIDE_RANGE[1]))
    h_filt = np.logical_and((z_points > HEIGHT_RANGE[0]), (z_points < HEIGHT_RANGE[1]))
    filt = np.logical_and(np.logical_and(f_filt, s_filt), h_filt)
    indices = np.argwhere(filt).flatten()

    # KEEPERS
    x_points = x_points[indices]
    y_points = y_points[indices]
    z_points = z_points[indices]
    points = np.vstack((x_points, y_points, z_points)).T
    N = points.shape[0]
    vel_r = vel_r[indices]
    rcs = rcs[indices]
    power = power[indices]

    # SHIFT to the BEV view

    x_img = np.floor((x_points) / RES)
    y_img = np.floor(-(y_points + SIDE_RANGE[0])/RES)
    bev_vel_r = vel_r / RES
    bev_vel_r[np.isnan(bev_vel_r)] = 0
    vel_r = bev_vel_r * RES

    # match radar and pose timestamps
    diff = abs(ts - pose_ts)
    idx = diff.argmin()
    pose = trans[idx]
    
    targets = {
        "car_loc_x": x_points,
        "car_loc_y": y_points,
        "car_loc_z": z_points,
        "car_vel_r": vel_r,
        "bev_loc_x": x_img,
        "bev_loc_y": y_img,
        "bev_vel_r": bev_vel_r,
        "rcs": rcs,
        "power": power,
        "pose": pose,
    }

    return targets


def estimate_optical_flow(img1,img2,model):

    resize_dim = (int(RESIZE_SCALE*img1.shape[1]),int(RESIZE_SCALE*img1.shape[0]))
    img1 = cv2.resize(img1,resize_dim)
    img2 = cv2.resize(img2,resize_dim)
    img1_torch = torch.from_numpy(img1).cuda().unsqueeze(0).transpose(1,3)
    img2_torch = torch.from_numpy(img2).cuda().unsqueeze(0).transpose(1,3)
    opt_flow = model(img1_torch, img2_torch, 12)
    np_flow = opt_flow.squeeze(0).permute(2,1,0).cpu().detach().numpy()
    resize_dim = (int(img1.shape[1]/RESIZE_SCALE),int(img1.shape[0]/RESIZE_SCALE))
    flow = cv2.resize(np_flow, resize_dim)

    return flow

def info_from_opt_flow(target, opt_flow):

    radar_p = np.vstack((-target["car_loc_y"], -target["car_loc_z"], target["car_loc_x"]))
    num_pnts = np.size(radar_p,1)
    radar_p = np.concatenate((radar_p,np.ones((1,len(radar_p[0])))),axis=0)
    ego_p = RADAR_T @ radar_p
    cam_p = np.linalg.inv(CAM_T) @ ego_p
    cam_uvz = CAM_K @ cam_p[:3,:]
    cam_u = np.round(cam_uvz[0]/cam_uvz[2]).astype(np.int)
    cam_v = np.round(cam_uvz[1]/cam_uvz[2]).astype(np.int)
    filt_uv = np.logical_and(np.logical_and(cam_v>0, cam_v<opt_flow.shape[0]),\
         np.logical_and(cam_u>0, cam_u<opt_flow.shape[1]))

    radar_opt = opt_flow[cam_v[filt_uv],cam_u[filt_uv]]

    opt_info = {"radar_u": cam_u,
                "radar_v": cam_v,
                "filt": filt_uv,
                "opt_flow": radar_opt,
                }
    return opt_info


def init_raft():

        parser = argparse.ArgumentParser()
        
        parser.add_argument('--model', default= "/home/fangqiang/shangqi_preprocess/models/raft-small.pth", help="restore checkpoint")
        parser.add_argument('--path', help="dataset for evaluation")
        parser.add_argument('--small', action='store_false', help='use small model')
        parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
        parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
        raft_args = parser.parse_args()

        raft = RAFT(raft_args).cuda()
        raft = torch.nn.DataParallel(raft)
        raft.load_state_dict(torch.load(raft_args.model))

        return raft


def main():
    
    if not os.path.exists(SAM_PATH):
        os.makedirs(SAM_PATH)
    if not os.path.exists(SHOW_PATH):
        os.makedirs(SHOW_PATH)
        
    seqs = sorted(os.listdir(ROOT_PATH))
    splits = {'test':[seqs[0]]}#{'train' : seqs[1:6], 'val': [seqs[6]], 'test': [seqs[0]],}

    ## extrinsic parameters of radar
    ego_to_radar = get_matrix_from_ext(RADAR_EXT)
    radar_to_ego = np.linalg.inv(ego_to_radar)
    dis_all = 0
    raft_model = init_raft()
    ## Read, process and save samples 
    for split in splits:
        
        num_pcs = 0 
        num_seq = 0
        ## Save path for current split
        sam_path = SAM_PATH + "/"+ split
        if not os.path.exists(sam_path):
            os.makedirs(sam_path)
        show_path = SHOW_PATH+ "/"+ split 
        if not os.path.exists(show_path):
            os.makedirs(show_path)
        
        for seq in splits[split]:
            
            base_ts = BASE_TS_LS[seq][0] ## utc base timestamp, for lidar and robosense
            base_ts_local = BASE_TS_LS[seq][1] ## local base timestamp, for radar and pose
            
            ## some used paths
            pose_path = ROOT_PATH + seq + "/" + "gnssimu-sample-v6@2.csv"
            gt_path = ROOT_PATH + seq + "/" + "sync_gt/"
            data_path = ROOT_PATH + seq + "/" + "sync_radar/"
            img_path = ROOT_PATH + seq + "/" + "sync_img/"
            
            # filter uncorrect groundtruth output
            if split == 'test':
                filt_path = ROOT_PATH + seq + "/" + "test_filt2/"
                filt_ls = sorted(os.listdir(filt_path))
                filt_idx = []
                for filt in filt_ls:
                    filt_idx.append(int(filt.split('.')[0]))
                
            ## extract the pose data
            ego_poses, pose_ts = get_interpolate_pose(pose_path,scale=1)
            radar_poses =  ego_poses @ ego_to_radar 
            pose_ts = (pose_ts-base_ts_local)/1e3
            dis = route_plot(ego_poses,seq)
            dis_all+=dis
            ## Getting radar raw data and gt data    
            pcs_ls = sorted(os.listdir(data_path))
            gts_ls = sorted(os.listdir(gt_path))
            pcs_len = len(pcs_ls)
            ## Getting image data files
            imgs_ls = sorted(os.listdir(img_path))
    
            print('Starting Aggregating radar pcs for {}: seq-{}'.format(split,num_seq))    
            for i in tqdm(range(pcs_len-1)):
                
                pc_path1 = pcs_ls[i]
                pc_path2 = pcs_ls[i+1]
                gt_path1 = gts_ls[i]
                gt_path2 = gts_ls[i+1]
                ts1 = (int(pc_path1.split('.')[0])-base_ts)/1e3
                ts2 = (int(pc_path2.split('.')[0])-base_ts)/1e3
                
                pd_data1 = pd.read_table(data_path+pc_path1, sep=",", header=None)
                data1 = pd_data1.values[1:,1:].T.astype(np.float32)
                pd_data2 = pd.read_table(data_path+pc_path2, sep=",", header=None)
                data2 = pd_data2.values[1:,1:].T.astype(np.float32)
                gt1 = np.loadtxt(gt_path+gt_path1)
                gt2 = np.loadtxt(gt_path+gt_path2)
                
                target1= get_radar_target(data1,ts1,radar_poses,pose_ts)
                target2= get_radar_target(data2,ts2,radar_poses,pose_ts)
                
            
                if np.size(target1['car_loc_x'],0)>256 and np.size(target2['car_loc_x'],0)>256:
                    
                    ## obtain groundtruth for the test set (only use reliable robosense output)
                    if split == 'test':
                        if (i in filt_idx) and ((i+1) in filt_idx):
                            ## transformation from coordinate 1 to coordinate 2
                            tran = np.dot(np.linalg.inv(target1['pose']), target2['pose'])
                            ## obtain the scene flow labels from rigid transform and tracking object bounding boxes
                            labels, mask = get_flow_label(target1,tran,gt1,gt2,radar_to_ego,'test')
                            mask_show(mask,target1,num_pcs,seq,show_path)
                            # show aligned two point clouds
                            align_show(target1,target2,tran,num_pcs,seq,show_path)
                            opt_info={}
                        else: 
                            continue
                    ## obtain groundtruth for train and val    
                    else:
                        tran = np.dot(np.linalg.inv(target1['pose']), target2['pose'])
                        labels, mask = get_flow_label(target1,tran,gt1,gt2,radar_to_ego,'train')
                        mask_show(mask,target1,num_pcs,seq,show_path)
                        align_show(target1,target2,tran,num_pcs,seq,show_path)
                        ## aggregate info from the images through optical flow
                        img1 = cv2.imread(img_path + imgs_ls[i])
                        img2 = cv2.imread(img_path + imgs_ls[i+1])
                        opt_flow = estimate_optical_flow(img1, img2, raft_model)
                        show_optical_flow(img1, img2, opt_flow, seq, show_path, num_pcs)
                        opt_info = info_from_opt_flow(target1,opt_flow)


                    num_pcs+=1
                    out_path_cur = sam_path + '/' + "radar_seqs-{}_samples-{}.json".format(num_seq, num_pcs)
                    for r_key in target1:
                          target1[r_key] = target1[r_key].tolist()
                          target2[r_key] = target2[r_key].tolist()
                    for o_key in opt_info:
                          opt_info[o_key] = opt_info[o_key].tolist()
                         
                    sample = {"pc1": target1,
                              "pc2": target2,
                              "interval": ts2-ts1,
                              "trans": tran.tolist(),
                              "gt": labels.tolist(),
                              "mask": mask.tolist(),
                              "opt_info": opt_info,
                              }
                    ujson.dump(sample, open(out_path_cur, "w"))
                    
            num_seq+=1
    print(dis_all)          

if __name__ == "__main__":
    main()
