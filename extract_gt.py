import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import os
from os.path import join as osp
from tqdm import tqdm

def get_info_idx(info_name, title):
    idx_dict = {}
    idx_list = []
    for i in info_name:
        idx = np.where(title == i)
        if idx[0].size == 0:
            print("info name of: " + str(i) + " doesn't exist")
            raise RuntimeError
        idx_dict[i] = idx[0].item()
        idx_list += [idx[0].item()]
    return idx_dict, idx_list



def extract_gt_box(root_path,save_path):
    gt_file = osp(root_path, "GT.csv")
    gt_data = np.loadtxt(gt_file, dtype=str, delimiter=',')
    title = gt_data[0]
    
    # filter gt based on table title
    info_list = ["stamp_sec", "type", "center.x", "center.y", "center.z", "length", "width", "height", "direction.x", "direction.y", "direction.z", "track_id","type_confidence"]
    info_idx, info_idx_list = get_info_idx(info_list, title)
    filter_gt = gt_data[:, info_idx_list]
    gt_arr = np.array(filter_gt[1:,:]).astype(float)
    
    #bias = 18000 # 18 second compensation
    
    # remove duplicate timestamp
    ts = gt_arr[:,0]
    ts = np.unique(ts)
    
    # collect gt data to each ts dict
    gt_dict = {}
    prev_ts = gt_arr[0,0]
    temp_dict = {}
    obj_cnt = 0
    for gt in gt_arr:
        ts = gt[0]
        if ts != prev_ts:
            obj_cnt = 0
            gt_dict[ts] = temp_dict
            temp_dict = {}
            prev_ts = ts
        else:
            temp_dict[obj_cnt] = np.concatenate((np.array([obj_cnt]), gt[1:]), axis=0)
            obj_cnt += 1
    
    # export gt to separate file
    gt_export_dir = osp(save_path, 'gt_abs')
    if not os.path.exists(gt_export_dir):
        os.mkdir(gt_export_dir)
    for k in tqdm(gt_dict):
        gt_obj = gt_dict[k]
        gt_obj_arr = np.array(list(gt_obj.values()))
        gt_ts = int(k*1e3)
        gt_fname = str(gt_ts).zfill(13) + ".csv"
        full_fname = osp(gt_export_dir, gt_fname)
        np.savetxt(full_fname, gt_obj_arr, fmt="%s")
        
def main():    
    
    root_path = ["../20220118-13-43-20/",
                  "../20220126-14-52-23/",
                  "../20220126-15-02-25/",
                  "../20220126-15-12-26/",
                  "../20220126-15-22-27/",
                  "../20220126-15-32-28/",
                  "../20220126-15-42-29/",
                  
                 ]
    save_path = root_path
    for i in range(len(root_path)):
        extract_gt_box(root_path[i],save_path[i])
        
if __name__ == '__main__':
    main()