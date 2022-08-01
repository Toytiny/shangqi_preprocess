import csv
import open3d
import numpy as np
from matplotlib import pyplot as plt
from math import sin, cos, pi
import pandas as pd
import os
from shutil import copyfile
from tqdm import tqdm



def main():

    inhouse_path = "/mnt/12T/fangqiang/"
    save_path = "/mnt/12T/fangqiang/inhouse/"

    root_path_ls = ["/20220222-10-32-36-part/"
                 ]
    # utc local
    base_ts_ls = {'20220222-10-32-36-part': [1645497156380,1645497156628]
                  }
    

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    
    for i in range(len(root_path_ls)):

        root = inhouse_path + root_path_ls[i]
        save_root = save_path + root_path_ls[i] 
        source_file = root + "/output/online/sample/gnssimu-sample-v6@2.csv"
        target_file = save_root + "/gnssimu-sample-v6@2.csv"
        copyfile(source_file, target_file)
        

if __name__ == '__main__':
    main()