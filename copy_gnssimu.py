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

    inhouse_path = "/mnt/12T/public/inhouse/"
    save_path = "/mnt/12T/fangqiang/inhouse/"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    root_path_ls = ["/20220118-13-43-20/",
                  "/20220126-14-52-23/",
                  "/20220126-15-02-25/",
                  "/20220126-15-12-26/",
                  "/20220126-15-22-27/",
                  "/20220126-15-32-28/",
                  "/20220126-15-42-29/",
                 ]
 
    
    for i in range(len(root_path_ls)):

        root = inhouse_path + root_path_ls[i]
        save_root = save_path + root_path_ls[i] 
        source_file = root + "/output/online/sample/gnssimu-sample-v6@2.csv"
        target_file = save_root + "/gnssimu-sample-v6@2.csv"
        copyfile(source_file, target_file)
        

if __name__ == '__main__':
    main()