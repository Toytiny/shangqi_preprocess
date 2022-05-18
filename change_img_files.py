import os
from glob import glob


path = '/mnt/12T/public/inhouse/20220118-13-43-20/input/image/B/'

img_files = glob(path+'*.png')
base_ts = 1642484600284
init_ts = str(base_ts)[6:-1]
img_files_tmp = []
for file in img_files:
                number = file.split('/')[-1].split('.')[0].split('_')[0].zfill(7)
                timestamp = int(number) - 400 #int(init_ts) + base_ts
                new_name = '/'.join(file.split('/')[:-1]) + '/' + str(timestamp) + '_B.png'
                os.renames(file,new_name)  

