import os
from glob import glob
import moviepy.video.io.ImageSequenceClip
image_files = sorted(glob("/mnt/12T/fangqiang/inhouse/20220222-10-32-36/vis_lidar/*.png"))
fps=10
duration = 100

frame = fps * duration

if len(image_files) > frame:
    image_files = image_files[:frame]

print("-----Start making clip------")
clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
print("-----Save video clip------")
clip.write_videofile('/mnt/12T/fangqiang/inhouse/20220222-10-32-36/vis_pcs_bbx.mp4')