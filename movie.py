import os
from glob import glob
import moviepy.video.io.ImageSequenceClip
image_files = sorted(glob("/mnt/12T/public/view_of_delft/radar/training/image_2/*.jpg"))
fps=10
duration = 1000

frame = fps * duration

if len(image_files) > frame:
    image_files = image_files[:frame]

print("-----Start making clip------")
clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
print("-----Save video clip------")
clip.write_videofile('/home/fangqiang/VoD/vis_image_vod.mp4')