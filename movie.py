import os
from glob import glob
import moviepy.video.io.ImageSequenceClip
image_files = sorted(glob("/home/toytiny/SAIC_radar/demo_4/*.png"))
fps=2
duration =200

frame = fps * duration

if len(image_files) > frame:
    image_files = image_files[:frame]

clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
clip.write_videofile('/home/toytiny/SAIC_radar/demo_4_img.mp4')