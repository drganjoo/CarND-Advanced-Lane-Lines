import os
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from glob import glob


def mark_lanes_video(video_filename):
    clip = VideoFileClip(video_filename)
    video_with_lanes = clip.fl_image(process_image)
    
    output = os.path.splitext(video_filename)
    video_with_lanes.write_videofile(output[0] + "-Lanes" + output[1], audio=False)
    
def mark_all_frames(video_filename):
    output = os.path.splitext(video_filename)
    output_folder = "{}-frames".format(output[0])
    
    if os.path.exists(output_folder):
        os.rmdir(output_folder)
    
    os.mkdir(output_folder)
    print('Folder created', output_folder)
    
    clip = VideoFileClip(video_filename)

    for i, image in enumerate(clip.iter_frames()):
        mpimg.imsave("{}/{}.jpg".format(output_folder, i), image)
        #mpimg.imsave("frames/%d-lanes.jpg" % (i), process_image(image))
        print("{} generated".format(i), end='\r')
    print("done")

mark_all_frames('project_video.mp4')