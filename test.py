import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from lane_detection import LaneDetection
from lane import LeftLane, RightLane

filename = './test_images/straight_lines2.jpg'
img = mpimg.imread(filename)

lane_algo = LaneDetection()
lanes = lane_algo.process_image(img)

# str = "Radius of Curvature: {} m".format(np.int(lane_algo.radius_of_curvature))
# cv2.putText(lanes, str, (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255))

f, (ax1, ax2) = plt.subplots(1,2,figsize=(20,10))

ax1.imshow(img)
ax2.imshow(lanes)
plt.show()
