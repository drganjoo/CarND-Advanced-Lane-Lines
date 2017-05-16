from edge_detection import EdgeDetection
from camera_calibration import CameraCalibration
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

camera = CameraCalibration()
camera.load_calibration()

#img = mpimg.imread('./test_images/straight_lines1.jpg')
img = mpimg.imread('./test_images/test5.jpg')
img = camera.undistort(img)

edge = EdgeDetection(img)
ksize = 7

gradx = edge.abs_sobel_thresh(edge.s, orient='x', sobel_kernel=ksize, thresh=(20, 90))
grady = edge.abs_sobel_thresh(edge.s, orient='y', sobel_kernel=ksize, thresh=(30, 140))
mag_binary = edge.mag_thresh(edge.s, sobel_kernel=9, mag_thresh=(30, 90))
dir_binary = edge.dir_threshold(edge.s, sobel_kernel=31, thresh=(45 * np.pi / 180.0, 60 * np.pi / 180))
color_thresh = edge.get_color_thresh_img()

plt.figure(figsize=(20,10))
plt.suptitle('Test Image', fontsize=20)
plt.imshow(img)
plt.show()

f, (ax1, ax2) = plt.subplots(1,2, figsize=(24,9))
ax1.imshow(gradx, cmap='gray')
ax1.set_title('gradx', fontsize=20)
ax2.imshow(grady, cmap='gray')
ax2.set_title('grady', fontsize=20)

f, (ax3, ax4) = plt.subplots(1,2, figsize=(24,9))
ax3.imshow(mag_binary, cmap='gray')
ax3.set_title('mag', fontsize=20)
#ax4.imshow(dir_binary, cmap='gray')
ax4.imshow(color_thresh, cmap='gray')
ax4.set_title('dir', fontsize=20)

final_img = edge.process_image()
plt.figure(figsize=(20,10))
plt.suptitle('Final Image', fontsize=20)
plt.imshow(final_img, cmap='gray')
plt.show()

