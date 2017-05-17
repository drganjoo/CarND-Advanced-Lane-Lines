# Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Final output video:

[Output Video](project_video-Lanes.mp4)


Different classes have been defined for finding lane lines.

- CameraCalibration (camera_calibration.py)
- EdgeDetection (edge_detection.py)
- Lane (Base class) (lane.py)
- LeftLane (Child class) (lane.py)
- RightLane(Child class) (lane.py)
- LaneDetection (lane_detection.py)

In order to demonstrate different functionality of the solution, jupyter notebooks have also been used:

- ImageToBinaryWarped - this notebook shows the process from test image to binary_warped image
- LaneMarkingTest - this notebook shows the process from binary_warped to lane line identification


[distort_example]: ./output_images/distortion_correction.png "Distortion Correction"
[frame_distort]: ./output_images/actual_correction.png "Actual Frame Correction"
[cs_sample]: ./output_images/cs_sample.png "Sample for Color Spaces"
[hls]: ./output_images/hls.png "hls"
[hsv]: ./output_images/hsv.png "hsv"
[lab]: ./output_images/lab.png "lab"
[luv]: ./output_images/luv.png "luv"
[ycbcr]: ./output_images/ycbcr.png "ycbcr"
[xyz]: ./output_images/xyz.png "xyz"
[cs_used]: ./output_images/cs_used.png "Color Space Used"
[binary_warped]: ./output_images/binary_warped.png "Binary warped image"
[perspective]: ./output_images/perspective.png "Straight Perspective"
[perspective_explain]: ./output_images/perspective_explain.png "Perspective Explain"
[perspective_sample]: ./output_images/perspective_sample.png "Perspective Sample"
[lane_start]: ./output_images/lane_start.png "Lane Start"
[windows]: ./output_images/windows.png "Windows"
[pixels]: ./output_images/pixels.png "pixels"
[polynomial]: ./output_images/polynomial.png "polynomial"
[lanes]: ./output_images/lanes.png "lanes"

## Pipeline Summary

- Undistort image
- Convert image to L,B,S channels
- Carry out gradient on L and color thresholding on S channels
- Combine both into binary image
- Carry out perspective transformation
- Use sliding window with convolution to find lane lines
- Choose pixels that lie within the sliding windows
- Fit a polynomial to the chosen pixels
- On next frame use polynomials to select pixels
- Smooth each frame by using low pass filter
- In case polynomial based search doesn't yeild a line in 3 consecutive frames, search using sliding windows again
- Update polynomial on each frame based on the new chosen pixels
- Draw polygon on the warped image
- Unwarp image and display it

## Camera Distortion Correction

I've used cv2's **findChessboardCorners** and **calibrateCamera** to find out the camera distortion. Sample files provided in the camera_cal folder were used for calibration.

A separate class **CameraCalibration** (camera_calibration.py) has been written that can calibrate using images residing in a folder, save the clibration to a pickled dump file and use that to restore calibration later on.

Test for calibration is provided in ImageToBinaryWarped.ipynb:

```
from camera_calibration import CameraCalibration

camera = CameraCalibration()
camera.calibrate_using_images()
```
**Output**:
```
Processing all files in folder:  ./camera_cal/calibration*.jpg
Chessboard cannot find corneres for: ./camera_cal/calibration5.jpg
Chessboard cannot find corneres for: ./camera_cal/calibration1.jpg
Chessboard cannot find corneres for: ./camera_cal/calibration4.jpg
Calibration done
mtx and dist arrays have been saved to: camera_calib.p
```

*Note: cv2.findChessboardCorners was not able to find chessboard corners for calibration1.jpg, calibration4.jpg and calibration15.jpg*

## Distortion Corrected Image

![distort_example]

Actual frame undistorted:

![frame_distort]

## Color Transformations

In order to convert colored image into a binary image suitable for searching lane lines, a spearate class has been written **EdgeDetection (edge_detection.py)**

### Exploring Color Spaces

All possible color space conversions provided by cv2 were tested on a few sample frames and test_images using:

```
all_spaces = ['cv2.' + i for i in dir(cv2) 
        if i.startswith('COLOR_RGB2') and not i.endswith('_FULL')]
```
Output:
```
['cv2.COLOR_RGB2BGR', 'cv2.COLOR_RGB2BGR555', 'cv2.COLOR_RGB2BGR565', 
'cv2.COLOR_RGB2BGRA', 'cv2.COLOR_RGB2GRAY', 'cv2.COLOR_RGB2HLS', 
'cv2.COLOR_RGB2HSV', 'cv2.COLOR_RGB2LAB', 'cv2.COLOR_RGB2LUV', 'cv2.COLOR_RGB2Lab', 
'cv2.COLOR_RGB2Luv', 'cv2.COLOR_RGB2RGBA', 'cv2.COLOR_RGB2XYZ', 
'cv2.COLOR_RGB2YCR_CB', 'cv2.COLOR_RGB2YCrCb', 'cv2.COLOR_RGB2YUV', 
'cv2.COLOR_RGB2YUV_I420', 'cv2.COLOR_RGB2YUV_IYUV', 'cv2.COLOR_RGB2YUV_YV12']
```

Sample Image:

![cs_sample]

Some spaces that were considered:

![hls]
![hsv]
![lab]
![luv]
![ycbcr]
![xyz]

Finally, after playing around with different color spaces, carrying out color and gradient thresholds on them, I decided to use the following:

**L, B, S**

This is being done in function **extract_channels** (edge_detection.py line #14)

A sample of the color space is:


![cs_used]

#### Color Thresholding

Color thresholding is done on **S channel** (line # 74 of edge_detection.py)

#### Gradient Thresholding

Gradient thresholding (edge_detection.py line # 83) is done using:

- cv2.Sobel on X using the **L channel** (line # 34/84 of edge_detection.py)
- Magnitude thresholding on **B channel** (line # 49/85 of edge_detection.py)

#### Combined Thresholding

Both gradient and color thresholded images are combined wherever either one of them is a 1 (line # 88 edge_detection.py)

![binary_warped]

## Perspective Transformation

The source and destination points used for cv2.warpPerspective function, defined in  clockwise fashion, are:

```
src =   [  585.   457.]
        [  701.   457.]
        [ 1100.   710.]
        [  230.   710.]

dst =   [  230.     0.]
        [ 1100.     0.]
        [ 1100.   720.]
        [  230.   720.]
```

These are defined on line #: 22 of lane_detection.py. 

The idea I used was to just move the top two points of the polygon to make it a rectangle hence the bottom two points are kept the same (almost).

![perspective_explain]

Sample output on straight_line1.jpg:

![perspective]

Output on an actual frame:

![perspective_sample]

# Finding Lane Lines

A class Lane has been defined in lane.py that handles all lane identification

## Finding Lane Start

Bottom 3/4th of the height of the image is used to sum up all pixels across X and then find the region that has the most number of pixels.

Line #38 of lane.py has **identify_lane_start()**:
```
y = int(3 / 4 * self.binary_warped.shape[0])
bottom_img = self.get_bottom_image(y)
sum_cols = np.sum(image_area, axis=0)
conv = np.convolve(self.window_for_conv, sum_cols, 'same')
center = np.argmax(conv)
```
![lane_start]

## Sliding Window

Line #57 of lane.py **lookfor_window_centers**:

Window Size:  50,80

Y offsets of all windows, with the 0th one being the bottom most, are defined using:

```
boxes_y = np.arange(0, self.binary_warped.shape[0], self.window_height)[::-1]
```
Then wihtin a margin of 100px from the last window's center a region is defined, convolved with an array containing all ones and having a size of 50px. New window is chosen based on the maximum convolution within the 100px margin.

![windows]

## Choosing Pixels

All pixels that lie within the sliding windows are chosen. Line # 118 (lane.py)

![pixels]

## Fitting a poynomial

A polynomial is fit for all pixels that have been chosen and then it is evaluated for 0 to image height. Line # 142 (lane.py)

![polynomial]

## Find Using Polynomial

The class "Lane" defines two methods, **find_using_sliding_windows** (Line #146 lane.py) that uses sliding algorithm to find windows and then fits a new polynomial to it and the other one **find_using_polynomial** (Line #166 Lane.py) that uses an existing polynomial to choose pixels that lie within 50px of the already defined polynomial and then fits a new one using the newly chosen pixels.

## Smoothing

A low pass filter is used to smooth polynomials from one frame to another (line # 200 lane.py):

```
current_fit = current_fit * A + last_fit * (1 - A)
```

## Finding Curvature

Line # 154 of lane.py defines method to calculate the radius of curvature for the lane lines. The following formula is being used:

```
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

# Fit new polynomials to x,y in world space
fit_cr = np.polyfit(self.chosen_y * ym_per_pix, self.chosen_x * xm_per_pix, 2)
y_eval_world = 719 * ym_per_pix

# R curve = ((1 + (2Ay + B) ^ 2) ^ 3/2) / 2A
self.radius_of_curvature = ((1 + (2 * fit_cr[0] * y_eval_world + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
```

## Drawing Lane Polygon

Line #50 of lane_detection.py has the code that processes the new frame and returns an image that has the polygon drawn and curvature / offset written on the image. It uses LeftLane and RightLane classes to find the two lanes and then averages the curvature defined by the two lane objects.

![lanes]
