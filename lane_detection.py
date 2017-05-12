import matplotlib.patches as patches
import numpy as np
import cv2
from camera_calibration import CameraCalibration
from edge_detection import EdgeDetection
from lane import LeftLane, RightLane

class LaneDetection:
    
    def __init__(self):
        self.camera = CameraCalibration()
        self.camera.load_calibration()
        self.binary_image = None
        self.binary_warped = None
        self.window_width = 50 
        self.window_height = 80
        
        # for perspective transformation
        top_y = 457
        tl_x = 585
        tr_x = 701
        br_x = 1100
        bl_x = 230
        bot_y = 710

        self.src = np.float32([
            [tl_x, top_y],
            [tr_x, top_y],
            [br_x, bot_y],
            [bl_x, bot_y]
        ])

        top_y = 0
        tl_x = bl_x
        tr_x = br_x
        bot_y = 720

        self.dst = np.float32([
            [tl_x, top_y],
            [tr_x, top_y],
            [tr_x, bot_y],
            [tl_x, bot_y]
        ])

        
    def get_warped_image(self, img):
        undistort_img = self.camera.undistort(img)
        edge_algo = EdgeDetection(undistort_img)
        self.binary_img = edge_algo.process_image()
        
        # get image size (width, height)
        img_size = undistort_img.shape[:2][::-1]

        M = cv2.getPerspectiveTransform(self.src, self.dst)
        self.binary_warped = cv2.warpPerspective(self.binary_img, M, img_size, flags = cv2.INTER_LINEAR)
        
        return self.binary_warped
    
    def detect(self, img):
        self.get_warped_image(img)


    def find_bottom_left_right(self):
        ll = LeftLane(self.binary_warped)
        rr = RightLane(self.binary_warped)

        return ll.identify_lane_start(), rr.identify_lane_start()
