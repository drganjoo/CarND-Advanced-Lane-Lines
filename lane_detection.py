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
        self.binary_warped = None
        self.window_width = 50 
        self.window_height = 80
        self.left_lane = None
        self.right_lane = None
        self.radius_of_curvature = 0.0
        
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

    def undistort(self, img):
        self.undistort_img = self.camera.undistort(img)
        return self.undistort_img

    def get_warped_image(self, img):
        self.undistort(img)

        edge_algo = EdgeDetection(self.undistort_img)
        binary_img = edge_algo.process_image()
        
        # get image size (width, height)
        img_size = self.undistort_img.shape[:2][::-1]

        M = cv2.getPerspectiveTransform(self.src, self.dst)
        self.binary_warped = cv2.warpPerspective(binary_img, M, img_size, flags = cv2.INTER_LINEAR)
        return self.binary_warped

    def get_unwarped_img(self, binary_warped):
        img_size = binary_warped.shape[:2][::-1]

        M = cv2.getPerspectiveTransform(self.dst, self.src)
        binary_unwarped = cv2.warpPerspective(binary_warped, M, img_size, flags = cv2.INTER_LINEAR)
        return binary_unwarped
    
    def draw_lanes(self):
        blank_image = np.zeros_like(self.binary_warped)
        lane_image = np.dstack((blank_image, blank_image, blank_image))

        pts_left = np.dstack((self.left_lane.current_fitx, self.left_lane.current_fity))
        pts_right = np.dstack((self.right_lane.current_fitx[::-1], self.right_lane.current_fity[::-1]))
        pts = np.hstack((pts_left, pts_right))

        cv2.fillPoly(lane_image, np.int_(pts), (0, 255, 0))

        unwarped_img = self.get_unwarped_img(lane_image)
        img_with_lanes = cv2.addWeighted(self.undistort_img, 1, unwarped_img, 0.3, 0)

        return img_with_lanes

    def write_curvature_offset(self, img):
        str = "Radius of Curvature: {} m".format(np.int(self.radius_of_curvature))
        cv2.putText(img, str, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255))
        str = "Lane Offset: {} m".format(np.int(self.offset))
        cv2.putText(img, str, (0, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255))
        
        return img

    def set_curvature_offset(self):
        center = (self.left_lane.current_fitx[-1] + self.right_lane.current_fitx[-1]) / 2
        self.offset = (640 - center) * 3.7/700 # meters per pixel in x dimension
        self.radius_of_curvature = (self.left_lane.radius_of_curvature + self.right_lane.radius_of_curvature) / 2

    def process_image(self, img):
        self.get_warped_image(img)
        if self.left_lane is None:
            self.left_lane = LeftLane(self.binary_warped)
        if self.right_lane is None:
            self.right_lane = RightLane(self.binary_warped)

        self.left_lane.process_frame(self.binary_warped)
        self.right_lane.process_frame(self.binary_warped)
        self.set_curvature_offset()

        img_with_lanes = self.draw_lanes()
        img_with_lanes = self.write_curvature_offset(img_with_lanes)

        return img_with_lanes

    def find_bottom_left_right(self):
        ll = LeftLane(self.binary_warped)
        rr = RightLane(self.binary_warped)

        return ll.identify_lane_start(), rr.identify_lane_start()
