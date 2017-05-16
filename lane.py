import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

MAX_UNDETECTED_FRAMES = 3

class Lane:
    def __init__(self, binary_warped):
        self.binary_warped = binary_warped
        self.window_width = 50 
        self.window_height = 80
        self.margin = 100
        self.start_x = 0
        self.window_for_conv = np.ones(self.window_width)
        self.centers = None
        self.chosen_x = None
        self.chosen_y = None
        self.current_fit = None
        self.current_fitx = None
        self.current_fity = None
        self.radius_of_curvature = 0.0
        self.detected = False
        self.last_fit = None
        self.times_undetected = 0
        self.last_chosen_x = None
        self.last_chosen_y = None
        
    def identify_lane_start(self):
        # start from about 3/4 of the image from top
        y = int(3 / 4 * self.binary_warped.shape[0])

        bottom_img = self.get_bottom_image(y)
        center, _ = self.find_conv_max(bottom_img)
        center += self.start_x
        return center

    def find_conv_max(self, image_area):
        sum_cols = np.sum(image_area, axis=0)
        conv = np.convolve(self.window_for_conv, sum_cols, 'same')
        conv_max = np.argmax(conv)
        return conv_max, conv

    def get_bottom_image(self, y):
        raise Exception("Use a child class not the Lane base class")
        None

    def lookfor_window_centers(self):
        boxes_y = np.arange(0, self.binary_warped.shape[0], self.window_height)[::-1]
        centers = np.zeros_like(boxes_y)
        conv_max = np.zeros_like(boxes_y)
        conv = np.zeros((len(boxes_y), self.margin))

        # set first center to the last one as the loop uses last center
        centers[-1] = self.identify_lane_start()
        
        for i in range(0, len(boxes_y)):
            y = boxes_y[i]

            margin_lx = max(centers[i - 1] - self.margin // 2, 0)
            margin_rx = min(centers[i - 1] + self.margin // 2, self.binary_warped.shape[1])

            lt = margin_lx, y
            rb = margin_rx, y + self.window_height

            image_area = self.binary_warped[y: y + self.window_height, margin_lx : margin_rx]
            conv_max[i], conv[i] = self.find_conv_max(image_area)
            
            # it could be that there are no pixles in the given window at all
            if conv[i,conv_max[i]] == 0:
                centers[i] = centers[i - 1]
                conv_max[i] = conv_max[i - 1]
                conv[i] = conv[i-1]
            else:
                centers[i] = conv_max[i] + margin_lx

                # how many pixels were there in this box compared to the previous one
                # if not many then don't believe so much on this box

            if i > 0:
                filled_previous = np.sum(conv[i - 1]) + 1e-4
                filled_area = np.sum(conv[i])
                filled_area_ratio = abs(filled_area / filled_previous)

                if filled_area_ratio < 0.3:
                    center_diff = (centers[i] - centers[i-1]) / self.window_width
                    print('Filled in area is less than the last. Diff Ratio', center_diff)

                    # a maximum of 30% change in center is allowed in case
                    # not many pixels are found in this window
                    
                    if abs(center_diff) > 0.3:
                        if centers[i] < centers[i-1]:
                            centers[i] = centers[i-1] * 0.7
                        else:
                            centers[i] = centers[i-1] * 1.3

                    # carry forward 80% weight of the previous conv_max as we go up we don't want
                    # one or two pixels in this window to allow another small ratio of pixels to be impacted
                    conv[i] = conv[i-1] * 0.8

        self.centers = centers
        return centers

    def get_window_centers(self):
        if self.centers is None:
            self.lookfor_window_centers()

        return self.centers

    def choose_window_pixels(self):
        self.chosen_y = []
        self.chosen_x = []

        windows = self.get_windows_from_centers()
        for window in windows:
            lt = window[0]
            rb = window[1]

            # area of the image wihtin this window
            image_area = self.binary_warped[lt[1]:rb[1], lt[0]:rb[0]]
            nonzero = image_area.nonzero()
            nonzeroy = nonzero[0] + lt[1]
            nonzerox = nonzero[1] + lt[0]

            self.chosen_x.append(nonzerox)
            self.chosen_y.append(nonzeroy)

        self.chosen_x = np.concatenate(self.chosen_x)
        self.chosen_y = np.concatenate(self.chosen_y)

        return (self.chosen_x, self.chosen_y)

    def fit_polynomial(self):
        self.current_fit = np.polyfit(self.chosen_y, self.chosen_x, 2)
        self.current_fity = np.arange(0, self.binary_warped.shape[0])
        self.current_fitx = np.polyval(self.current_fit, self.current_fity)

    def find_using_sliding_window(self):
        self.lookfor_window_centers()
        self.choose_window_pixels()
        self.fit_polynomial()

        return self.current_fit

    def calculate_curvature(self):
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        fit_cr = np.polyfit(self.chosen_y * ym_per_pix, self.chosen_x * xm_per_pix, 2)
        y_eval_world = 719 * ym_per_pix

        # R curve = ((1 + (2Ay + B) ^ 2) ^ 3/2) / 2A
        self.radius_of_curvature = ((1 + (2 * fit_cr[0] * y_eval_world + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])

    def find_using_polynomial(self):
        self.last_fit = self.current_fit
        self.last_chosen_x = self.chosen_x
        self.last_chosen_y = self.chosen_y

        nonzero = self.binary_warped.nonzero()
        nonzeroy = nonzero[0]
        nonzerox = nonzero[1]

        center = np.polyval(self.last_fit, nonzeroy)

        margin = self.margin // 2
        lane_inds = (nonzerox > (center - margin)) & (nonzerox < (center + margin))

        self.chosen_x = nonzerox[lane_inds]
        self.chosen_y = nonzeroy[lane_inds]

        if (len(self.chosen_x) == 0):
            self.chosen_x = self.last_chosen_x
            self.chosen_y = self.last_chosen_y

            self.times_undetected += 1
            if self.times_undetected > MAX_UNDETECTED_FRAMES:
                print("Finding window centers and fitting polynomial again")
                self.find_using_sliding_window()
        else:
            self.fit_polynomial()
            

    def process_frame(self, binary_warped):
        self.binary_warped = binary_warped

        if not self.detected:
            self.find_using_sliding_window()
            self.detected = True
        else:
            self.find_using_polynomial()
            print('using existing polynomial to find lane lines')
        
        self.calculate_curvature()

    def get_windows_from_centers(self):
        # window y locations's in reverse order (from bottom to up)
        y = np.arange(0, self.binary_warped.shape[0], self.window_height)[::-1]
        centers = self.get_window_centers()
        
        windows = []

        for i, center in enumerate(centers):
            lt = center - self.window_width // 2, y[i]
            rb = center + self.window_width // 2, y[i] + self.window_height
            windows.append([lt, rb])

        return windows
            
        
class LeftLane(Lane):
    def __init__(self, binary_warped):
        super().__init__(binary_warped)

    def get_bottom_image(self, y):
        mid_point = self.binary_warped.shape[0] // 2
        return self.binary_warped[y:,:mid_point]

class RightLane(Lane):
    def __init__(self, binary_warped):
        super().__init__(binary_warped)
        self.mid_point = self.binary_warped.shape[0] // 2
        self.start_x = self.mid_point

    def get_bottom_image(self, y):
        return self.binary_warped[y:,self.mid_point:]

if __name__ == "__main__":
    from lane_detection import LaneDetection

    filename = './test_images/straight_lines2.jpg'
    img = mpimg.imread(filename)

    lane_algo = LaneDetection()
    binary_warped = lane_algo.get_warped_image(img)
    ll = LeftLane(binary_warped)
    centers = ll.lookfor_window_centers()

    y = np.arange(0, binary_warped.shape[0], ll.window_height)[::-1]
    output_img = np.dstack((binary_warped * 255, binary_warped * 255, binary_warped * 255))

    for i, center in enumerate(centers):
        lt = center - ll.window_width // 2, y[i]
        rb = center + ll.window_width // 2, y[i] + ll.window_height
        print(center, lt, rb)

        cv2.rectangle(output_img, lt, rb, (255, 0, 0), 10)
        
    plt.imshow(output_img)
    plt.show()