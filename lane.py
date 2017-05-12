import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

class Lane:
    def __init__(self, binary_warped):
        self.binary_warped = binary_warped
        self.window_width = 50 
        self.window_height = 80
        self.margin = 100
        self.start_x = 0
        self.window_for_conv = np.ones(self.window_width)

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

    def get_centers(self):
        center = self.identify_lane_start()

        boxes_y = np.arange(0, self.binary_warped.shape[0], lane_algo.window_height)[::-1]
        centers = np.zeros_like(boxes_y)
        conv_max = np.zeros_like(boxes_y)

        # set first center to the last one as the loop uses last center
        centers[-1] = center

        for i in range(0, len(boxes_y)):
            y = boxes_y[i]

            margin_lx = max(centers[i - 1] - self.window_width // 2 - self.margin // 2, 0)
            margin_rx = min(centers[i - 1] + self.window_width // 2 + self.margin // 2, self.binary_warped.shape[1])

            lt = margin_lx, y
            rb = margin_rx, y + self.window_height
            
            image_area = self.binary_warped[y: y + self.window_height, margin_lx : margin_rx]
            conv_max[i], conv = self.find_conv_max(image_area)

            # it could be that there are no pixles in the given window at all
            if conv[conv_max[i]] == 0:
                centers[i] = centers[i - 1]
                conv_max[i] = conv_max[i - 1]
            else:
                centers[i] = conv_max[i] + margin_lx

                # how many pixels were there in this box compared to the previous one
                # if not many then don't believe so much on this box

                if i > 0:
                    div = (conv_max[i - 1]) + 1e-4
                    filled_area_ratio = abs(conv_max[i] / div)

                    if filled_area_ratio < 0.3:
                        center_diff = (centers[i] - centers[i-1]) / self.window_width
                        print('This', centers[i], 'Prev', centers[i-1], 'Diff:', center_diff, 'Filled:', filled_area_ratio)

                        if center_diff > 0.2:
                            centers[i] = np.int(centers[i-1] * -0.8)
                        elif center_diff < -0.2:
                            centers[i] = np.int(centers[i-1] * 0.8)

                        print('This', centers[i], 'Prev', centers[i-1], 'Diff:', center_diff, 'Filled:', filled_area_ratio)

                        # carry forward 80% weight of the previous conv_max as we go up we don't want
                        # one or two pixels in this window to allow another small ratio of pixels to be impacted
                        conv_max[i] = conv_max[i-1] * 0.8
                        
        self.centers = centers
        return self.centers
        
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
    centers = ll.get_centers()

    y = np.arange(0, binary_warped.shape[0], ll.window_height)[::-1]
    output_img = np.dstack((binary_warped * 255, binary_warped * 255, binary_warped * 255))

    for i, center in enumerate(centers):
        lt = center - ll.window_width // 2, y[i]
        rb = center + ll.window_width // 2, y[i] + ll.window_height
        print(center, lt, rb)

        cv2.rectangle(output_img, lt, rb, (255, 0, 0), 10)
        
    plt.imshow(output_img)
    plt.show()