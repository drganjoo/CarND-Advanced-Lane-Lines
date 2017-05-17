import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle


class EdgeDetection:
    
    def __init__(self, img):
        self.img = img
        self.extract_channels()
        
    def extract_channels(self):
        self.hls = cv2.cvtColor(self.img, cv2.COLOR_RGB2HLS)
        self.lab = cv2.cvtColor(self.img, cv2.COLOR_RGB2LAB)

        self.l = self.lab[:,:,0]
        self.b = self.lab[:,:,2]
        self.s = self.hls[:,:,2]
        
        # remove everything < 200 intensity to 0 in the L channel
        self.l[self.l < 200] = 0

    def process_image(self):
        grad = self.get_gradient_img()
        color_threshold = self.get_color_thresh_img()

        combined = np.zeros_like(grad)
        mask = (grad == 1) | (color_threshold == 1)
        combined[mask] = 1
        return combined

    def abs_sobel_thresh(self, channel, orient='x', sobel_kernel=3, thresh=(0, 255)):
        if orient == 'x':
            sobel = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        else:
            sobel = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        abs_sobel = np.abs(sobel)
        scaled = np.uint8(255.0 * abs_sobel / np.max(abs_sobel))

        mask = (scaled >= thresh[0]) & (scaled <= thresh[1])

        grad_binary = np.zeros_like(channel)
        grad_binary[mask] = 1
        return grad_binary

    def mag_thresh(self, channel, sobel_kernel=3, mag_thresh=(0, 255)):
        sobelx = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        mag = np.sqrt(sobelx ** 2 + sobely ** 2)
        scaled = np.uint8(255.0 * mag / np.max(mag))

        mask = (scaled >= mag_thresh[0]) & (scaled <= mag_thresh[1])

        mag_binary = np.zeros_like(channel)
        mag_binary[mask] = 1
        return mag_binary

    def dir_threshold(self, channel, sobel_kernel=3, thresh=(0, np.pi/2)):
        sobelx = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        dir_sobel = np.arctan2(np.abs(sobely), np.abs(sobelx))

        mask = (dir_sobel >= thresh[0]) & (dir_sobel <= thresh[1])

        dir_binary = np.zeros_like(channel)
        dir_binary[mask] = 1
        return dir_binary

    def get_color_thresh_img(self):
        color_thresh = (140,255)
        
        #mask = (self.s >= color_thresh[0]) & (self.s <= color_thresh[1])
        mask = (self.b >= color_thresh[0]) & (self.b <= color_thresh[1])
        binary = np.zeros_like(self.b)
        binary[mask] = 1
        return binary

    def get_gradient_img(self):
        grad_x = self.abs_sobel_thresh(self.l, 'x', sobel_kernel=7, thresh=(40,120))
        mag = self.mag_thresh(self.b, sobel_kernel=19, mag_thresh=(40, 150))
        mask = (grad_x == 1) | (mag == 1)
        
        combined = np.zeros_like(self.s)
        combined[mask] = 1
        return combined

