import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt
import pickle
import matplotlib.image as mpimg

    
class CameraCalibration:
    def __init__(self, nx = 9, ny = 6):
        self.nx = nx
        self.ny = ny
        self.mtx = None
        self.dist = None


    def show_chessboard(self, image_name):
        test_image = cv2.imread(image_name)
        test_image_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(test_image, (nx, ny), None)
        cv2.drawChessboardCorners(test_image_gray, (nx,ny), corners, ret)
        
        plt.imshow(test_image_gray, cmap='gray')
        plt.show()
        
        
    def get_obj_img_points(self, pattern):
        images = glob.glob(pattern)

        # generate an array that has possible point locations e.g. 0,0,0; 0,1,0; 0,2,0
        points = np.float32([[x, y, 0] for y in range(self.ny) for x in range(self.nx)])

        obj_points = []
        img_points = []

        print("Processing all files in folder: ", pattern)

        for filename in images:
            image = cv2.imread(filename)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(image, (self.nx, self.ny), None)

            if ret:
                obj_points.append(points)
                img_points.append(corners)
            else:
                print('Chessboard cannot find corneres for:', filename)

        print("Calibration done")
        return obj_points, img_points
    

    def calibrate_using_images(self, image_files = './camera_cal/calibration*.jpg', save_to = 'camera_calib.p'):
        obj_points, img_points = self.get_obj_img_points(image_files)
        
        # figure out the image size
        images = glob.glob(image_files)
        image = mpimg.imread(images[0])
        img_size = image.shape[:2]
        print(img_size)
        
        ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img_size, None, None)
        if ret:
            with open(save_to, "wb") as f:
                pickle.dump({'mtx': self.mtx, 'dist': self.dist}, f)
                print("mtx and dist arrays have been saved to:", save_to)
        else:
            print("Could not calibrate camera using the given parameters")
    
    def load_calibration(self, from_file = './camera_calib.p'):
        with open(from_file, "rb") as f:
            data = pickle.load(f)
            self.mtx = data['mtx']
            self.dist = data['dist']
            
    def undistort(self, img, useOptimal = False):
        if useOptimal:
            h,  w = img.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w,h), 1, (w,h))

            dst = cv2.undistort(img, self.mtx, self.dist, None, newcameramtx)

            # crop the image
            x,y,w,h = roi
            dst = dst[y:y+h, x:x+w]

            return dst
        else:
            return cv2.undistort(img, self.mtx, self.dist, None, None)

if __name__ == "__main__":
    camera = CameraCalibration()
    camera.calibrate_using_images()