import cv2
import numpy as np
import os


class Dataset():
    def __init__(self,):

        self.focal_length = 100
        self.cx = 960
        self.cy = 540

        self.imgs = self.get_images()

        self.K = np.array([[self.focal_length, 0,                 self.cx],
                           [0,                 self.focal_length, self.cy],
                           [0,                 0,                 1]])
        self.points2d = np.load('data/vr2d.npy')
        self.points3d = np.load('data/vr3d.npy')



    def get_images(self):
        # return [cv2.imread("data/"+i,0) for i in sorted(os.listdir("data/")) if i.endswith("png")]
        return [cv2.imread("data/img"+str(i)+'.png', 0) for i in range(1, 4)]
