import cv2
import numpy as np

# reference = 'RC v1.1.jpg'
reference = 'RC_RGB_v1.2.1.jpg'
target = 'filters/RCV1.1_P1_12h.jpg'

width = 720
height=1024

def run_split_visualizer(img1, img2, name):
    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    
    cv2.imshow('Original', img1)
    cv2.imshow(name, img2)

    cv2.resizeWindow('Original', width, height)
    cv2.resizeWindow(name, width, height)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def run_visualizer(img1, img2, name):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    # x, y = img2.shape
    x, y, z = img2.shape
    img1 = cv2.resize(img1, (y, x))
    print(img1.shape, img2.shape)
    stacked = np.hstack((img1, img2))
    cv2.imshow(name, stacked)
    cv2.resizeWindow(name, 1440, 960)

    cv2.waitKey(0)
    cv2.destroyAllWindows()