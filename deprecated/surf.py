import numpy as np
import cv2 as cv
from config import reference, target
from utils import plot

ori = cv.imread(reference)
img = cv.imread(reference)

surf = cv.xfeatures2d.SURF_create(400)
kp, des = surf.detectAndCompute(img,None)
image = cv.drawKeypoints(img,kp,None,(255,0,0),4)

run_visualizer([ori, image])
