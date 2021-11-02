import numpy as np
import cv2 as cv
from config import reference, target, run_visualizer

img1 = cv.imread(reference)
img2 = cv.imread(target)

img1 = cv.medianBlur(img1, 7)
img2 = cv.medianBlur(img2, 7)

gray1 = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()
kp, des = sift.detectAndCompute(gray1,None)
res1 = cv.drawKeypoints(img1,kp,img1,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

sift = cv.SIFT_create()
kp, des = sift.detectAndCompute(gray2,None)
res2 = cv.drawKeypoints(img2,kp,img2,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

run_visualizer(res1, res2, 'SIFT')
