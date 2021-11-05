import numpy as np
import cv2 as cv
from config import reference, target
from utils import run_visualizer, plot

img1 = cv.imread(reference)
img2 = cv.imread(target)

# img1 = cv.medianBlur(img1, 7)
# img2 = cv.medianBlur(img2, 7)

# gray1 = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
# gray2 = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)

gray1, gray2 = img1, img2

# Initiate FAST detector
star = cv.xfeatures2d.StarDetector_create()

# Initiate BRIEF extractor
brief = cv.xfeatures2d.BriefDescriptorExtractor_create()

# find the keypoints with STAR
kp = star.detect(gray1,None)

# compute the descriptors with BRIEF
kp, des = brief.compute(gray1, kp)

res1 = cv.drawKeypoints(gray1,kp,img1,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Initiate FAST detector
star = cv.xfeatures2d.StarDetector_create()

# Initiate BRIEF extractor
brief = cv.xfeatures2d.BriefDescriptorExtractor_create()

# find the keypoints with STAR
kp = star.detect(gray2,None)

# compute the descriptors with BRIEF
kp, des = brief.compute(gray2, kp)

res2 = cv.drawKeypoints(gray2,kp,img2,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

run_visualizer(res1, res2, 'BRIEF')
