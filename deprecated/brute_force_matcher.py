# Source: https://docs.opencv.org/3.4.4/dc/dc3/tutorial_py_matcher.html

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from config import reference, target
from utils import plot


img1 = cv.imread(reference)
img2 = cv.imread(target)

img1 = cv.medianBlur(img1, 7)
img2 = cv.medianBlur(img2, 7)

gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
res1 = cv.drawKeypoints(
    gray1, kp1, img1, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

sift = cv.SIFT_create()
kp2, des2 = sift.detectAndCompute(gray2, None)
res2 = cv.drawKeypoints(
    gray2, kp2, img2, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


# Draw Top Matches

# create BFMatcher object
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1, des2)

# Sort them in the order of their distance.
matches = sorted(matches, key=lambda x: x.distance)

# Draw first 10 matches.
img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:10], None, flags=2)

cv.imshow("Matches", img3)

cv.waitKey(0)
cv.destroyAllWindows()
