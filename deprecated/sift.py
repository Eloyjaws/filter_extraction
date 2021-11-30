import numpy as np
import cv2 as cv
import cv2
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
    img1, kp1, img1, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


sift = cv.SIFT_create()
kp2, des2 = sift.detectAndCompute(gray2, None)
res2 = cv.drawKeypoints(
    img2, kp2, img2, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


# run_visualizer(res1, res2, 'SIFT')

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0, 0] for i in range(len(matches))]

# ratio test as per Lowe's paper
for i, (m, n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i] = [1, 0]

draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=(255, 0, 0),
                   matchesMask=matchesMask,
                   flags=0)

img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)


# store all the good matches as per Lowe's ratio test.
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)

MIN_MATCH_COUNT = 10
if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
else:
    print("Not enough matches are found - %d/%d" %
          (len(good), MIN_MATCH_COUNT))

out = cv2.warpPerspective(img2, M, (img1.shape[1], img1.shape[0]))
plot([img1, img2, out, img3], nrows=2, ncols=2)
