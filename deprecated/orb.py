import numpy as np
import cv2 as cv
from config import reference, target
import utils

img1 = utils.resize(cv.imread(reference))
img2 = utils.resize(cv.imread(target))
# img = cv.medianBlur(img, 7)
# Source: https://docs.opencv.org/3.4.4/d1/de0/tutorial_py_feature_homography.html
# Slightly edited by methylDragon

MIN_MATCH_COUNT = 10

# Initiate ORB detector
orb = cv.ORB_create(edgeThreshold=25, patchSize=31, nlevels=8, fastThreshold=32,
                    scaleFactor=1.2, WTA_K=2, scoreType=cv.ORB_HARRIS_SCORE, firstLevel=0, nfeatures=50)

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

for k in range(len(kp1)):
    print(kp1[k].pt)

for k in range(len(kp2)):
    print(kp2[k].pt)

print(len(kp1), len(kp2))


FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH,
                    table_number=6,  # 12
                    key_size=12,     # 20
                    multi_probe_level=1)  # 2

# Then set number of searches. Higher is better, but takes longer
search_params = dict(checks=100)

flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for match in matches:
    if(len(match) != 2):
        continue
    m, n = match
    if m.distance < 0.7*n.distance:
        good.append(m)

if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

    matchesMask = mask.ravel().tolist()

    try:
        h, w, d = img1.shape
    except:
        h, w = img1.shape

    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]
                     ).reshape(-1, 1, 2)

    dst = cv.perspectiveTransform(pts, M)
    img2 = cv.polylines(img2, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
else:
    print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
    matchesMask = None

draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                   singlePointColor=None,
                   matchesMask=matchesMask,  # draw only inliers
                   flags=2)
img3 = cv.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

utils.plot([img3], ncols=1)
