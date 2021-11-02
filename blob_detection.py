import numpy as np
import cv2 as cv
from config import reference, target, run_visualizer


ori = cv.imread(reference)
im = cv.imread(reference)

detector = cv.SimpleBlobDetector_create()
keypoints = detector.detect(im)
im_with_keypoints = cv.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

run_visualizer(ori, im_with_keypoints, 'Blob Detection')
