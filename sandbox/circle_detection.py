import numpy as np
import cv2
import cv2 as cv
import sys

import matplotlib.pyplot as plt
from config import reference, target, targets, WIDTH
import utils


# https://docs.opencv.org/3.4/d4/d70/tutorial_hough_circle.html
# https://docs.opencv.org/4.x/da/d53/tutorial_py_houghcircles.html




USE_SIFT = True
SHOW_PLOT = False

reference_card = utils.resize(cv.imread(reference))

for target in targets[::2]:
    # Load image
    image = utils.resize(cv.imread(target))

    corrected_image = utils.run_sift(reference_card, image, SHOW_PLOT=SHOW_PLOT) if USE_SIFT else image
    gray_image = utils.convert_to_grayscale(corrected_image)

    # https://dsp.stackexchange.com/questions/22648/in-opecv-function-hough-circles-how-does-parameter-1-and-2-affect-circle-detecti
    

    # dp: This parameter is the inverse ratio of the accumulator resolution 
    # to the image resolution (see Yuen et al. for more details). 
    # Essentially, the larger the dp gets, the smaller the accumulator array gets.
    dp = 1
    # minDist: Minimum distance between the center (x, y) coordinates of detected circles. 
    # If the minDist is too small, multiple circles in the same neighborhood as the 
    # original may be (falsely) detected. 
    # If the minDist is too large, then some circles may not be detected at all.
    minDist = 100
    # param1: Gradient value used to handle edge detection in the Yuen et al. method.
    param1 = 100
    # param2: Accumulator threshold value for the cv2.HOUGH_GRADIENT method. 
    # The smaller the threshold is, the more circles will be detected (including false circles). 
    # The larger the threshold is, the more circles will potentially be returned.
    param2 = 75
    # minRadius: Minimum size of the radius (in pixels).
    minRadius = 0
    # maxRadius: Maximum size of the radius (in pixels).
    maxRadius = 0

    circles = cv2.HoughCircles(gray_image, cv2.HOUGH_GRADIENT,
                            dp, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

    if circles is None:
        circles = np.array([[]])
    circles = np.uint16(np.around(circles))
    padding = 30
    circles_detected = len(circles[0])
    if(circles_detected > 1):
        print("Error: More than one circle detected")
        sys.exit(1)
    if(circles_detected == 0):
        print("Error: No filter detected")
        sys.exit(1)
    
    
    for (x, y, r) in circles[0]:
        mask = np.zeros((corrected_image.shape[:2]), np.uint8)
        cv2.circle(mask,(x,y),r-padding,(255,0,0), -1)
        bgr = cv2.mean(corrected_image, mask=mask)
        cv2.circle(corrected_image, (x, y), r-padding, (0, 0, 255), 2)
        print(bgr[:3])

    cv2.imshow("Detected Circle", corrected_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
