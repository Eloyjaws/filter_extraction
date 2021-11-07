import numpy as np
import cv2
import cv2 as cv

import matplotlib.pyplot as plt
from config import reference, target, WIDTH
import utils

USE_SIFT = True
SHOW_PLOT = False

# Load image
reference_card = utils.resize(cv.imread(reference))
image = utils.resize(cv.imread(target))

corrected_image = utils.run_sift(reference_card, image, SHOW_PLOT=SHOW_PLOT) if USE_SIFT else image
gray_image = utils.convert_to_grayscale(corrected_image)

circles = cv2.HoughCircles(gray_image, cv2.HOUGH_GRADIENT,
                           1, 100, param1=100, param2=70, minRadius=0, maxRadius=0)

if circles is None:
    circles = np.array([[]])
circles = np.uint16(np.around(circles))
padding = 8
for (x, y, r) in circles[0]:
    cv2.putText(corrected_image, f"{x}, {y}, {r}", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.circle(corrected_image, (x, y), r-padding, (0, 255, 0), 3)

cv2.imshow("Detected Circle", corrected_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
