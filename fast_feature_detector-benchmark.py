import numpy as np
import cv2
import cv2 as cv
from config import reference, target
import matplotlib.pyplot as plt
import utils
from utils import run_visualizer, run_split_visualizer

USE_UI_FOR_CALIBRATION = False
USE_SIFT = True
SHOW_SIFT_PLOT = True

reference_card = utils.resize(cv.imread(reference))
original_input_image = utils.resize(cv.imread(target))
input_image = utils.run_sift(
    reference_card, original_input_image, SHOW_PLOT=SHOW_SIFT_PLOT) if USE_SIFT else original_input_image


reference_card_grayscale = utils.convert_to_grayscale(reference_card)
input_image_grayscale = utils.convert_to_grayscale(input_image)

ref_low, ref_high = utils.calibrate_threshold(
    reference_card_grayscale, use_ui=USE_UI_FOR_CALIBRATION)
input_low, input_high = utils.calibrate_threshold(
    input_image_grayscale, use_ui=USE_UI_FOR_CALIBRATION)

(ref_thresh, ref_threshold) = cv2.threshold(
    reference_card_grayscale, ref_low, ref_high, cv2.THRESH_BINARY_INV)
(input_thresh, input_threshold) = cv2.threshold(
    input_image_grayscale, input_low, input_high, cv2.THRESH_BINARY_INV)

# utils.remove_noise_before_keypoint_detecton(input_threshold, use_ui=True)
# utils.extract_all_points(reference_card, ref_threshold)
utils.extract_all_points(input_image, input_threshold)


# utils.plot([input_threshold,
#            input_image], nrows=1, ncols=2)
# utils.plot([reference_card, original_input_image,
#            input_image], nrows=1, ncols=3)

# utils.plot([ref_threshold, input_threshold,
#            input_image], nrows=1, ncols=3)


cnts = cv2.findContours(ref_threshold, cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]


# Technique adapted from https://stackoverflow.com/a/57193144/5837671
points = []

min_area = 5500
max_area = 6500

reference_card_with_rectangles = reference_card.copy()
image_number = 0
for c in cnts:
    area = cv2.contourArea(c)
    # if area > 500:
    #     print(area)
    if area > min_area and area < max_area:
        x, y, w, h = cv2.boundingRect(c)

        points.append((x, y))
        points.append((x+w, y))
        points.append((x, y+h))
        points.append((x+w, y+h))

        ROI = reference_card[y:y+h, x:x+w]
        # cv2.imwrite('ROI_{}.png'.format(image_number), ROI)
        cv2.rectangle(reference_card_with_rectangles,
                      (x, y), (x + w, y + h), (255, 0, 0), 2)
        image_number += 1
# utils.plot([reference_card, ref_threshold])


fast = cv.FastFeatureDetector_create()
fast.setNonmaxSuppression(False)
fast.setThreshold(32)
# # Print all default params
# print("Threshold: ", fast.getThreshold())
# print("nonmaxSuppression: ", fast.getNonmaxSuppression())
# print("neighborhood: ", fast.getType())
# print("Total Keypoints with nonmaxSuppression: ", len(kp))


kp1 = fast.detect(ref_threshold, None)
# print(len(kp1))
referenceWithCircles = utils.drawKeyPts(
    reference_card.copy(), kp1, (0, 0, 255), 2)

kp2 = fast.detect(input_threshold, None)
# print(len(kp2))
targetWithCircles = utils.drawKeyPts(input_image.copy(), kp2, (0, 0, 255), 2)

# utils.plot([reference_card, referenceWithCircles, input_image,
#            targetWithCircles], nrows=2, ncols=2)
