import numpy as np
import cv2
import cv2 as cv
import matplotlib.pyplot as plt

from config import reference, target
import utils
from utils import resize, convert_to_grayscale, calibrate_threshold, plot

reference_card = resize(cv.imread(reference))
original_input_image = resize(cv.imread(target))
input_image = utils.run_sift(
    reference_card, original_input_image, SHOW_PLOT=False)

reference_card_grayscale = convert_to_grayscale(reference_card)
input_image_grayscale = convert_to_grayscale(input_image)

ref_low, ref_high = calibrate_threshold(reference_card_grayscale, use_ui=False)
input_low, input_high = calibrate_threshold(
    input_image_grayscale, use_ui=False)

(ref_thresh, ref_threshold) = cv2.threshold(
    reference_card_grayscale, ref_low, ref_high, cv2.THRESH_BINARY_INV)
(input_thresh, input_threshold) = cv2.threshold(
    input_image_grayscale, input_low, input_high, cv2.THRESH_BINARY_INV)

# plot([ref_threshold, input_threshold])
# utils.run_orb(reference_card, input_image)
# corrected_image = utils.run_sift(reference_card, input_image)
# utils.run_sift(ref_threshold, input_threshold)
utils.extract_fast_features(
    ref_threshold, input_threshold, reference_card, input_image, USE_CONTOURS=True)
