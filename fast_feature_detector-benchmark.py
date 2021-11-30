import numpy as np
import cv2
import cv2 as cv
from config import reference, targets
import matplotlib.pyplot as plt
import utils
from utils import run_visualizer, run_split_visualizer
import colour


USE_UI_FOR_CALIBRATION = False
USE_SIFT = True
SHOW_SIFT_PLOT = False

# USE_SIFT = False
# SHOW_SIFT_PLOT = True

reference_card = utils.resize(cv.imread(reference))
reference_card_grayscale = utils.convert_to_grayscale(reference_card)
ref_low, ref_high = utils.calibrate_threshold(
    reference_card_grayscale, use_ui=USE_UI_FOR_CALIBRATION)
(ref_thresh, ref_threshold) = cv2.threshold(
    reference_card_grayscale, ref_low, ref_high, cv2.THRESH_BINARY_INV)
im1, c1, ref_colors = utils.extract_all_points(reference_card, ref_threshold)

for target in targets:
    original_input_image = utils.resize(cv.imread(target))
    input_image = utils.run_sift(
        reference_card, original_input_image, SHOW_PLOT=SHOW_SIFT_PLOT) if USE_SIFT else original_input_image
    
    input_image_grayscale = utils.convert_to_grayscale(input_image)

    input_low, input_high = utils.calibrate_threshold(
        input_image_grayscale, use_ui=USE_UI_FOR_CALIBRATION)

    (input_thresh, input_threshold) = cv2.threshold(
        input_image_grayscale, input_low, input_high, cv2.THRESH_BINARY_INV)

    # utils.remove_noise_before_keypoint_detecton(input_threshold, use_ui=True)
    im2, c2, trgt_colors = utils.extract_all_points(input_image, input_threshold)
    print("Ref: \n", ref_colors, "\n", "Target: \n", trgt_colors)
    color_corrector = utils.get_color_calibration_model(ref_colors, trgt_colors)
    utils.plot([im1, im2])
    corrected = input_image.copy()  
    corrected[:] = colour.colour_correction(corrected[:], trgt_colors, ref_colors, 'Finlayson 2015')
    utils.plot([im1, im2, corrected], ncols=3)

