import numpy as np
import cv2
import cv2 as cv
from config import reference, targets_with_ring_light_v2 as targets
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

results = []

for target in targets[:]:
    print(target)
    original_input_image = utils.resize(cv.imread(target))
    input_image = utils.run_sift(
        reference_card, original_input_image, SHOW_PLOT=SHOW_SIFT_PLOT) if USE_SIFT else original_input_image
    input_image_grayscale = utils.convert_to_grayscale(input_image)

    input_low, input_high = utils.calibrate_threshold(
        input_image_grayscale, use_ui=USE_UI_FOR_CALIBRATION)

    (input_thresh, input_threshold) = cv2.threshold(
        input_image_grayscale, input_low, input_high, cv2.THRESH_BINARY_INV)

    im2, c2, trgt_colors = utils.extract_all_points(input_image, input_threshold)
    # print("Ref: \n", ref_colors, "\n", "Target: \n", trgt_colors)
    if(len(ref_colors) != len(trgt_colors)):
        print("Could not extract target colors")
        continue

    # color_corrector = utils.get_color_calibration_model(ref_colors, trgt_colors)
    corrected = input_image.copy()  
    for row in corrected:
        # row[:] = colour.colour_correction(row[:], trgt_colors, ref_colors, 'Cheung 2004')
        # row[:] = colour.colour_correction(row[:], trgt_colors, ref_colors, 'Finlayson 2015')
        row[:] = colour.colour_correction(row[:], trgt_colors, ref_colors, 'Vandermonde')
        # row[:] = color_corrector.predict(row[:])
    # utils.plot([im1, im2, corrected], ncols=3)
    utils.plot([reference_card, input_image, corrected], ncols=3)
    results.append(input_image)
    results.append(corrected)
utils.plot(results, nrows=len(results)//2, ncols=2)