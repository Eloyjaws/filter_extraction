import os
import utils
import colour
import numpy as np
import pandas as pd
import cv2
import cv2 as cv
import matplotlib.pyplot as plt
import colour


def run_filter_extraction(args):
    print("\n\nRunning filter extraction\n\n")

    use_color_correction = True

    path_to_reference = args.get('reference')
    path_to_inputs = args.get('inputs')
    camera_id = args.get('camera')

    use_sift = args.get('use_sift')
    show_sift_plot = args.get('show_sift_plot')
    show_ui_results = args.get('use_ui_for_calibration')
    show_color_plot = args.get('show_color_correction_plot')

    list_of_images = utils.get_filenames_from_folder(path_to_inputs)

    (reference_image, reference_contours,
     ref_colors) = utils.load_image_with_features(path_to_reference)
    assert(len(reference_contours) == 30)
    assert(len(ref_colors) == 30)

    columns = ['filename', 'R', 'G', 'B', 'BC_TOT']
    results = []
    box_colors = []
    for image_path in list_of_images:
        original_input_image = utils.resize(cv.imread(image_path))
        input_image = utils.run_sift(
            reference_image, original_input_image, SHOW_PLOT=show_sift_plot) if use_sift else original_input_image
        input_image_grayscale = utils.convert_to_grayscale(input_image)

        input_low, input_high = utils.calibrate_threshold(
            input_image_grayscale, use_ui=show_ui_results)

        (input_thresh, input_threshold) = cv2.threshold(
            input_image_grayscale, input_low, input_high, cv2.THRESH_BINARY_INV)

        (target_image, target_contours,
         target_colors) = utils.extract_all_points(
            input_image, input_threshold)

        box_colors.append(target_colors)

        color_corrected_image = input_image.copy()
        for row in color_corrected_image:
            row[:] = colour.colour_correction(
                row[:], target_colors, ref_colors, 'Vandermonde')
        if show_color_plot:
            utils.plot([reference_image, input_image,
                       color_corrected_image], ncols=3)

        # Extract filter
        file_name = image_path.split("/")[-1]
        filter_value = utils.extract_filter(color_corrected_image, input_image)
        if(filter_value == -1):
            print(f"Could not extract filter for {file_name}")
            continue
        bgr_strings = [str(intensity) for intensity in filter_value]
        entry = [file_name, bgr_strings[2], bgr_strings[1],
                 bgr_strings[0], 'to_be_calculated']
        results.append(entry)
    # Write results to csv
    dataframe = pd.DataFrame(results, columns=columns)
    dataframe.to_csv("results.csv", index=False)

    dataframe_boxes = pd.DataFrame(box_colors, columns=np.arange(30))
    dataframe_boxes.to_csv("boxes.csv", index=False)

if __name__ == "__main__":
    print(f"\n\nRun main.py\n\n")
