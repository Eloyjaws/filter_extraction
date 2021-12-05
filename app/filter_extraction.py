import os
import utils
import colour
import numpy as np
import cv2, cv2 as cv
import matplotlib.pyplot as plt

def run_filter_extraction(args):
    print("\n\nRunning filter extraction\n\n")

    path_to_reference = args.get('reference')
    path_to_inputs = args.get('inputs')
    camera_id = args.get('camera')
    
    use_sift = args.get('use_sift')
    show_sift_plot = args.get('show_sift_plot')
    show_results = args.get('use_ui_for_calibration')
    
    list_of_images = utils.get_filenames_from_folder(path_to_inputs)

    (image, sorted_contours, ref_colors) = utils.load_image_with_features(path_to_reference)
    assert(len(sorted_contours) == 30)
    assert(len(ref_colors) == 30)
    
    results = []
    for image_path in list_of_images:
        (image, sorted_contours, target_colors) = utils.load_image_with_features(image_path)
        if(sorted_contours.shape[0] != 30):
            continue
        print(ref_colors[0], target_colors[0])
    
    # print(target_colors, ref_colors)

    # consider dataframe.append() and dataframe.to_csv()
    # Write results to csv

if __name__ == "__main__":
    print(f"\n\nRun main.py\n\n")