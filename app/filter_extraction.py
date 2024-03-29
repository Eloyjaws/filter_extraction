import cv2
import colour
import numpy as np
import pandas as pd
from pathlib import Path

import utils


def run_filter_extraction(args):
    print("\n\nRunning filter extraction\n\n")

    path_to_reference = args.get('reference')
    path_to_inputs = args.get('inputs')
    camera_id = args.get('camera')

    use_sift = args.get('use_sift')
    show_sift_plot = args.get('show_sift_plot')
    use_ui_for_calibration = args.get('use_ui_for_calibration')
    show_color_correction_plot = args.get('show_color_correction_plot')
    apply_color_correction = args.get('apply_color_correction')
    show_extracted_circles = args.get('show_extracted_circles')

    list_of_images = utils.get_filenames_from_folder(path_to_inputs)
    box_colors = {}

    # Load up the reference image, the positions and colors of all 30 boxes and 32 colors
    (
        reference_image,
        reference_contours,
        ref_colors
    ) = utils.load_image_with_features(path_to_reference)
    assert(len(reference_contours) == 30)
    assert(len(ref_colors) == 32)

    box_colors['reference'] =  ref_colors

    # Iterate through the files, extract filter values, and append the result to a list
    columns = ['filename', 'R', 'G', 'B', 'BC_TOT']
    results = []
    
    def extract_artifacts(original_input_image, LOWE_RATIO = 0.85):

        if(LOWE_RATIO > 1):
            raise Exception("Failed to extract artfacts")

        input_image = utils.run_sift(reference_image, original_input_image, SHOW_PLOT=show_sift_plot, LOWE_RATIO=LOWE_RATIO) if use_sift else original_input_image
        input_image_grayscale = utils.convert_to_grayscale(input_image)

        # Spins up UI to allow experimentation with different low/high values if use_ui = True
        # if use_ui for calibration = = False: will use default values low = 127, high = 255
        input_low, input_high = utils.calibrate_threshold(input_image_grayscale, use_ui=use_ui_for_calibration)

        # Convert grayscale image to B/W
        (input_thresh, input_threshold) = cv2.threshold(input_image_grayscale, input_low, input_high, cv2.THRESH_BINARY_INV)

        # Load up image, extract the positions and colors of all 30 boxes and 32 colors
        (marked_image, target_contours, target_colors) = utils.extract_all_points(input_image, input_threshold)
        
        if((len(target_contours) == 30) and (len(target_colors) == 32)):
            return (input_image, marked_image, target_contours, target_colors)
        
        print("Increasing lowe ratio by 0.05")
        return extract_artifacts(original_input_image, LOWE_RATIO=LOWE_RATIO+0.05)


    for image_path in list_of_images:
        try:
            file_name = image_path.split("/")[-1]
            print(file_name)
            original_input_image = utils.resize(cv2.imread(image_path))

            (input_image, marked_image, contours, target_colors) = extract_artifacts(original_input_image)

            # Store RGB values of extracted boxes
            rgbs = [np.array(list(reversed(bgrs))) for bgrs in target_colors]
            box_colors[file_name] = rgbs
            
            """
            Apply color correction
            This technique ensures that the color correction process does not leave red artifacts
            https://stackoverflow.com/questions/62993366/color-calibration-with-color-checker-using-using-root-polynomial-regression-not
            """
            # Create a float copy
            color_corrected_image = input_image.astype(np.float)
            # Normalise the image to have pixel values from 0 to 1
            color_corrected_image = (color_corrected_image - np.min(color_corrected_image))/np.ptp(color_corrected_image)
            # Decode the image with sRGB EOTF
            color_corrected_image = colour.models.eotf_sRGB(color_corrected_image)
            if apply_color_correction:
                for row in color_corrected_image:
                    # row[:] = colour.colour_correction(row[:], target_colors, ref_colors, 'Cheung 2004')
                    row[:] = colour.colour_correction(row[:], target_colors, ref_colors, 'Finlayson 2015')
                    # row[:] = colour.colour_correction(row[:], target_colors, ref_colors, 'Vandermonde')
            
            # Encode image back to sRGB
            color_corrected_image = colour.models.eotf_inverse_sRGB(color_corrected_image)  
            # Denormalize image to fit 255 pixel values (also clip to ensure values fall between 0 - 255)
            color_corrected_image = np.clip((color_corrected_image * 255), 0, 255)
            # Convert floats back to integers
            color_corrected_image = color_corrected_image.astype(np.uint8)

            if show_color_correction_plot:
                utils.plot([reference_image, input_image, color_corrected_image], ncols=3)

            # Extract filter
            filter_value = utils.extract_filter(color_corrected_image, radius=24, show_circle=show_extracted_circles)
        
            # Append R G and B values to result array
            bgr_strings = [str(intensity) for intensity in filter_value]
            entry = [file_name, bgr_strings[2], bgr_strings[1], bgr_strings[0], 'to_be_calculated']
            results.append(entry)
        except Exception as e:
            print(f"SKIPPING {file_name}:", e)
            continue


    # Write extracted filter values to csv file
    Path("output").mkdir(parents=True, exist_ok=True)
    dataframe = pd.DataFrame(results, columns=columns)
    dataframe.to_csv("output/results.csv", index=False)
    print(f"Results written to: output/results.csv")

    # Write extracted box colors to csv file
    column_names = ['filename'] + [str(i) for i in np.arange(32)]
    dataframe_boxes = pd.DataFrame(box_colors).T.reset_index().set_axis(column_names,1,inplace=False)
    dataframe_boxes.to_csv("output/boxes.csv", index=False)
    print(f"Box colors written to: output/boxes.csv")


if __name__ == "__main__":
    print(f"\n\nYou've executed this program incorrectly. Please run `python main.py extract` \n")
