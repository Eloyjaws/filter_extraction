import yaml
import argparse

from calibration import run_calibration
from filter_extraction import run_filter_extraction
from train import run_model_training

PROGRAM_NAME = "SIS Pipeline"
PROGRAM_VERSION = "0.0.1"


if __name__ == "__main__":

    defaults = {}
    force_required = False
    try:
        with open(r'./config.yaml') as file:
            defaults = yaml.safe_load(file)
    except FileNotFoundError:
        force_required = True
        print("Failed to load config file with defaults... \n\n")

    # https://docs.python.org/3/library/argparse.html
    # https://realpython.com/command-line-interfaces-python-argparse/
    parser = argparse.ArgumentParser(
        description="{} - Version {}".format(PROGRAM_NAME, PROGRAM_VERSION))
    subparsers = parser.add_subparsers(
        help='help for subcommand', required=True, dest='mode')

    calibration_parser = subparsers.add_parser(
        'calibrate', help='Run camera calibration routine')
    extraction_parser = subparsers.add_parser(
        'extract', help='Run filter extraction routine')
    train_parser = subparsers.add_parser(
        'train', help='Run model training routine')

    calibration_parser.add_argument("-c", "--camera_id", help="Camera ID - to identify saved camera parameters",
                                    default="default")
    calibration_parser.add_argument("-p", "--path", help="Path to input images for camera calibration",
                                    default=defaults.get('calibration_images', None), required=force_required)
    calibration_parser.add_argument("-r", "--reference", help="Path to the image of the reference card",
                                    default=defaults.get('reference_card', None), required=force_required)
    calibration_parser.add_argument("-ui", "--use_ui_for_calibration", help="Show Results from Calibration",
                                    default=defaults.get('use_ui_for_calibration', False), required=force_required)

    extraction_parser.add_argument("-r", "--reference", help="Path to the image of the reference card",
                                   default=defaults.get('reference_card', None), required=force_required)
    extraction_parser.add_argument("-i", "--inputs", help="Path to the folder containing images of filters",
                                   default=defaults.get('input_images', None), required=force_required)
    extraction_parser.add_argument("-c", "--camera", help="Camera ID - for loading camera calibration parameters",
                                   default=defaults.get('camera_id', None), required=force_required)
    extraction_parser.add_argument("-ui", "--use_ui_for_calibration", help="Use CV window to display threshold image",
                                   default=defaults.get('use_ui_for_calibration', False), required=force_required)
    extraction_parser.add_argument("-sift", "--use_sift", help="Use SIFT to fix perspective and skew",
                                   default=defaults.get('use_sift', True), required=force_required)
    extraction_parser.add_argument("-ssp", "--show_sift_plot", help="Show results from applying SIFT",
                                   default=defaults.get('show_sift_plot', False), required=force_required)
    extraction_parser.add_argument("-sccp", "--show_color_correction_plot", help="Show results from applying Color Calibration",
                                   default=defaults.get('show_color_correction_plot', False), required=force_required)
    extraction_parser.add_argument("-acc", "--apply_color_correction", help="Apply Color Correction",
                                   default=defaults.get('apply_color_correction', True), required=force_required)
    extraction_parser.add_argument("-sec", "--show_extracted_circles", help="Draw circles around extracted filters and visualize results",
                                   default=defaults.get('show_extracted_circles', False), required=force_required)
    

    train_parser.add_argument("-d", "--dataset", help="Path to train dataset",
                                   default=defaults.get('path_to_train_dataset', False), required=force_required)
    train_parser.add_argument("-m", "--metadata", help="Path to metadata",
                                default=defaults.get('path_to_metadata', False), required=force_required)
    train_parser.add_argument("-n", "--modelname", help="Model Name",
                                default=defaults.get('model_name', False), required=force_required)
    train_parser.add_argument("-p", "--use_polynomial_model", help="Use Polynomial Model",
                                default=defaults.get('use_polynomial_model', True), required=force_required)

    

    args = vars(parser.parse_args())
    if args['mode'] == 'calibrate':
        run_calibration(args)
    elif args['mode'] == 'extract':
        run_filter_extraction(args)
    elif args['mode'] == 'train':
        run_model_training(args)
        
