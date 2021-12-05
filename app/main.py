import yaml
import argparse

from calibration import run_calibration
from filter_extraction import run_filter_extraction

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
    extraction_parser.add_argument("-show_sift_plot", "--show_sift_plot", help="Show results from applying SIFT",
                                   default=defaults.get('show_sift_plot', False), required=force_required)

    args = vars(parser.parse_args())
    if args['mode'] == 'calibrate':
        run_calibration(args)
    else:
        run_filter_extraction(args)
