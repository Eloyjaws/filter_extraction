# Use pyyaml to save and load camera parameters
import os
import sys
import yaml
import numpy as np
import cv2
from pprint import pprint

import utils

# Interesting links
# http://www.ipb.uni-bonn.de/html/teaching/photo12-2021/2021-pho1-22-Zhang-calibration.pptx.pdf
# https://docs.opencv.org/3.4.15/d9/d0c/group__calib3d.html
# https://stackoverflow.com/questions/43563988/camera-calibration-without-chess-boards-corners
# http://www.vision.caltech.edu/bouguetj/calib_doc/
# https://opencv.org/evaluating-opencvs-new-ransacs/


def run_calibration(args):
    print("Running camera calibration")
    path_to_reference = args.get('reference')
    path_to_samples = args.get('path')
    camera_id = args.get('camera_id')
    show_results = args.get('use_ui_for_calibration')
    
    list_of_images = utils.get_filenames_from_folder(path_to_samples)

    (image, sorted_contours, colors) = utils.load_image_with_features(path_to_reference)
    assert(len(sorted_contours) == 30)
    assert(len(colors) == 30), "Could not extract all boxes from reference successfully"

    no_of_valid_images = 0
    imgpoints = []
    objpoints = utils.flatten(sorted_contours, expand=True)
    for image_path in list_of_images:
        (image, sorted_contours, colors) = utils.load_image_with_features(image_path)
        assert(len(sorted_contours) == 30), f"Could not extract all boxes from {image_path} successfully"
        imgpoints.append(utils.flatten(sorted_contours))
        no_of_valid_images += 1

    imgpoints = np.array(imgpoints)
    objpoints = np.array([objpoints for i in range(no_of_valid_images)])
    grayscale_image = utils.convert_to_grayscale(image)

    # ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    #     objpoints, imgpoints, grayscale_image.shape[::-1], None, None)
    # reprojection_error = utils.calculate_reprojection_error(objpoints, imgpoints, mtx, dist, rvecs, tvecs)
    
    # Use normalized values for calibration
    k = (objpoints - np.min(objpoints))/(np.max(objpoints) - np.min(objpoints))
    l = (imgpoints - np.min(imgpoints))/(np.max(imgpoints) - np.min(imgpoints))
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        k, l, grayscale_image.shape[::-1], None, None)
    reprojection_error = utils.calculate_reprojection_error(k, l, mtx, dist, rvecs, tvecs)

    print("RMS error: ", ret)
    print(f"Reprojection Error: {reprojection_error}")

    h,  w = image.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    # Write out camera parameters to yaml file
    camera_parameters = {
        'ret': ret,
        'camera_matrix': np.asarray(mtx).tolist(),
        'dist_coeff': np.asarray(dist).tolist(),
        'rvecs': np.asarray(rvecs).tolist(),
        'tvecs': np.asarray(tvecs).tolist()
    }
    save_camera_parameters(camera_parameters, camera_id)

    if show_results:
        for image_path in list_of_images:
            image = utils.resize(cv2.imread(image_path))
            h,  w = image.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
                mtx, dist, (w, h), 1, (w, h))

            # Option 1: undistort
            # dst = cv2.undistort(image, mtx, dist, None, newcameramtx)

            # Option 2: remap
            mapx, mapy = cv2.initUndistortRectifyMap(
                mtx, dist, None, newcameramtx, (w, h), 5)
            dst = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)

            # crop the image
            x, y, w, h = roi
            dst = dst[y:y+h, x:x+w]
            utils.plot([image, dst])


def save_camera_parameters(camera_parameters, camera_id=""):
    with open(f"data/camera_parameters_{camera_id}.yaml", 'w') as file:
        documents = yaml.dump(camera_parameters, file)
    print(f"Camera parameters saved to: data/camera_parameters_{camera_id}.yaml")

def load_camera_parameters(camera_id):
    with open(f'data/camera_parameters_{camera_id}.yaml') as file:
        camera_parameters = yaml.safe_load(file)
        ret = camera_parameters.get('ret')
        camera_matrix = np.array(camera_parameters.get('camera_matrix'))
        dist_coeff = np.array(camera_parameters.get('dist_coeff'))
        rvecs = np.array(camera_parameters.get('rvecs'))
        tvecs = np.array(camera_parameters.get('tvecs'))
        print(f"Camera parameters loaded from: data/camera_parameters_{camera_id}.yaml")
        return ret, camera_matrix, dist_coeff, rvecs, tvecs




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--camera_id", help="Camera ID - to identify saved camera parameters",
                                default="default")
    args = vars(parser.parse_args())
    camera_id = args.get('camera_id')

    print(f"\n\nLoading camera calibration parameters - Camera ID: {camera_id}\n\n")
    ret, camera_matrix, dist_coeff, rvecs, tvecs = load_camera_parameters(camera_id)
    camera_parameters = {
        'ret': ret,
        'camera_matrix': camera_matrix,
        'dist_coeff': dist_coeff,
        'rvecs': rvecs,
        'tvecs': tvecs
    }
    pprint(camera_parameters)
