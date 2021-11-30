# Use pyyaml to save and load camera parameters
import os
import sys
import yaml
import numpy as np
import cv2

import utils


def run_calibration(args):
    print("Running camera calibration")
    path_to_reference = args.get('reference')
    path_to_samples = args.get('path')
    show_results = False
    list_of_images = utils.get_filenames_from_folder(path_to_samples)

    (image, sorted_contours, colors) = utils.load_image_with_features(path_to_reference)
    assert(len(sorted_contours) == 30)
    assert(len(colors) == 30)
    objpoints = utils.flatten(sorted_contours, expand=True)
    imgpoints = []

    no_of_valid_images = 0
    for image_path in list_of_images:
        (image, sorted_contours, colors) = utils.load_image_with_features(image_path)
        if(sorted_contours.shape[0] != 30):
            continue
        imgpoints.append(utils.flatten(sorted_contours))
        no_of_valid_images += 1

    # imgpoints = np.array(imgpoints)
    # objpoints = np.array([objpoints for i in range(no_of_valid_images)])
    imgpoints = np.array(imgpoints).reshape((1, -1, 2))
    objpoints = np.array([objpoints for i in range(
        no_of_valid_images)]).reshape((1, -1, 3))

    # print(objpoints.shape, imgpoints.shape)
    # print(objpoints[0], imgpoints[0])

    grayscale_image = utils.convert_to_grayscale(image)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, grayscale_image.shape[::-1], None, None)

    print(ret, "\n\n", mtx, "\n\n", dist, "\n\n", rvecs, "\n\n", tvecs)

    camera_parameters = {
        'ret': ret,
        'camera_matrix': np.asarray(mtx).tolist(),
        'dist_coeff': np.asarray(dist).tolist(),
        'rvecs': np.asarray(rvecs).tolist(),
        'tvecs': np.asarray(tvecs).tolist()
    }
    save_camera_parameters(camera_parameters)

    if not show_results:
        return

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

def load_camera_parameters(camera_id):
    with open(f'data/camera_parameters_{camera_id}.yaml') as file:
        camera_parameters = yaml.safe_load(file)
        ret = camera_parameters.get('ret')
        camera_matrix = np.array(camera_parameters.get('camera_matrix'))
        dist_coeff = np.array(camera_parameters.get('dist_coeff'))
        rvecs = np.array(camera_parameters.get('rvecs'))
        tvecs = np.array(camera_parameters.get('tvecs'))
        return ret, camera_matrix, dist_coeff, rvecs, tvecs



if __name__ == "__main__":
    camera_id = ""
    print(f"Running Calibration Routine - Camera ID: {camera_id}")
    ret, camera_matrix, dist_coeff, rvecs, tvecs = load_camera_parameters(camera_id)
    print(ret, camera_matrix, dist_coeff, rvecs, tvecs)
