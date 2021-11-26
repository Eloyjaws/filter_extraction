import cv2 as cv2
import cv2 as cv
import numpy as np 
from config import reference, target
from utils import plot

img1 = cv2.imread(reference)
img2 = cv2.imread(target)

img1 = cv.medianBlur(img1, 21)
img2 = cv.medianBlur(img2, 21)

def harris_corner_detection(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,3,3,0.04)
    dst = cv2.dilate(dst,None)
    image[dst>0.01*dst.max()]=[0,0,255]

    kernel = np.ones((5,5),np.uint8)
    gradient = cv.morphologyEx(dst, cv.MORPH_GRADIENT, kernel)
    # dst = np.expand_dims(dst, axis=2)
    # run_split_visualizer(image, dst, "Harris")
    return gradient
    # run_visualizer(gradient, gradient, "Harris")

def harris_corner_detection_v1(img):

    cv.namedWindow('Harris Corner Detection Test', cv.WINDOW_NORMAL)

    def f(x=None):
        return

    cv.createTrackbar('Harris Window Size', 'Harris Corner Detection Test', 5, 25, f)
    cv.createTrackbar('Harris Parameter', 'Harris Corner Detection Test', 1, 100, f)
    cv.createTrackbar('Sobel Aperture', 'Harris Corner Detection Test', 1, 14, f)
    cv.createTrackbar('Detection Threshold', 'Harris Corner Detection Test', 1, 100, f)

    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    img_bak = img

    while True:
        img = img_bak.copy()

        window_size = cv.getTrackbarPos('Harris Window Size', 'Harris Corner Detection Test')
        harris_parameter = cv.getTrackbarPos('Harris Parameter', 'Harris Corner Detection Test')
        sobel_aperture = cv.getTrackbarPos('Sobel Aperture', 'Harris Corner Detection Test')
        threshold = cv.getTrackbarPos('Detection Threshold', 'Harris Corner Detection Test')

        sobel_aperture = sobel_aperture * 2 + 1

        if window_size <= 0:
            window_size = 1

        dst = cv.cornerHarris(gray, window_size, sobel_aperture, harris_parameter/100)

        # Result is dilated for marking the corners, not important
        dst = cv.dilate(dst,None)

        # Threshold for an optimal value, it may vary depending on the image.
        img[dst > threshold/100 * dst.max()] = [0, 0, 255]

        dst_show = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
        dst_show = (255*dst_show).astype(np.uint8)

        cv.imshow('Harris Corner Detection Test', np.hstack((img, dst_show)))

        if cv.waitKey(0) & 0xFF == 27:
            break
        cv.destroyAllWindows()


# harris_corner_detection_v1(img1)
grad1 = harris_corner_detection(img1)
grad2 = harris_corner_detection(img2)
plot([grad1, grad2])