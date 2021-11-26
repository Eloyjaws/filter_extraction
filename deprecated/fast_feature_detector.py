import numpy as np
import cv2
import cv2 as cv
from config import reference, target
from utils import plot
import matplotlib.pyplot as plt

img1 = cv.imread("filters/test.jpg")
img2 = cv.imread(reference)
# img2 = cv.imread(target)

WHITE = [255,255,255]
img1 = cv2.copyMakeBorder(img1.copy(),32,32,32,32,cv2.BORDER_CONSTANT,value=WHITE)

# Too much noise - apply median blur
# Median blur helps eliminate salt and pepper noise - and also preserves edges
# img1 = cv.medianBlur(img1, 7)
# img2 = cv.medianBlur(img2, 7)

# Make image black and white - apply a low threshold
grayImage = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
(thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 192, 255, cv2.THRESH_BINARY_INV)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(20,20))
blackAndWhiteImage = cv2.morphologyEx(blackAndWhiteImage, cv2.MORPH_OPEN, kernel, iterations=4)

contours, hierarchy  = cv2.findContours(blackAndWhiteImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(contours, hierarchy)
# blackAndWhiteImage = cv2.drawContours(blackAndWhiteImage, [max(contours, key = cv2.contourArea)], -1, (0,255,75), -1)

blackAndWhiteImage = cv2.cvtColor(blackAndWhiteImage, cv2.COLOR_BGR2RGB) 

plt.imshow(blackAndWhiteImage)
plt.show(block=True)


fast = cv.FastFeatureDetector_create()
fast.setNonmaxSuppression(False)
fast.setThreshold(32)

import matplotlib.pyplot as plt

def drawKeyPts(im,keyp,col,th):
    for curKey in keyp:
        x=np.int(curKey.pt[0])
        y=np.int(curKey.pt[1])
        size = np.int(curKey.size)
        cv.circle(im,(x,y),size, col,thickness=th, lineType=8, shift=0) 
    plt.imshow(im)    
    return im    


kp = fast.detect(blackAndWhiteImage,None)
# img1_with_keypoints = cv.drawKeypoints(blackAndWhiteImage, kp, np.array([]), color=(255,0,255))
imWithCircles = drawKeyPts(img1.copy(),kp,(0,0,255),15)

# kp = fast.detect(img2,None)
# img2_with_keypoints = cv.drawKeypoints(img2, kp, np.array([]), color=(255,0,255))
# im2WithCircles = drawKeyPts(img2.copy(),kp,(0,0,255),10)

# # Print all default params
print("Threshold: ", fast.getThreshold())
print("nonmaxSuppression: ", fast.getNonmaxSuppression())
print("neighborhood: ", fast.getType())
print("Total Keypoints with nonmaxSuppression: ", len(kp))
plot([blackAndWhiteImage, imWithCircles])

# run_visualizer(img1_with_keypoints, img2_with_keypoints, 'Fast Feature detector')