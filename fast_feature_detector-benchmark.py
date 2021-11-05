import numpy as np
import cv2
import cv2 as cv
from config import reference, target
import matplotlib.pyplot as plt
import utils
from utils import run_visualizer, run_split_visualizer

reference_card = cv.imread(reference)
input_image = cv.imread(target)

# Make image black and white - apply a low threshold
grayImage = utils.convert_to_grayscale(reference_card)
(thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 192, 255, cv2.THRESH_BINARY_INV)
# (thresh, blackAndWhiteImage) = utils.apply_threshold(grayImage, 127, 255, cv2.THRESH_BINARY_INV)

# contours, hierarchy  = cv2.findContours(blackAndWhiteImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# blackAndWhiteImage = cv2.drawContours(blackAndWhiteImage, [max(contours, key = cv2.contourArea)], -1, (0,255,75), -1)

# blackAndWhiteImage = utils.apply_opening(blackAndWhiteImage, kernel_size=40, iterations=4)
# blackAndWhiteImage = utils.apply_closing(blackAndWhiteImage, kernel_size=40, iterations=4)


cnts = cv2.findContours(blackAndWhiteImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]


# https://stackoverflow.com/a/57193144/5837671

points = []

# min_area = 18000
# max_area = 25000
min_area = 100
max_area = 3500
# max_area = 1500
image_number = 0
for c in cnts:
    area = cv2.contourArea(c)
    # if area > 500:
    #     print(area)
    if area > min_area and area < max_area:
        x,y,w,h = cv2.boundingRect(c)

        points.append((x,y))
        points.append((x+w,y))
        points.append((x,y+h))
        points.append((x+w,y+h))

        ROI = reference_card[y:y+h, x:x+w]
        # cv2.imwrite('ROI_{}.png'.format(image_number), ROI)
        cv2.rectangle(reference_card, (x, y), (x + w, y + h), (255,0,0), 2)
        image_number += 1
# 

print(points)

cv2.imshow('b/w', blackAndWhiteImage)
cv2.imshow('contours', reference_card)
# cv2.imshow('close', close)
# cv2.imshow('mask', mask)
cv2.waitKey()
# 


blackAndWhiteImage = cv2.cvtColor(blackAndWhiteImage, cv2.COLOR_BGR2RGB) 
# plt.imshow(blackAndWhiteImage)
# plt.show(block=True)


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
imWithCircles = drawKeyPts(reference_card.copy(),kp,(0,0,255),2)
# reference_card_with_keypoints = cv.drawKeypoints(blackAndWhiteImage, kp, np.array([]), color=(255,0,255))

# kp = fast.detect(input_image,None)
# input_image_with_keypoints = cv.drawKeypoints(input_image, kp, np.array([]), color=(255,0,255))
# im2WithCircles = drawKeyPts(input_image.copy(),kp,(0,0,255),10)

# # Print all default params
print("Threshold: ", fast.getThreshold())
print("nonmaxSuppression: ", fast.getNonmaxSuppression())
print("neighborhood: ", fast.getType())
print("Total Keypoints with nonmaxSuppression: ", len(kp))
# run_split_visualizer(blackAndWhiteImage, imWithCircles, 'Fast Feature detector')

# run_visualizer(reference_card_with_keypoints, input_image_with_keypoints, 'Fast Feature detector')