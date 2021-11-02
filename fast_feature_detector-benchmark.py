import numpy as np
import cv2
import cv2 as cv
from config import reference, target, run_visualizer, run_split_visualizer
import matplotlib.pyplot as plt

img1 = cv.imread(reference)
# img1 = cv.imread("filters/test.jpeg")
img2 = cv.imread(target)

# WHITE = [255,255,255]
# img1 = cv2.copyMakeBorder(img1.copy(),32,32,32,32,cv2.BORDER_CONSTANT,value=WHITE)

# Too much noise - apply median blur
# Median blur helps eliminate salt and pepper noise - and also preserves edges
# img1 = cv.medianBlur(img1, 7)
# img2 = cv.medianBlur(img2, 7)

# Make image black and white - apply a low threshold
grayImage = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
(thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY_INV)
# (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 192, 255, cv2.THRESH_BINARY_INV)

# contours, hierarchy  = cv2.findContours(blackAndWhiteImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# blackAndWhiteImage = cv2.drawContours(blackAndWhiteImage, [max(contours, key = cv2.contourArea)], -1, (0,255,75), -1)

# kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(40,40))
# blackAndWhiteImage = cv2.morphologyEx(blackAndWhiteImage, cv2.MORPH_OPEN, kernel, iterations=4)
# blackAndWhiteImage = cv2.morphologyEx(blackAndWhiteImage, cv2.MORPH_CLOSE, kernel, iterations=4)


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

        ROI = img1[y:y+h, x:x+w]
        # cv2.imwrite('ROI_{}.png'.format(image_number), ROI)
        cv2.rectangle(img1, (x, y), (x + w, y + h), (255,0,0), 2)
        image_number += 1
# 

print(points)

cv2.imshow('b/w', blackAndWhiteImage)
cv2.imshow('contours', img1)
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
# img1_with_keypoints = cv.drawKeypoints(blackAndWhiteImage, kp, np.array([]), color=(255,0,255))
imWithCircles = drawKeyPts(img1.copy(),kp,(0,0,255),2)

# kp = fast.detect(img2,None)
# img2_with_keypoints = cv.drawKeypoints(img2, kp, np.array([]), color=(255,0,255))
# im2WithCircles = drawKeyPts(img2.copy(),kp,(0,0,255),10)

# # Print all default params
print("Threshold: ", fast.getThreshold())
print("nonmaxSuppression: ", fast.getNonmaxSuppression())
print("neighborhood: ", fast.getType())
print("Total Keypoints with nonmaxSuppression: ", len(kp))
# run_split_visualizer(blackAndWhiteImage, imWithCircles, 'Fast Feature detector')

# run_visualizer(img1_with_keypoints, img2_with_keypoints, 'Fast Feature detector')