import numpy as np
import cv2
from config import reference, target, run_visualizer


img1 = cv2.imread(reference)
img2 = cv2.imread(target)

def shi_tomasi(img1, img2):
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray,100,0.01,10)
    corners = np.int0(corners) 

    for i in corners:
        x,y = i.ravel()
        cv2.circle(img1,(x,y),25,(0, 0, 255),-1) 

    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray,100,0.01,10)
    corners = np.int0(corners) 

    for i in corners:
        x,y = i.ravel()
        cv2.circle(img2,(x,y),25,(0, 0, 255),-1) 

    run_visualizer(img1, img2, 'Shi-Tomasi')

shi_tomasi(img1, img2)