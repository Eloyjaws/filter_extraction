import cv2
import numpy as np
from config import reference, target

# load images
img = cv2.imread(reference)

# convert to LAB
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

# separate B channel
b = lab[:,:,2]

# threshold and invert
thresh = cv2.threshold(b, 192, 255, cv2.THRESH_BINARY)[1]
thresh = 255 - thresh

# apply morphology to clean it up
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)

# get min enclosing circle
# numpy points are (y,x), so need to transpose
points = np.argwhere(morph.transpose()>0)

center, radius = cv2.minEnclosingCircle(points)
print('center:', center, 'radius:', radius)

# draw circle on copy of input
result = img.copy()
cx = int(round(center[0]))
cy = int(round(center[1]))
rr = int(round(radius))
cv2.circle(result, (cx,cy), rr, (255,255,255), 2)

# save output
# cv2.imwrite('out.jpg', result)

# display results
cv2.imshow('thresh',thresh)
cv2.imshow('morph',morph)
cv2.imshow('result',result)
cv2.waitKey(0)
cv2.destroyAllWindows()