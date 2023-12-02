#!/usr/bin/env python

import numpy as np
import cv2 as cv
import time

# Enter path to image you want to threshold on
img = cv.imread('/home/fizzer/ros_ws/src/controller_repo/src/imitation_learning/training/training_imgs/T_1701476085862_LX_0.0_AZ_0.0.jpeg',cv.IMREAD_COLOR)
img = cv.medianBlur(img,5)

ub = 130
ug = 255
ur = 255
lb = 110
lg = 50
lr = 50
lower_bgr = np.array([lb,lg,lr])
upper_bgr = np.array([ub,ug,ur])

# Threshold the bgr image to get only blue colors
mask = cv.inRange(img, lower_bgr, upper_bgr)
result = cv.bitwise_and(img, img, mask=mask)

window_name = "BGR Calibrator"
cv.namedWindow(window_name)

def nothing(x):
    print("Trackbar value: " + str(x))
    pass

# create trackbars for Upper bgr
cv.createTrackbar('UpperB',window_name,0,255,nothing)
cv.setTrackbarPos('UpperB',window_name, ub)

cv.createTrackbar('UpperG',window_name,0,255,nothing)
cv.setTrackbarPos('UpperG',window_name, ug)

cv.createTrackbar('UpperR',window_name,0,255,nothing)
cv.setTrackbarPos('UpperR',window_name, ur)

# create trackbars for Lower bgr
cv.createTrackbar('LowerB',window_name,0,255,nothing)
cv.setTrackbarPos('LowerB',window_name, lb)

cv.createTrackbar('LowerG',window_name,0,255,nothing)
cv.setTrackbarPos('LowerG',window_name, lg)

cv.createTrackbar('LowerR',window_name,0,255,nothing)
cv.setTrackbarPos('LowerR',window_name, lr)

font = cv.FONT_HERSHEY_SIMPLEX

print("Loaded images")

while(1):
    # Threshold the bgr image to get only blue colors
    mask = cv.inRange(img, lower_bgr, upper_bgr)
    result = cv.bitwise_and(img, img, mask=mask)
    
    cv.putText(mask,'Lower bgr: [' + str(lb) +',' + str(lg) + ',' + str(lr) + ']', (10,30), font, 0.5, (200,255,155), 1, cv.LINE_AA)
    cv.putText(mask,'Upper bgr: [' + str(ub) +',' + str(ug) + ',' + str(ur) + ']', (10,60), font, 0.5, (200,255,155), 1, cv.LINE_AA)

    cv.imshow(window_name,mask)

    k = cv.waitKey(1) & 0xFF
    if k == 27:
        break
    # get current positions of Upper bgr trackbars
    ub = cv.getTrackbarPos('UpperB',window_name)
    ug = cv.getTrackbarPos('UpperG',window_name)
    ur = cv.getTrackbarPos('UpperR',window_name)
    upper_blue = np.array([ub,ug,ur])
    # get current positions of Lower HSCV trackbars
    lb = cv.getTrackbarPos('LowerB',window_name)
    lg = cv.getTrackbarPos('LowerG',window_name)
    lr = cv.getTrackbarPos('LowerR',window_name)
    upper_bgr = np.array([ub,ug,ur])
    lower_bgr = np.array([lb,lg,lr])

    time.sleep(.1)

cv.destroyAllWindows()