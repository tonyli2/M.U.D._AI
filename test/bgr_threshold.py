#!/usr/bin/env python

import numpy as np
import cv2 as cv
import time

# Enter path to image you want to threshold on
img = cv.imread('/home/fizzer/ros_ws/src/controller_repo/src/imitation_learning/training/training_imgs/T_1701476085862_LX_0.0_AZ_0.0.jpeg',cv.IMREAD_COLOR)
bgr = cv.medianBlur(img,5)

ub = 130
ug = 255
ur = 255
lb = 110
lg = 50
lr = 50
lower_bgr = np.array([lb,lg,lr])
upper_bgr = np.array([ub,ug,ur])

# Threshold the bgr image to get only blue colors
mask = cv.inRange(bgr, lower_bgr, upper_bgr)
result = cv.bitwise_and(bgr, bgr, mask=mask)

window_name = "BGR Calibrator"
cv.namedWindow(window_name)

def nothing(x):
    print("Trackbar value: " + str(x))
    pass

# create trackbars for Upper bgr
cv.createTrackbar('UpperH',window_name,0,255,nothing)
cv.setTrackbarPos('UpperH',window_name, ub)

cv.createTrackbar('UpperS',window_name,0,255,nothing)
cv.setTrackbarPos('UpperS',window_name, ug)

cv.createTrackbar('UpperV',window_name,0,255,nothing)
cv.setTrackbarPos('UpperV',window_name, ur)

# create trackbars for Lower bgr
cv.createTrackbar('LowerH',window_name,0,255,nothing)
cv.setTrackbarPos('LowerH',window_name, lb)

cv.createTrackbar('LowerS',window_name,0,255,nothing)
cv.setTrackbarPos('LowerS',window_name, lg)

cv.createTrackbar('LowerV',window_name,0,255,nothing)
cv.setTrackbarPos('LowerV',window_name, lr)

font = cv.FONT_HERSHEY_SIMPLEX

print("Loaded images")

while(1):
    # Threshold the bgr image to get only blue colors
    mask = cv.inRange(bgr, lower_bgr, upper_bgr)
    result = cv.bitwise_and(bgr, bgr, mask=mask)

    cv.putText(result,'Lower bgr: [' + str(lb) +',' + str(lg) + ',' + str(lr) + ']', (10,30), font, 0.5, (200,255,155), 1, cv.LINE_AA)
    cv.putText(result,'Upper bgr: [' + str(ub) +',' + str(ug) + ',' + str(ur) + ']', (10,60), font, 0.5, (200,255,155), 1, cv.LINE_AA)

    cv.imshow(window_name,img)

    k = cv.waitKey(1) & 0xFF
    if k == 27:
        break
    # get current positions of Upper bgr trackbars
    ub = cv.getTrackbarPos('UpperB',window_name)
    ug = cv.getTrackbarPos('UpperG',window_name)
    ur = cv.getTrackbarPos('UpperR',window_name)
    upper_blue = np.array([ub,ug,ur])
    # get current positions of Lower HSCV trackbars
    lb = cv.getTrackbarPos('LowerHB',window_name)
    lg = cv.getTrackbarPos('LowerSG',window_name)
    lr = cv.getTrackbarPos('LowerVR',window_name)
    upper_bgr = np.array([ub,ug,ur])
    lower_bgr = np.array([lb,lg,lr])

    time.sleep(.1)

cv.destroyAllWindows()