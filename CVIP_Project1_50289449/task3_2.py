# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 10:20:38 2018

@author: shrey
"""

import numpy as np
import cv2
 
image = cv2.imread('C:\\Users\\shrey\\Downloads\\proj1_cse573.zip\\task3\\pos_5.jpg')
template = cv2.imread('C:\\Users\\shrey\\Downloads\\proj1_cse573.zip\\task3\\template.png')
 
# resize images
image = cv2.resize(image, (0,0), fx=0.5, fy=0.5) 
template = cv2.resize(template, (0,0), fx=0.5, fy=0.5) 
 
# Convert to grayscale
imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
 
# Find template
result = cv2.matchTemplate(imageGray,templateGray, cv2.TM_CCOEFF)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
top_left = max_loc
h,w = templateGray.shape
bottom_right = (top_left[0] + w, top_left[1] + h)
cv2.rectangle(image,top_left, bottom_right,(0,0,255),4)
 
# Show result
cv2.imshow("Template", template)
cv2.imshow("Result", image)
 
cv2.moveWindow("Template", 10, 50);
cv2.moveWindow("Result", 150, 50);
 
cv2.waitKey(0)