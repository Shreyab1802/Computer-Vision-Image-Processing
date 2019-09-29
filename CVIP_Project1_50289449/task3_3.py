# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 15:33:20 2018

@author: shrey
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 16:54:47 2018

@author: 12144
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('C:\\Users\\shrey\\Downloads\\proj1_cse573\\task3\\pos_3.jpg')
blur_img = cv2.GaussianBlur(img,(3,3),0)

img_gray= cv2.cvtColor(blur_img, cv2.COLOR_BGR2GRAY)

img_input=cv2.Laplacian(img_gray,cv2.CV_64F)

cv2.namedWindow('Temp image', cv2.WINDOW_NORMAL)


cv2.imshow('Temp image',img_input)
cv2.waitKey(0)
cv2.destroyAllWindows()


template = cv2.imread('C:\\Users\\shrey\\Downloads\\proj1_cse573\\task3\\template.png',0)
print(template)
w, h = template.shape[::-1]

img_temp=cv2.Laplacian(template,cv2.CV_64F)

cv2.namedWindow('Temp image', cv2.WINDOW_NORMAL)
cv2.imshow('Temp image',img_temp)
cv2.waitKey(0)
cv2.destroyAllWindows()


# All the 6 methods for comparison in a list
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR','cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    
for meth in methods:
    img = img_input.copy()
    method = eval(meth)
    
        # Apply template Matching
    res = cv2.matchTemplate(img_input,img_temp,method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    
    cv2.rectangle(img,top_left, bottom_right, 255, 2)
    
    plt.subplot(121),plt.imshow(res,cmap = 'gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img,cmap = 'gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)
    
    plt.show()

    cv2.namedWindow('Detected Point', cv2.WINDOW_NORMAL)
    cv2.imshow('Detected Point',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()    