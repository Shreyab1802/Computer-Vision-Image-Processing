# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 10:14:23 2018

@author: shreya
"""


import numpy as np
import cv2


img = cv2.imread('C:\\Users\\shrey\\Downloads\\proj1_cse573\\task3\\pos_5.jpg')

blur_img = cv2.GaussianBlur(img,(3,3),0)

cv2.namedWindow('Temp image', cv2.WINDOW_NORMAL)
#cv2.imshow('Temp image',blur_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

img_gray= cv2.cvtColor(blur_img, cv2.COLOR_BGR2GRAY)

img_input=cv2.Laplacian(img_gray,cv2.CV_32F)

template = cv2.imread('C:\\Users\\shrey\\Downloads\\proj1_cse573\\task3\\template.png')
img = cv2.resize(img, (0,0), fx=0.5, fy=0.5) 
template = cv2.resize(template, (0,0), fx=0.5, fy=0.5)

w= template.shape[0]
h= template.shape[1]

blur_templ= cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
img_temp=cv2.Laplacian(blur_templ,cv2.CV_32F)

cv2.namedWindow('Temp image', cv2.WINDOW_NORMAL)
cv2.imshow('Temp image',img_input)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.namedWindow('Temp blur', cv2.WINDOW_NORMAL)
cv2.imshow('Temp blur',img_temp)
cv2.waitKey(0)
cv2.destroyAllWindows()

res=cv2.matchTemplate(img_input,img_temp,cv2.TM_CCORR_NORMED)
print(res)
cv2.namedWindow('Result image', cv2.WINDOW_NORMAL)
cv2.imshow('Result image',res)
cv2.waitKey(0)
cv2.destroyAllWindows()

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
cv2.rectangle(img,top_left, bottom_right,(255,0,0),4)
 
cv2.namedWindow('detected', cv2.WINDOW_NORMAL)
cv2.imshow('detected',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

