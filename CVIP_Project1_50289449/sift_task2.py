import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
img= cv2.imread('task2.jpg',cv2.IMREAD_GRAYSCALE)
img_out=img.copy()


first_sigma_values=[(1/(np.sqrt(2))),1,np.sqrt(2),2,2*(np.sqrt(2))]
second_sigma_values=[np.sqrt(2),2,2*(np.sqrt(2)),4,4*(np.sqrt(2))]
third_sigma_values=[2*np.sqrt(2),4,4*(np.sqrt(2)),8,8*(np.sqrt(2))]
     

def operation(img, kernel): 
    img_h=img.shape[0]
    img_w=img.shape[1]

    kernel_h=kernel.shape[0]
    kernel_w=kernel.shape[1]

    h=kernel_h//2
    w=kernel_w//2

    image_conv=np.zeros(img.shape)
    for i in range(h,img_h-h):
        for j in range(w,img_w-w):
            sum=0

            for m in range(kernel_h):
                for n in range(kernel_w):
                    sum=sum+ kernel[m][n]*img[i-h+m][j-w+n]
            image_conv[i][j]=sum
    return image_conv        

    


def gkern(l,sigma):
    """
    creates gaussian kernel with side length l and a sigma of sig
    """

    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))

    return kernel / np.sum(kernel)

# First Octave
 
g11=gkern(7,first_sigma_values[0])
n11=operation(img,g11)
cv2.imshow('octave11',img_out)
cv2.waitKey(0)

g12=gkern(7,first_sigma_values[1])
n12=operation(img,g12)
cv2.imshow('octave12',img_out)
cv2.waitKey(0)

g13=gkern(7,first_sigma_values[2])
n13=operation(img,g13)
cv2.imshow('octave13',img_out)
cv2.waitKey(0)

g14=gkern(7,first_sigma_values[3])
n14=operation(img,g14)
cv2.imshow('octave14',img_out)
cv2.waitKey(0)

g15=gkern(7,first_sigma_values[4])
n15=operation(img,g15)
cv2.imshow('octave15',img_out)
cv2.waitKey(0)
cv2.destroyAllWindows()

#second octave
img=img[0::2,0::2]
print('Second Ocatve size')
print(img.shape)

g21=gkern(7,second_sigma_values[0])
n21=operation(img,g21)
cv2.imshow('octave21',img_out)
cv2.waitKey(0)

g22=gkern(7,second_sigma_values[1])
n22=operation(img,g22)
cv2.imshow('octave22',img_out)
cv2.waitKey(0)

g23=gkern(7,second_sigma_values[2])
n23=operation(img,g23)
cv2.imshow('octave23',img_out)
cv2.waitKey(0)

g24=gkern(7,second_sigma_values[3])
n24=operation(img,g24)
cv2.imshow('octave24',img_out)
cv2.waitKey(0)

g25=gkern(7,second_sigma_values[4])
n25=operation(img,g25)
cv2.imshow('octave25',img_out)
cv2.waitKey(0)

# #third octave
img=img[0::2,0::2]
print('Third Ocatve size')
print(img.shape)
g31=gkern(7,third_sigma_values[0])
n31=operation(img,g31)
cv2.imshow('octave31',img_out)
cv2.waitKey(0)

g32=gkern(7,third_sigma_values[1])
n32=operation(img,g32)
cv2.imshow('octave32',img_out)
cv2.waitKey(0)

g33=gkern(7,third_sigma_values[2])
n33=operation(img,g33)
cv2.imshow('octave33',img_out)
cv2.waitKey(0)

g34=gkern(7,third_sigma_values[3])
n34=operation(img,g34)
cv2.imshow('octave34',img_out)
cv2.waitKey(0)

g35=gkern(7,third_sigma_values[4])
n35=operation(img,g35)
cv2.imshow('octave35',img_out)
cv2.waitKey(0)

# #fourth octave
img=img[0::2,0::2]

g41=gkern(7,fourth_sigma_values[0])
n41=operation(img,g41)
cv2.imshow('octave41',img_out)
cv2.waitKey(0)

g42=gkern(7,fourth_sigma_values[1])
n42=operation(img,g42)
cv2.imshow('octave42',img_out)
cv2.waitKey(0)

g43=gkern(7,fourth_sigma_values[2])
n43=operation(img,g43)
cv2.imshow('octave43',img_out)
cv2.waitKey(0)

g44=gkern(7,fourth_sigma_values[3])
n44=operation(img,g44)
cv2.imshow('octave44',img_out)
cv2.waitKey(0)

g45=gkern(7,fourth_sigma_values[4])
n45=operation(img,g45)
cv2.imshow('octave45',img_out)
cv2.waitKey(0)

# calculating D0G's for second:

dog21=n21-n22
cv2.imshow('DoG21',dog21)
cv2.waitKey(0)

dog22=n22-n23
print('Second octave DoG Size')
print(dog22.shape)
cv2.imshow('DoG22',dog22)
cv2.waitKey(0)

dog23=n23-n24
cv2.imshow('DoG23',dog23)
cv2.waitKey(0)

dog24=n24-n25
cv2.imshow('DoG24',dog24)
cv2.waitKey(0)

# calculating D0G's for third:

dog31=n31-n32
cv2.imshow('DoG31',dog31)
cv2.waitKey(0)

dog32=n32-n33
print('Third octave DoG size')
print(dog32.shape)
cv2.imshow('DoG32',dog32)
cv2.waitKey(0)

dog33=n33-n34
cv2.imshow('DoG33',dog33)
cv2.waitKey(0)

dog34=n34-n35
cv2.imshow('DoG34',dog34)
cv2.waitKey(0)















      

