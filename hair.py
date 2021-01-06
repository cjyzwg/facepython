import cv2
import numpy as np
img = cv2.imread('./hair/hair.png')

cv2.imshow('img',img)
img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

cv2.imshow('img_3',img_hsv)
cv2.waitKey(50)

rows, cols, n = img_hsv.shape
lower_yellow=np.array([12,43,46])
upper_yellow=np.array([34,255,255])

lower_blue=np.array([35,43,46])
upper_blue=np.array([124,255,255])
mask_blue=cv2.inRange(img_hsv,lower_blue,upper_blue)
mask_yellow=cv2.inRange(img_hsv,lower_yellow,upper_yellow)

image_mask=img_hsv.copy()

for row in range(rows): #遍历每一行
    for col in range(cols): #遍历每一列
        if mask_yellow[row,col] == 255 or mask_blue[row,col] == 255:
            image_mask[row, col, 0] = image_mask[row, col, 0] + 50
            image_mask[row, col, 1] = image_mask[row, col, 1] + 50
        else:''
yellow=cv2.bitwise_and(img,img,mask=mask_yellow)
img_2 = cv2.cvtColor(image_mask,cv2.COLOR_HSV2BGR)
cv2.imshow('img_2',img_2)
cv2.waitKey(50)
cv2.imwrite("./hair/hair_out.png", img_2)