import cv2 as cv
import numpy as np

img = cv.imread('images/wallpaper1.jpg')
img = cv.resize(img, (1000, 500))
cv.imshow('wallpaper1', img)

grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('grayscale', grayscale)

blur = cv.GaussianBlur(img, (3, 3), cv.BORDER_DEFAULT)
canny = cv.Canny(blur, 125, 175)
cv.imshow('canny', canny)


### Thresholding
## Simple Threshold
ret, threshold = cv.threshold(grayscale, 150, 255, cv.THRESH_BINARY)
cv.imshow('threshold', threshold)

ret, threshINV = cv.threshold(grayscale, 150, 255, cv.THRESH_BINARY_INV)
cv.imshow('threshINV', threshINV)

## Adaptive Threshold
adaptiveThreshold = cv.adaptiveThreshold(grayscale, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 3)
cv.imshow('Adaptive Threshold', adaptiveThreshold)


contours, heirarchies = cv.findContours(threshold, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
print(len(contours))

blank = np.zeros(img.shape, dtype='uint8')
cv.imshow('blank', blank)

cv.drawContours(blank, contours, -1, (0, 0, 255), thickness=1)
cv.imshow('contours', blank)

cv.waitKey(0)