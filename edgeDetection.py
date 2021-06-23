import cv2 as cv
import numpy as np

img = cv.imread('images/wallpaper2.jpg')
img = cv.resize(img, (1000, 500), interpolation=cv.INTER_AREA)
cv.imshow('image', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('gray', gray)

# Laplacian
laplacian = cv.Laplacian(gray, cv.CV_64F)
laplacian = np.uint8(np.absolute(laplacian))
cv.imshow('laplacian', laplacian)

# Sobel
sobelX = cv.Sobel(gray, cv.CV_64F, 1, 0)
sobelY = cv.Sobel(gray, cv.CV_64F, 0, 1)
combinedSobel = cv.bitwise_or(sobelX, sobelY)

cv.imshow('sobelX', sobelX)
cv.imshow('sobelY', sobelY)
cv.imshow('combinedSobel', combinedSobel)

canny = cv.Canny(gray, 150, 175)
cv.imshow('canny',canny)

cv.waitKey(0)