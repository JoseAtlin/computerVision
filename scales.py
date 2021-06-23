import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('images/wallpaper1.jpg')
img = cv.resize(img, (1000, 500), interpolation=cv.INTER_AREA)
cv.imshow('image', img)
# plt.imshow(img)
# plt.show()

blank = np.zeros(img.shape[:2], dtype='uint8')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('gray', gray)

hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
cv.imshow('HSV', hsv)

LAB = cv.cvtColor(img, cv.COLOR_BGR2LAB)
cv.imshow('LAB', LAB)

b, g, r = cv.split(img)
blue = cv.merge([b, blank, blank])
green = cv.merge([blank, g, blank])
red = cv.merge([blank, blank, r])

cv.imshow('blue', blue)
cv.imshow('green', green)
cv.imshow('red', red)

cv.imshow('b', b)
cv.imshow('g', g)
cv.imshow('r', r)

print(img.shape)
print(b.shape)
print(g.shape)
print(r.shape)

merge = cv.merge([b, g, r])
cv.imshow('merge', merge)

cv.waitKey(0)