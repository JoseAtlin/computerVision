import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread('images/wallpaper2.jpg')
img = cv.resize(img, (1000, 500), interpolation=cv.INTER_AREA)
cv.imshow('image', img)

## Grayscale Histogram
# blank = np.zeros(img.shape[:2], dtype='uint8')
# circle = cv.circle(blank, (img.shape[1] // 2, img.shape[0] // 2), 100, 255, -1)

# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('gray', gray)

# mask = cv.bitwise_and(gray, gray, mask=circle)
# cv.imshow('mask', mask)

# grayHist = cv.calcHist([gray], [0], mask, [256], [0, 256])
# plt.figure()
# plt.title('Greyscale Histogram')
# plt.xlabel('Bins')
# plt.ylabel('No. of pixels')
# plt.plot(grayHist)
# plt.xlim([0, 256])
# plt.show()

## Color Histogram
blank = np.zeros(img.shape[:2], dtype='uint8')
cirle = cv.circle(blank, (img.shape[1] // 2, img.shape[0] // 2), 100, 255, -1)

mask = cv.bitwise_and(img, img, mask=cirle)
cv.imshow('mask', mask)


plt.figure()
plt.title('Color Histogram')
plt.xlabel('Bins')
plt.ylabel('No. of pixels')
colors = ('b', 'g', 'r')
for i, color in enumerate(colors):
    colorHist = cv.calcHist([img], [i], cirle, [256], [0, 256])
    plt.plot(colorHist, color=color)
    plt.xlim([0, 256])
plt.show()

cv.waitKey(0)