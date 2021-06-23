import cv2 as cv
import numpy as np

img = cv.imread('images/wallpaper1.jpg')
img = cv.resize(img, (400, 400), interpolation=cv.INTER_AREA)
cv.imshow('image', img)

blank = np.zeros((400, 400), dtype='uint8')
cv.imshow('blank', blank)

rectangle = cv.rectangle(blank.copy(), (30, 30), (370, 370), 255, -1)
circle = cv.circle(blank.copy(), (200, 200), 200, 255, -1)

cv.imshow('rectangle', rectangle)
cv.imshow('circle', circle)

bitwiseAND = cv.bitwise_and(rectangle, circle)
cv.imshow('bitwiseAND', bitwiseAND)

bitwiseOR = cv.bitwise_or(rectangle, circle)
cv.imshow('bitwiseOR', bitwiseOR)

bitwiseXOR = cv.bitwise_xor(rectangle, circle)
cv.imshow('bitwiseXOR', bitwiseXOR)

bitwiseNOT = cv.bitwise_not(rectangle)
cv.imshow('bitwiseNOT', bitwiseNOT)

mask = cv.circle(blank, (img.shape[1] // 2, img.shape[0] // 2), 100, 255, -1)
cv.imshow('mask', mask)

masked = cv.bitwise_and(img, img, mask=mask)
cv.imshow('masked', masked)

cv.waitKey(0)