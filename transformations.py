import cv2 as cv
import numpy as np

def translate(frame, x, y):
    transMatrix = np.float32([[1, 0, x], [0, 1, y]])
    dimensions = (frame.shape[1], frame.shape[0])
    return cv.warpAffine(frame, transMatrix, dimensions)

def rotate(frame, angle, rotpoint=None):
    height, width = frame.shape[0:2]

    if rotpoint == None:
        rotpoint = (width // 2, height // 2)

    rotMatrix = cv.getRotationMatrix2D(rotpoint, angle, 1.0)
    dimensions = (width, height)

    return cv.warpAffine(frame, rotMatrix, dimensions)

img = cv.imread('images/wallpaper1.jpg')
img = cv.resize(img, (1000, 500), interpolation=cv.INTER_AREA)
cv.imshow('image', img)

translatedImage = translate(img, -100, 100)
cv.imshow('translatedImage', translatedImage)

rotatedImage = rotate(img, 45)
cv.imshow('rotatedImage', rotatedImage)

flip = cv.flip(img, 1)
cv.imshow('flip', flip)

cv.waitKey(0)
