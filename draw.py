import cv2 as cv
import numpy as np

# img = cv.imread('images/wallpaper1.jpg')
# cv.imshow('wallpaper1' ,img)

blankImage = np.zeros((500, 500, 3), dtype='uint8')
cv.imshow('blankImage', blankImage)

# paint image
# blankImage[:] = 255, 255, 255
# cv.imshow('newImage', blankImage)

blankImage[100:200, 300:400] = 255, 255, 255
cv.imshow('newImage', blankImage)

# draw a line
cv.line(blankImage, (0, 0), (250, 250), (255, 255, 255), thickness=1)
cv.line(blankImage, (blankImage.shape[1] // 2, blankImage.shape[0] // 2), (499, 499), (255, 255, 255), thickness=1)
cv.imshow('line', blankImage)

# draw a rectangle
cv.rectangle(blankImage, (0, 0), (250, 250), (255, 255, 255), thickness=1)
cv.rectangle(blankImage, (blankImage.shape[1] // 2, blankImage.shape[0] // 2), (499, 499), (255, 255, 255), thickness=cv.FILLED)
cv.imshow('rectangle', blankImage)

# draw a circle
cv.circle(blankImage, (250, 250), 50, (250, 250, 250), thickness=2)
cv.imshow('circle', blankImage)

# write text
cv.putText(blankImage, 'Jose Atlin', (180, 250), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), thickness=2)
cv.imshow('text', blankImage)

cv.waitKey(0)