import cv2 as cv

def resizeImage(frame, scale=.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

img = cv.imread('images/wallpaper1.jpg')
img = resizeImage(img, scale=.50)
cv.imshow('wallpaper1', img)

greyScale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('greyScale', greyScale)


#blurring
averageBlur = cv.blur(img, (3, 3), cv.BORDER_DEFAULT)
cv.imshow('averageBlur', averageBlur)

gauss = cv.GaussianBlur(img, (3, 3), 0, cv.BORDER_DEFAULT)
cv.imshow('gaussianBlur', gauss)

medianBlur = cv.medianBlur(img, 3)
cv.imshow('medianBlur', medianBlur)

bilateralBlur = cv.bilateralFilter(img, 10, 50, 50)
cv.imshow('bilateralBlur', bilateralBlur)


canny = cv.Canny(img, 125, 175)
cv.imshow('canny', canny)
blurCanny = cv.Canny(gauss, 125, 175)
cv.imshow('blurCanny', blurCanny)

dilation = cv.dilate(canny, (3, 3), iterations=1)
cv.imshow('dilation', dilation)
blurDilation = cv.dilate(blurCanny, (3, 3), iterations=1)
cv.imshow('blurDilation', blurDilation)

erosion = cv.erode(dilation, (3, 3), iterations=1)
cv.imshow('erosion', erosion)

resize = cv.resize(img, (500, 500), interpolation=cv.INTER_AREA)
cv.imshow('resize', resize)

cropped = img[0:200, 0:400]
cv.imshow('cropped', cropped)

cv.waitKey(0)