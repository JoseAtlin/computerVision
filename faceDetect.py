import cv2 as cv

img = cv.imread('images/lady.jpg')
cv.imshow('image', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('gray', gray)

facedetect = cv.CascadeClassifier('haarFace.xml')
eyeDetect = cv.CascadeClassifier('haarEye.xml')

faces = facedetect.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
print(f'No. of faces : {len(faces)}')

for (x, y, w, h) in faces:
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
    
    grayROI = gray[y:y + h, x:x + w]
    colorROI = img[y:y + h, x:x + w]
    eyes = eyeDetect.detectMultiScale(grayROI)

    for (ex, ey, ew, eh) in eyes:
        cv.rectangle(colorROI, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

print(f'No. of eyes : {len(eyes)}')
cv.imshow('Detected Face', img)

cv.waitKey(0)