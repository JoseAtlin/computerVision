from faceTrain import train
import os
import numpy as np
import cv2 as cv

DIR = r'C:\Users\JoseAtlin\Desktop\local_repo\openCV\validate'
people = [p for p in os.listdir(DIR)]

features = np.load('features.npy', allow_pickle=True)
labels = np.load('labels.npy', allow_pickle=True)

faceRecognizer = cv.face.LBPHFaceRecognizer_create()
faceRecognizer.read('faceTrained.yml')

image = cv.imread(r'C:\Users\JoseAtlin\Desktop\local_repo\openCV\validate\Neymar jr\Neymar.jpg')
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

haarCascade = cv.CascadeClassifier('haarFace.xml')
faceDetect = haarCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
for (x, y, w, h) in faceDetect:
    faceROI = gray[y:y + h, x:x + w]

    label, confidence = faceRecognizer.predict(faceROI)
    print(f'Label = {people[label]} with confidence of {confidence}')

    cv.putText(image, str(people[label]), (20, 20), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), thickness=1)
    cv.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), thickness=1)

cv.imshow('detected Face', image)
cv.waitKey(0)