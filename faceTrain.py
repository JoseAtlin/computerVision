import os
import numpy as np
import cv2 as cv

DIR = r'C:\Users\JoseAtlin\Desktop\local_repo\openCV\train'
people = [p for p in os.listdir(DIR)]

haarCascade = cv.CascadeClassifier('haarFace.xml')
features = []
labels = []

def train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for image in os.listdir(path):
            imagePath = os.path.join(path, image)

            imageArray = cv.imread(imagePath)
            gray = cv.cvtColor(imageArray, cv.COLOR_BGR2GRAY)

            faceDetect = haarCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

            for (x, y, w, h) in faceDetect:
                faceROI = gray[y:y + h, x:x + w]
                features.append(faceROI)
                labels.append(label)


train()
print(f'No. of features : {len(features)}')
print(f'No. of labels : {len(labels)}')


features = np.array(features, dtype='object')
labels = np.array(labels)
faceRecognizer = cv.face.LBPHFaceRecognizer_create()
faceRecognizer.train(features, labels)

faceRecognizer.save('faceTrained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)