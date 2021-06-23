import cv2 as cv

def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


capture = cv.VideoCapture('videos/dog.mp4')

while True:
    isTrue, frame = capture.read()

    cv.imshow('video', frame)
    cv.imshow('resizedvideo', rescaleFrame(frame, scale=0.5))

    if cv.waitKey(10) and 0xFF == 'd':
        break

capture.release()
cv.destroyAllWindows()