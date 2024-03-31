import cv2
import numpy as np


img = cv2.imread("img2.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray, 1.3,5)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y,))
