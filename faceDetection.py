import cv2
import numpy as np

img = cv2.imread("")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascade+"haarcascade_frontalface_default.xml")
faces = face_cascade.detectMultiScale(gray, 1.3, 5, (30,30) )
