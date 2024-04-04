import cv2
import numpy as np

img = cv2.imread("img2.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
faces = face_cascade.detectMultiScale(image= gray, scaleFactor= 1.1, minNeighbors= 3, minSize=(30,30))
for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 3)
cv2.imshow("face", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
