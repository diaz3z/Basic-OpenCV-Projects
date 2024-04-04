import cv2
import numpy as np

img = cv2.imread("img2.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
faces = face_cascade.detectMultiScale(gray,scaleFactor=1.1, minNeighbors=5,minSize=(30,30) )
for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y),(w+x, h+y),(0,255,0),5)
cv2.imshow("dflk",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
