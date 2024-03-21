import cv2
import numpy as np

cap = cv2.VideoCapture(1)

while True:
    ret, img = cap.read()
    cv2.imshow("ksdajflk", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release
cv2.destroyAllWindows