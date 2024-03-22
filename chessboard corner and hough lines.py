import cv2
import numpy as np

def detect_chessboard(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform corner detection
    corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)

    # Convert corners to integers
    corners = np.int0(corners)

    # Perform Hough Line Transform
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)

    # Draw lines on the frame
    if lines is not None:
        for rho, theta in lines[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Draw corners on the frame
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

    return frame

# Open the webcam
cap = cv2.VideoCapture("5.mp4")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If frame is not captured, break the loop
    if not ret:
        break

    # Detect chessboard
    resize_window = cv2.resize(frame,(640,640))
    chessboard_frame = detect_chessboard(resize_window)

    # Display the result
    cv2.imshow('Chessboard Detection', chessboard_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
