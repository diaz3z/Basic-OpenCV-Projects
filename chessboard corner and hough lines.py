import cv2
import numpy as np

def detect_chessboard(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform corner detection
    corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)

    # Convert corners to integers
    corners = np.int0(corners)

    return corners

def transform_chessboard(frame, corners):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # If corners are detected, perform perspective transform
    if corners is not None:
        # Ensure we have at least 4 corners
        if len(corners) >= 4:
            # Order the corners
            corners = sorted(corners, key=lambda x: x[0][1])

            # Calculate the width and height of the chessboard
            width = max(corners[7][0][0] - corners[0][0][0], corners[8][0][0] - corners[1][0][0])
            height = max(corners[3][0][1] - corners[0][0][1], corners[7][0][1] - corners[4][0][1])

            # Define the destination points for the perspective transform
            dst_points = np.float32([[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]])

            # Calculate the perspective transform matrix
            src_points = np.float32([corners[0][0], corners[1][0], corners[4][0], corners[5][0]])
            matrix = cv2.getPerspectiveTransform(src_points, dst_points)

            # Apply the perspective transform to the frame
            warped = cv2.warpPerspective(frame, matrix, (width, height))

            # Draw lines on the warped image
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
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
                    cv2.line(warped, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Draw corners on the warped image
            for corner in corners:
                x, y = corner.ravel()
                cv2.circle(warped, (x, y), 3, (0, 255, 0), -1)

            return warped

    # If no corners are detected or less than 4 corners are found, return None
    return None

# Open the webcam
cap = cv2.VideoCapture("6.mp4")

# Create a new window for detected corners
cv2.namedWindow('Detected Corners', cv2.WINDOW_NORMAL)

# Create a new window for the warped chessboard
cv2.namedWindow('Warped Chessboard', cv2.WINDOW_NORMAL)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If frame is not captured, break the loop
    if not ret:
        break

    # Detect chessboard corners
    resize_window = cv2.resize(frame, (640, 640))
    corners = detect_chessboard(resize_window)

    # Transform the detected chessboard
    transformed_frame = transform_chessboard(resize_window, corners)

    # Display the detected corners
    if corners is not None:
        corners_frame = resize_window.copy()
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(corners_frame, (x, y), 3, (0, 255, 0), -1)
        cv2.imshow('Detected Corners', corners_frame)

    # Display the warped chessboard
    if transformed_frame is not None:
        cv2.imshow('Warped Chessboard', transformed_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
