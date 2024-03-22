import cv2
import numpy as np






def detectChessboardCorners(videoSource=0):
    # Create a video capture object
    cap = cv2.VideoCapture("6.mp4")

    while True:
        # Capture a frame from the video
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture frame!")
            break

        # Convert frame to grayscale for corner detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply corner detection (e.g., Harris corner detection)
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)

        # Draw detected corners on the frame
        if corners is not None:
            corners = np.int0(corners)
            for corner in corners:
                x, y = corner.ravel()
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        # Display the frame with detected corners
        resize = cv2.resize(frame,(640,640))
        cv2.imshow('Chessboard Corner Detection', resize)

        # Exit loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release capture resources
    cap.release()
    cv2.destroyAllWindows()

def drawWarpGrid(corners, image):
  """
  This function takes the detected chessboard corners and the original image,
  warps the image to show an 8x8 square grid, and displays it in a new window.
  """

  # Assuming a standard 8x8 chessboard
  rows, cols = 8, 8

  # Extract corner coordinates from all corners
  corner_points = corners.reshape(-1, 2)

  # Re-order corners based on a chessboard layout (e.g., top-left to bottom-right)
  # You might need to adjust the indexing based on your corner detection order
  ordered_corners = corner_points[[0, 7, 14, 21, 28, 35, 42, 49]]
  for i in range(1, rows):
    offset = i * 8
    ordered_corners = np.vstack((ordered_corners, corner_points[offset:offset+8]))

  # Destination points for the warped image (an 8x8 grid)
  dst_points = np.zeros((rows * cols, 2), dtype=np.float32)
  dst_width = image.shape[1] / cols
  dst_height = image.shape[0] / rows
  for row in range(rows):
    for col in range(cols):
      dst_points[row * cols + col] = [col * dst_width, row * dst_height]

  # Perform perspective transform
  homography, mask = cv2.findHomography(ordered_corners, dst_points)
  warped_image = cv2.warpPerspective(image, homography, (image.shape[1], image.shape[0]))

  # Display the warped image with the grid
  cv2.imshow('Warped Grid', warped_image)
  cv2.waitKey(0)  # Wait for key press to close the window

detectChessboardCorners(videoSource=0)  # Use webcam by default
