import cv2

def detectChessboard(templatePath, webcam=False):
  # Load the chessboard template image
  template = cv2.imread(templatePath)

  # Ensure template has correct depth (grayscale) if color image
  if len(template.shape) > 2:
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

  # Capture video from webcam or video file
  cap = cv2.VideoCapture("6.mp4") if webcam else cv2.VideoCapture("6.mp4")

  while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if frame capture successful
    if not ret:
      print("Error: Failed to capture frame!")
      break

    # Convert frame to grayscale for template matching
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Check template size compatibility and resize if needed
    if template.shape[0] > gray.shape[0] or template.shape[1] > gray.shape[1]:
      # Option 1: Resize the template (preferred if feasible)
      template = cv2.resize(template, (gray.shape[1], gray.shape[0]))

      # Option 2: Resize the image (carefully, to avoid distortion)
      # gray = cv2.resize(gray, (int(gray.shape[1]*1.2), int(gray.shape[0]*1.2)))

    # Perform template matching using normalized cross-correlation (NCC)
    res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)

    # Set a threshold to determine the best match
    threshold = 0.8
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # Check if the match value is above the threshold
    if max_val > threshold:
      # Extract the location of the best match
      top_left = max_loc

      # Get the template width and height
      w, h = template.shape[:2]

      # Draw a rectangle around the detected chessboard
      bottom_right = (top_left[0] + w, top_left[1] + h)
      cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
      print("Chessboard detected!")
    else:
      print("No chessboard detected.")

    # Display the resulting frame
    resize = cv2.resize(frame,(640,640))
    cv2.imshow('Chessboard Detection', resize)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  # Release capture resources
  cap.release()
  cv2.destroyAllWindows()

# Path to your chessboard template image
templatePath = "img1.png"  

# Choose webcam or video file
useWebcam = True  # Set to True for webcam, False for video file

detectChessboard(templatePath, webcam=useWebcam)
