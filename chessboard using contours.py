import cv2

def detect_chessboard(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Perform edge detection using Canny algorithm
    edges = cv2.Canny(blur, 50, 150)
    
    # Find contours in the edge map
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Initialize variables for chessboard detection
    chessboard_contour = None
    max_area = 0
    
    # Loop through the contours to find the largest quadrilateral (the chessboard)
    for contour in contours:
        # Approximate the contour to a polygon
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        
        # Check if the contour has 4 vertices and is convex
        if len(approx) == 4 and cv2.isContourConvex(approx):
            # Calculate the area of the contour
            area = cv2.contourArea(approx)
            
            # Update the chessboard contour if the current contour is larger
            if area > max_area:
                chessboard_contour = approx
                max_area = area
    
    return chessboard_contour

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If frame is not captured, break the loop
    if not ret:
        break
    
    # Detect the chessboard contour
    chessboard_contour = detect_chessboard(frame)

    # Draw the detected chessboard contour on the frame
    if chessboard_contour is not None:
        cv2.drawContours(frame, [chessboard_contour], -1, (0, 255, 0), 2)

    # Display the result
    cv2.imshow('Chessboard Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
