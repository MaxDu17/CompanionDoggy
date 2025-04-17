import cv2
import numpy as np

def nothing(x):
    pass

# Create a window for the trackbars
cv2.namedWindow("Trackbars")

# Create trackbars for lower and upper HSV
cv2.createTrackbar("Lower H", "Trackbars", 90, 179, nothing)
cv2.createTrackbar("Lower S", "Trackbars", 40, 255, nothing)
cv2.createTrackbar("Lower V", "Trackbars", 40, 255, nothing)
cv2.createTrackbar("Upper H", "Trackbars", 110, 179, nothing)
cv2.createTrackbar("Upper S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("Upper V", "Trackbars", 255, 255, nothing)

# Open the webcam
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get current positions of all trackbars
    l_h = cv2.getTrackbarPos("Lower H", "Trackbars")
    l_s = cv2.getTrackbarPos("Lower S", "Trackbars")
    l_v = cv2.getTrackbarPos("Lower V", "Trackbars")
    u_h = cv2.getTrackbarPos("Upper H", "Trackbars")
    u_s = cv2.getTrackbarPos("Upper S", "Trackbars")
    u_v = cv2.getTrackbarPos("Upper V", "Trackbars")

    # Set the HSV range from trackbars
    lower_green = np.array([l_h, l_s, l_v])
    upper_green = np.array([u_h, u_s, u_v])

    # Create a mask using the HSV range
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If contours found
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        if area > 500:
            cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(largest_contour)
            cx, cy = x + w // 2, y + h // 2
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
            cv2.putText(frame, f"Line Center: ({cx}, {cy})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Show windows
    cv2.imshow("Green Line Tracker", frame)
    cv2.imshow("Mask", mask)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()