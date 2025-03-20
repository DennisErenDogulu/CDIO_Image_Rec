import cv2
import numpy as np

# Open the camera
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # Use 0 if it's the default camera

# Set resolution (optional)
cap.set(3, 800)
cap.set(4, 600)

def get_dominant_hsv(frame, x, y, radius):
    """ Get the dominant HSV color in a region of interest (ROI). """
    roi = frame[max(0, y - radius):y + radius, max(0, x - radius):x + radius]
    if roi.size == 0: 
        return np.array([15, 200, 200])  # Default orange
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    avg_color = np.mean(roi_hsv.reshape(-1, 3), axis=0)
    return avg_color.astype(int)  # Returns (H, S, V)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # ✅ Apply CLAHE for better contrast (fixes lighting issues)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # ✅ Fix White Ball Detection (Expanded HSV Range)
    white_lower = np.array([0, 0, 140])  # Looser range to capture more white
    white_upper = np.array([255, 80, 255])

    # ✅ Use adaptive HSV filtering for orange ball
    orange_lower = np.array([0, 100, 100])
    orange_upper = np.array([30, 255, 255])

    # Create masks
    white_mask = cv2.inRange(hsv, white_lower, white_upper)
    orange_mask = cv2.inRange(hsv, orange_lower, orange_upper)
    mask = cv2.bitwise_or(white_mask, orange_mask)

    # ✅ Apply Morphological Filtering to Reduce Noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # ✅ Use Edge Detection to Improve Ball Separation
    edges = cv2.Canny(frame, 50, 150)  # Detect edges
    mask = cv2.bitwise_or(mask, edges)  # Combine edges with mask

    # Convert to grayscale and blur for circle detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # ✅ Detect Circles with Refined Parameters
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1.2, 30,
                               param1=50, param2=50, minRadius=20, maxRadius=50)

    detected_balls = []  # List to store detected balls with type

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            x, y, r = i[0], i[1], i[2]

            # Get the dominant color of the detected ball
            avg_h, avg_s, avg_v = get_dominant_hsv(frame, x, y, r)

            # ✅ Dynamically adjust HSV for orange ball detection
            orange_lower = np.array([max(0, avg_h - 10), 100, 100])
            orange_upper = np.array([min(30, avg_h + 10), 255, 255])

            # Check which mask the ball belongs to
            if orange_mask[y, x] > 0:
                ball_type = "Orange Ball"
                color = (0, 140, 255)  # Orange color
            elif white_mask[y, x] > 0:
                ball_type = "White Ball"
                color = (255, 255, 255)  # White color
            else:
                continue  # Ignore if not detected in any mask

            # Store the ball's type and position
            detected_balls.append((ball_type, x, y, r, color))

    # ✅ Sort balls so Orange Balls are processed first
    detected_balls.sort(key=lambda b: 0 if b[0] == "Orange Ball" else 1)

    # Draw detected balls and labels
    for ball_type, x, y, r, color in detected_balls:
        cv2.circle(frame, (x, y), r, color, 3)  # Draw the circle
        cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)  # Draw center
        cv2.putText(frame, ball_type, (x - 40, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2, cv2.LINE_AA)  # Label the ball

    # Show results
    cv2.imshow("Live Feed", frame)
    cv2.imshow("HSV Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
