import cv2
import numpy as np

# Open the camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use 0 if it's the default camera
cap.set(3, 800)
cap.set(4, 600)

def calculate_confidence(mask, x, y, r):
    """ Calculate confidence score based on mask coverage. """
    roi = mask[max(0, y - r):y + r, max(0, x - r):x + r]
    if roi.size == 0:
        return 0
    confidence = np.sum(roi > 0) / roi.size  # Percentage of pixels that match
    return round(confidence * 100, 1)  # Convert to percentage

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # ✅ Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # ✅ Define HSV Ranges
    white_lower, white_upper = np.array([0, 0, 120]), np.array([255, 80, 255])
    orange_lower, orange_upper = np.array([5, 80, 80]), np.array([30, 255, 255])

    # ✅ Create Masks
    white_mask = cv2.inRange(hsv, white_lower, white_upper)
    orange_mask = cv2.inRange(hsv, orange_lower, orange_upper)
    combined_mask = cv2.bitwise_or(white_mask, orange_mask)

    # ✅ Convert to Grayscale and Blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # ✅ Detect Circles with Reduced Sensitivity
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1.2, 30,
                               param1=50, param2=40, minRadius=10, maxRadius=40)

    detected_balls = []

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            x, y, r = i[0], i[1], i[2]

            # ✅ Ignore Very Small Circles
            if r < 10:
                continue

            # ✅ Calculate Confidence
            white_conf = calculate_confidence(white_mask, x, y, r)
            orange_conf = calculate_confidence(orange_mask, x, y, r)

            # ✅ Assign Ball Type Based on Higher Confidence
            if orange_conf > white_conf and orange_conf > 60:
                ball_type = "Orange Ball"
                confidence = orange_conf
                color = (0, 140, 255)  # Orange color
            elif white_conf > 60:
                ball_type = "White Ball"
                confidence = white_conf
                color = (255, 255, 255)  # White color
            else:
                continue  # Ignore low-confidence detections

            detected_balls.append((ball_type, x, y, r, color, confidence))

    # ✅ Use Contours to Verify Shape
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        circularity = 4 * np.pi * (area / (perimeter ** 2)) if perimeter > 0 else 0
        if circularity > 0.7:  # Only accept nearly circular shapes
            cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)

    # ✅ Draw Detected Balls and Confidence Scores
    for ball_type, x, y, r, color, confidence in detected_balls:
        cv2.circle(frame, (x, y), r, color, 3)
        label = f"{ball_type} ({confidence}%)"
        cv2.putText(frame, label, (x - 40, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

    # ✅ Show Results
    cv2.imshow("Live Feed", frame)
    cv2.imshow("HSV Mask", combined_mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()