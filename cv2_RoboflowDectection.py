from roboflow import Roboflow
import cv2
import random
import threading
import time
from queue import Queue
import numpy as np

# =====================
#  1) ROBOFLOW SETUP
# =====================
rf = Roboflow(api_key="qJTLU5ku2vpBGQUwjBx2")  # Replace with your Roboflow API key
project = rf.workspace("cdio-nczdp").project("detect-and-classify-4")  
model = project.version(6).model  # Ensure this version exists

# Assign random BGR colors for each model-predicted class
class_colors = {}

# =====================
#  2) COLOR-BASED DETECTION HELPERS
# =====================

def calculate_confidence(mask, x, y, r):
    """Calculate a simple confidence score based on how many mask pixels 
       fall within the circle region (x, y, r)."""
    # Make sure we stay within image bounds:
    roi = mask[max(0, y - r):y + r, max(0, x - r):x + r]
    if roi.size == 0:
        return 0
    confidence = (cv2.countNonZero(roi) / roi.size)
    return round(confidence * 100, 1)

def detect_balls_and_walls(frame):
    """
    Runs color-based detection:
      - White & Orange balls using HoughCircles + confidence checks
      - Red walls using contour approximation
    Draws annotations directly on 'frame'. 
    Returns the annotated frame (same reference).
    """

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # --- White & Orange Masks ---
    white_lower, white_upper = np.array([0, 0, 120]), np.array([255, 80, 255])
    orange_lower, orange_upper = np.array([5, 80, 80]), np.array([30, 255, 255])

    white_mask = cv2.inRange(hsv, white_lower, white_upper)
    orange_mask = cv2.inRange(hsv, orange_lower, orange_upper)
    combined_mask = cv2.bitwise_or(white_mask, orange_mask)

    # --- Red Mask (walls) ---
    red_lower1, red_upper1 = np.array([0, 120, 70]), np.array([10, 255, 255])
    red_lower2, red_upper2 = np.array([170, 120, 70]), np.array([180, 255, 255])
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    # Optional morphological close on red_mask to reduce noise/holes
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

    # -----------------------------
    # 1) DETECT BALLS via HoughCircles
    # -----------------------------
    # Convert to grayscale & blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.4,        # Adjust if needed for your camera distance
        minDist=30,
        param1=50,
        param2=35,
        minRadius=5,
        maxRadius=50
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for x, y, r in circles[0, :]:
            # Skip trivial circles
            if r < 5:
                continue

            # Calculate color confidences
            white_conf = calculate_confidence(white_mask, x, y, r)
            orange_conf = calculate_confidence(orange_mask, x, y, r)

            # Decide if White or Orange
            if orange_conf > white_conf and orange_conf > 60:
                ball_type = "Orange Ball"
                color = (0, 140, 255)  # BGR "orange-ish"
                conf_value = orange_conf
            elif white_conf > 60:
                ball_type = "White Ball"
                color = (255, 255, 255)  # BGR White
                conf_value = white_conf
            else:
                continue

            # Draw circle + label
            cv2.circle(frame, (x, y), r, color, 2)
            cv2.putText(frame, f"{ball_type} ({conf_value}%)", 
                        (x - 30, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # -----------------------------
    # 2) DETECT RED WALLS via contours
    # -----------------------------
    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in red_contours:
        area = cv2.contourArea(cnt)
        if area < 2000:
            # skip small noise or small objects
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
        if len(approx) == 4:
            # It's roughly rectangular → label as "Wall"
            cv2.drawContours(frame, [approx], -1, (0, 255, 255), 3)
            x, y, w, h = cv2.boundingRect(approx)
            cv2.putText(frame, "Wall", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    return frame


# =====================
#  3) THREADING SETUP
# =====================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FPS, 30)

frame_queue = Queue(maxsize=1)
output_queue = Queue(maxsize=1)

skip_frames = 3  # process every 3rd frame
frame_count = 0

def capture_frames():
    """Captures frames and puts every Nth (skip_frames) frame into the queue."""
    global frame_count
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize for Roboflow inference + color detection
        frame = cv2.resize(frame, (416, 416))

        frame_count += 1
        if frame_count % skip_frames == 0:
            if not frame_queue.full():
                frame_queue.put(frame)

def draw_coordinate_system(frame):
    """ Overlays a coordinate grid on the frame. """
    h, w, _ = frame.shape

    # Center cross
    cv2.line(frame, (w // 2, 0), (w // 2, h), (255, 255, 255), 1)
    cv2.line(frame, (0, h // 2), (w, h // 2), (255, 255, 255), 1)

    # Grid lines
    step_size = 50
    for x in range(0, w, step_size):
        cv2.line(frame, (x, 0), (x, h), (50, 50, 50), 1)
    for y in range(0, h, step_size):
        cv2.line(frame, (0, y), (w, y), (50, 50, 50), 1)

    return frame

def process_frames():
    """ 
    Gets frames from frame_queue, runs:
        1) Roboflow inference
        2) Color-based detection (white/orange balls, red walls)
    Then draws all results and places final image in output_queue.
    """
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()

            start_time = time.time()

            # 1) Roboflow Inference
            predictions = model.predict(frame, confidence=30, overlap=20).json()

            object_counts = {}
            # Draw Roboflow bounding boxes
            for pred in predictions.get('predictions', []):
                x, y = int(pred['x']), int(pred['y'])
                w, h = int(pred['width']), int(pred['height'])
                label = pred['class']
                confidence = pred['confidence']

                # Count object
                object_counts[label] = object_counts.get(label, 0) + 1

                # Assign color if new class
                if label not in class_colors:
                    class_colors[label] = (
                        random.randint(0, 255),
                        random.randint(0, 255),
                        random.randint(0, 255)
                    )

                color = class_colors[label]
                # Draw bounding box
                cv2.rectangle(
                    frame,
                    (x - w // 2, y - h // 2),
                    (x + w // 2, y + h // 2),
                    color, 2
                )
                cv2.putText(frame, f"{label}: {confidence:.2f}", 
                            (x - w // 2, y - h // 2 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                # Show coordinates
                cv2.putText(frame, f"({x}, {y})",
                            (x + 5, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Display object counts
            y_offset = 30
            for obj, count in object_counts.items():
                cv2.putText(frame, f"{obj}: {count}", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                y_offset += 30

            # 2) Color-Based Detection (Balls & Walls)
            frame = detect_balls_and_walls(frame)

            # 3) Draw coordinate system
            frame = draw_coordinate_system(frame)

            # Put final annotated frame in the output queue
            if not output_queue.full():
                output_queue.put(frame)

            end_time = time.time()
            print(f"Inference Time: {end_time - start_time:.2f}s | Detected by Model: {object_counts}")

def display_frames():
    """ Continuously displays the latest processed frame from output_queue. """
    while True:
        if not output_queue.empty():
            frame = output_queue.get()
            cv2.imshow("Live Object Detection + Color-Based", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

# =====================
#  4) START THREADS
# =====================
cap_thread = threading.Thread(target=capture_frames, daemon=True)
proc_thread = threading.Thread(target=process_frames, daemon=True)
disp_thread = threading.Thread(target=display_frames, daemon=True)

cap_thread.start()
proc_thread.start()
disp_thread.start()

cap_thread.join()
proc_thread.join()
disp_thread.join()

cap.release()
cv2.destroyAllWindows()
