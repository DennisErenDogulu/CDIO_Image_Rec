import cv2
import random
import threading
import time
import logging
from queue import Queue
from roboflow import Roboflow

# ✅ Roboflow Setup
rf = Roboflow(api_key="qJTLU5ku2vpBGQUwjBx2")
project = rf.workspace("cdio-nczdp").project("cdio-golfbot2025")
model = project.version(9).model

# ✅ Assign colors for different classes
class_colors = {}

# ✅ Video Capture Setup
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FPS, 30)

frame_queue = Queue(maxsize=1)
output_queue = Queue(maxsize=1)
frame_count = 0
skip_frames = 5  # Tune as needed

stop_event = threading.Event()

def capture_frames():
    global frame_count
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            logging.warning("Failed to read frame.")
            break

        frame = cv2.resize(frame, (416, 416))
        frame_count += 1

        if frame_count % skip_frames == 0 and not frame_queue.full():
            frame_queue.put(frame)

def draw_coordinate_system(frame):
    h, w, _ = frame.shape
    cv2.line(frame, (w // 2, 0), (w // 2, h), (255, 255, 255), 1)
    cv2.line(frame, (0, h // 2), (w, h // 2), (255, 255, 255), 1)

    step_size = 50
    for x in range(0, w, step_size):
        cv2.line(frame, (x, 0), (x, h), (50, 50, 50), 1)
    for y in range(0, h, step_size):
        cv2.line(frame, (0, y), (w, y), (50, 50, 50), 1)

    return frame

def process_frames():
    while not stop_event.is_set():
        if not frame_queue.empty():
            frame = frame_queue.get()
            start_time = time.time()

            try:
                result = model.predict(frame, confidence=60, overlap=20).json()
            except Exception as e:
                logging.error(f"Model prediction error: {e}")
                continue

            object_counts = {}
            for pred in result.get('predictions', []):
                x, y = int(pred['x']), int(pred['y'])
                w, h = int(pred['width']), int(pred['height'])
                label = pred['class']
                confidence = pred['confidence']

                object_counts[label] = object_counts.get(label, 0) + 1

                if label not in class_colors:
                    class_colors[label] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

                color = class_colors[label]
                cv2.rectangle(frame, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), color, 2)
                cv2.putText(frame, f"{label}: {confidence:.2f}", (x - w // 2, y - h // 2 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(frame, f"({x}, {y})", (x + 5, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # ✅ Draw coordinate system
            frame = draw_coordinate_system(frame)

            if not output_queue.full():
                output_queue.put(frame)

def display_frames():
    while not stop_event.is_set():
        if not output_queue.empty():
            frame = output_queue.get()
            cv2.imshow("Live Object Detection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                stop_event.set()
# ✅ Thread Setup
cap_thread = threading.Thread(target=capture_frames, daemon=True)
proc_thread = threading.Thread(target=process_frames, daemon=True)
disp_thread = threading.Thread(target=display_frames)

cap_thread.start()
proc_thread.start()
disp_thread.start()

disp_thread.join()
stop_event.set()
cap_thread.join()
proc_thread.join()

cap.release()
cv2.destroyAllWindows()
