from roboflow import Roboflow
import cv2
import random
import threading
import time
from queue import Queue

rf = Roboflow(api_key="qJTLU5ku2vpBGQUwjBx2")   # ← your API key
project = rf.workspace("cdio-n5rcb").project("cdio_4")
model   = project.version(8).model              # ← use the version that’s deployed

# ✅ Assign colors for different classes
class_colors = {}

# ✅ Start video capture
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffering
cap.set(cv2.CAP_PROP_FPS, 30)  # Adjust FPS for smoother performance

frame_queue = Queue(maxsize=1)  # Stores latest frame
output_queue = Queue(maxsize=1)  # Stores processed frame

skip_frames = 3  # ✅ Process every 3rd frame for better FPS
frame_count = 0

def capture_frames():
    """ Captures frames and skips unnecessary ones for higher FPS. """
    global frame_count
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (416, 416))  # ✅ Smaller frame for faster processing

        frame_count += 1
        if frame_count % skip_frames == 0:
            if not frame_queue.full():
                frame_queue.put(frame)

def draw_coordinate_system(frame):
    """ Overlays a coordinate system on the frame. """
    h, w, _ = frame.shape

    # Draw center lines (X and Y axis)
    cv2.line(frame, (w // 2, 0), (w // 2, h), (255, 255, 255), 1)  # Vertical (Y-axis)
    cv2.line(frame, (0, h // 2), (w, h // 2), (255, 255, 255), 1)  # Horizontal (X-axis)

    # Draw grid lines
    step_size = 50  # Adjust for denser/sparser grid
    for x in range(0, w, step_size):
        cv2.line(frame, (x, 0), (x, h), (50, 50, 50), 1)
    for y in range(0, h, step_size):
        cv2.line(frame, (0, y), (w, y), (50, 50, 50), 1)

    return frame

def process_frames():
    """ Runs object detection in parallel & counts objects. """
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            
            start_time = time.time()  # ✅ Start time for measuring speed
            
            # ✅ Run inference (Optimize confidence & overlap if needed)
            predictions = model.predict(frame, confidence=30, overlap=20).json()

            # ✅ Count instances of each class
            object_counts = {}

            # ✅ Process predictions (draw bounding boxes)
            for pred in predictions.get('predictions', []):
                x, y, w, h = int(pred['x']), int(pred['y']), int(pred['width']), int(pred['height'])
                label = pred['class']
                confidence = pred['confidence']

                # ✅ Count detected objects by class
                if label not in object_counts:
                    object_counts[label] = 1
                else:
                    object_counts[label] += 1

                # ✅ Assign a unique color to each class
                if label not in class_colors:
                    class_colors[label] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

                color = class_colors[label]

                # Draw bounding box with class-specific color
                cv2.rectangle(frame, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), color, 2)
                cv2.putText(frame, f"{label}: {confidence:.2f}", (x - w // 2, y - h // 2 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # ✅ Show object coordinates
                cv2.putText(frame, f"({x}, {y})", (x + 5, y - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # ✅ Display the object count on the frame
            y_offset = 30
            for obj, count in object_counts.items():
                cv2.putText(frame, f"{obj}: {count}", (10, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                y_offset += 30

            # ✅ Draw coordinate system
            frame = draw_coordinate_system(frame)

            if not output_queue.full():
                output_queue.put(frame)

            end_time = time.time()  # ✅ End time for measuring speed
            print(f"Inference Time: {end_time - start_time:.2f} sec | Detected Objects: {object_counts}")

def display_frames():
    """ Displays frames in a separate thread for smoother performance. """
    while True:
        if not output_queue.empty():
            frame = output_queue.get()
            cv2.imshow("Live Object Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

# ✅ Run capture, processing, and display in parallel
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