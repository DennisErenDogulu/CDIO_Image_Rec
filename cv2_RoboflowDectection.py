import cv2
import threading
import random
from inference import InferencePipeline

stop_event = threading.Event()
class_colors = {}

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

def my_sink(result, video_frame):
    # Get the image from the result
    if "output_image" in result:
        np_frame = result["output_image"].numpy_image
    elif "original_image" in result:
        np_frame = result["original_image"].numpy_image
    else:
        np_frame = video_frame.numpy_image

    frame = draw_coordinate_system(np_frame.copy())

    # Get predictions and make sure they are usable
    predictions = result.get("predictions", [])
    if isinstance(predictions, list):
        for pred in predictions:
            # DEBUG: Uncomment if you want to see raw output
            # print("Raw prediction:", pred)

            # Unwrap if tuple
            if isinstance(pred, tuple):
                pred = pred[0]

            if isinstance(pred, dict):
                label = pred.get("class", "unknown")
                confidence = pred.get("confidence", 0.0) * 100  # Convert to percent
                x = pred.get("x", 0)
                y = pred.get("y", 0)
                w_box = pred.get("width", 0)
                h_box = pred.get("height", 0)

                if label not in class_colors:
                    class_colors[label] = (
                        random.randint(0, 255),
                        random.randint(0, 255),
                        random.randint(0, 255)
                    )
                color = class_colors[label]

                top_left = (int(x - w_box / 2), int(y - h_box / 2))
                bottom_right = (int(x + w_box / 2), int(y + h_box / 2))

                # Draw bounding box
                cv2.rectangle(frame, top_left, bottom_right, color, 2)

                # Draw label + confidence below box to avoid offscreen
                label_text = f"{label}: {confidence:.1f}%"
                text_position = (top_left[0], top_left[1] - 10)
                cv2.putText(frame, label_text, text_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Show the frame
    cv2.imshow("Live Inference", frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        pipeline.stop()
        stop_event.set()

# Initialize the Roboflow pipeline
pipeline = InferencePipeline.init_with_workflow(
    api_key="qJTLU5ku2vpBGQUwjBx2",
    workspace_name="cdio-nczdp",
    workflow_id="detect-and-classify-2",
    video_reference=0,
    max_fps=30,
    on_prediction=my_sink
)

pipeline.start()
stop_event.wait()
pipeline.join()
cv2.destroyAllWindows()
print("Done.")
print("Predictions:", predictions)
