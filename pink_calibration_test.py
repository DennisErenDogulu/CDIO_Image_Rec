#!/usr/bin/env python3
"""
Pink Marker Calibration Tool

This standalone script helps you calibrate the HSV color range for the pink direction marker.
Use this to find the correct color values before running the main ball collection system.
"""

import cv2
import numpy as np

# Initial pink HSV range (you can adjust these starting values)
PINK_LOWER = np.array([145, 50, 50])
PINK_UPPER = np.array([175, 255, 255])

def calibrate_pink_marker():
    """Interactive pink marker color calibration"""
    
    # Initialize camera
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # Use camera index 1 for USB camera
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Create trackbars window
    cv2.namedWindow("Pink Marker Calibration")
    cv2.namedWindow("Controls")
    
    # Create trackbars for HSV values
    cv2.createTrackbar("H Min", "Controls", PINK_LOWER[0], 179, lambda x: None)
    cv2.createTrackbar("S Min", "Controls", PINK_LOWER[1], 255, lambda x: None)
    cv2.createTrackbar("V Min", "Controls", PINK_LOWER[2], 255, lambda x: None)
    cv2.createTrackbar("H Max", "Controls", PINK_UPPER[0], 179, lambda x: None)
    cv2.createTrackbar("S Max", "Controls", PINK_UPPER[1], 255, lambda x: None)
    cv2.createTrackbar("V Max", "Controls", PINK_UPPER[2], 255, lambda x: None)
    
    print("Pink marker calibration started.")
    print("Adjust the trackbars to isolate your pink marker in the right panel.")
    print("The goal is to have ONLY your pink marker show up as white in the mask.")
    print("Press 'c' to confirm and print values, 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Get trackbar values
        h_min = cv2.getTrackbarPos("H Min", "Controls")
        s_min = cv2.getTrackbarPos("S Min", "Controls")
        v_min = cv2.getTrackbarPos("V Min", "Controls")
        h_max = cv2.getTrackbarPos("H Max", "Controls")
        s_max = cv2.getTrackbarPos("S Max", "Controls")
        v_max = cv2.getTrackbarPos("V Max", "Controls")
        
        # Update pink ranges
        pink_lower = np.array([h_min, s_min, v_min])
        pink_upper = np.array([h_max, s_max, v_max])
        
        # Convert to HSV and create mask
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        pink_mask = cv2.inRange(hsv, pink_lower, pink_upper)
        
        # Find pink contours for visualization
        pink_contours, _ = cv2.findContours(pink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw detected pink areas on original frame
        result_frame = frame.copy()
        detected_count = 0
        largest_area = 0
        
        if pink_contours:
            largest_pink = max(pink_contours, key=cv2.contourArea)
            largest_area = cv2.contourArea(largest_pink)
            
            if largest_area > 50:
                M = cv2.moments(largest_pink)
                if M["m00"] != 0:
                    pink_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                    cv2.circle(result_frame, pink_center, 8, (255, 192, 203), -1)
                    cv2.circle(result_frame, pink_center, 10, (255, 255, 255), 2)
                    cv2.putText(result_frame, "PINK DETECTED", 
                              (pink_center[0] + 15, pink_center[1]), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    detected_count = 1
            
            # Draw all pink contours
            cv2.drawContours(result_frame, pink_contours, -1, (255, 192, 203), 2)
            detected_count = len([c for c in pink_contours if cv2.contourArea(c) > 50])
        
        # Show current HSV values and detection info on frame
        cv2.putText(result_frame, f"Pink HSV: [{h_min},{s_min},{v_min}] - [{h_max},{s_max},{v_max}]", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(result_frame, f"Detected objects: {detected_count}, Largest area: {int(largest_area)}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(result_frame, "Press 'c' to confirm, 'q' to quit", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Add status indicator
        if detected_count == 1 and largest_area > 200:
            cv2.putText(result_frame, "GOOD DETECTION!", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        elif detected_count > 1:
            cv2.putText(result_frame, "TOO MANY OBJECTS - Narrow range", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif detected_count == 0:
            cv2.putText(result_frame, "NO DETECTION - Widen range", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Create a side-by-side view
        pink_mask_colored = cv2.cvtColor(pink_mask, cv2.COLOR_GRAY2BGR)
        combined = np.hstack((result_frame, pink_mask_colored))
        
        # Resize if too large
        height, width = combined.shape[:2]
        if width > 1600:
            scale = 1600 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            combined = cv2.resize(combined, (new_width, new_height))
        
        cv2.imshow("Pink Marker Calibration", combined)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            # Confirm and print values
            print("\n" + "="*50)
            print("PINK MARKER CALIBRATION RESULTS:")
            print("="*50)
            print(f"PINK_LOWER = np.array([{h_min}, {s_min}, {v_min}])")
            print(f"PINK_UPPER = np.array([{h_max}, {s_max}, {v_max}])")
            print("\nCopy these values to your continuous_collector.py file!")
            print("Update lines 31-32 with these new values.")
            print("="*50)
            break
        elif key == ord('q'):
            print("Pink marker calibration cancelled")
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Pink Marker Calibration Tool")
    print("="*40)
    print("This tool helps you find the correct HSV range for your pink direction marker.")
    print("Make sure your pink marker is visible in the camera view.")
    print()
    calibrate_pink_marker() 