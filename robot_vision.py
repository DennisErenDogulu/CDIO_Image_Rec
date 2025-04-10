#!/usr/bin/env pybricks-micropython
import cv2
import numpy as np
from pybricks.hubs import EV3Brick
from pybricks.ev3devices import Motor
from pybricks.parameters import Port
from pybricks.robotics import DriveBase
from pybricks.tools import wait
from time import sleep

# Initialize EV3 and motors
ev3 = EV3Brick()
left_motor = Motor(Port.B)
right_motor = Motor(Port.C)
collection_motor = Motor(Port.A)
robot = DriveBase(left_motor, right_motor, wheel_diameter=55.5, axle_track=104)

# Initialize camera
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # Use 0 if it's the default camera
cap.set(3, 800)
cap.set(4, 600)

# Robot functions
def collect_ball():
    ev3.speaker.beep()
    collection_motor.run_time(500, 1000)
    sleep(1)

def drop_ball():
    ev3.speaker.beep()
    collection_motor.run_time(-500, 3000)

# Vision functions
def calculate_confidence(mask, x, y, r):
    """ Calculate confidence score based on mask coverage. """
    roi = mask[max(0, y - r):y + r, max(0, x - r):x + r]
    if roi.size == 0:
        return 0
    confidence = np.sum(roi > 0) / roi.size
    return round(confidence * 100, 1)

def detect_balls(frame):
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define HSV Ranges
    white_lower, white_upper = np.array([0, 0, 120]), np.array([255, 80, 255])
    orange_lower, orange_upper = np.array([5, 80, 80]), np.array([30, 255, 255])
    
    # Create Masks
    white_mask = cv2.inRange(hsv, white_lower, white_upper)
    orange_mask = cv2.inRange(hsv, orange_lower, orange_upper)
    combined_mask = cv2.bitwise_or(white_mask, orange_mask)
    
    # Convert to Grayscale and Blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Detect Circles
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1.2, 30,
                              param1=50, param2=40, minRadius=10, maxRadius=40)
    
    detected_balls = []
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            x, y, r = i[0], i[1], i[2]
            
            # Ignore Very Small Circles
            if r < 10:
                continue
                
            # Calculate Confidence
            white_conf = calculate_confidence(white_mask, x, y, r)
            orange_conf = calculate_confidence(orange_mask, x, y, r)
            
            # Assign Ball Type Based on Higher Confidence
            if orange_conf > white_conf and orange_conf > 60:
                ball_type = "Orange Ball"
                confidence = orange_conf
                color = (0, 140, 255)
            elif white_conf > 60:
                ball_type = "White Ball"
                confidence = white_conf
                color = (255, 255, 255)
            else:
                continue
                
            detected_balls.append((ball_type, x, y, r, color, confidence))
            
            # Draw circles on the frame
            cv2.circle(frame, (x, y), r, color, 3)
            label = f"{ball_type} ({confidence}%)"
            cv2.putText(frame, label, (x - 40, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)
    
    # Show Results
    cv2.imshow("Live Feed", frame)
    cv2.imshow("Combined Mask", combined_mask)
    
    return detected_balls, frame

# Main robot control with vision
def vision_based_collection():
    collected_balls = 0
    max_balls = 3  # Set maximum number of balls to collect
    
    while collected_balls < max_balls:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Detect balls in the frame
        balls, annotated_frame = detect_balls(frame)
        
        if balls:
            # Sort balls by size (largest first) to prioritize closer balls
            balls.sort(key=lambda x: x[3], reverse=True)
            
            # Get the closest/largest ball
            ball_type, x, y, r, color, confidence = balls[0]
            
            # Calculate position relative to center
            center_x = frame.shape[1] // 2
            offset_x = x - center_x
            
            # Decide how to move based on ball position
            if abs(offset_x) < 50:  # Ball is centered
                ev3.speaker.beep()
                # Move forward toward the ball
                robot.straight(100)
                # If the ball is close enough (large enough in frame)
                if r > 30:
                    collect_ball()
                    collected_balls += 1
                    ev3.speaker.say(f"Collected {ball_type}")
                    # Step back after collection
                    robot.straight(-50)
            elif offset_x < 0:  # Ball is to the left
                robot.turn(-5)  # Turn left slightly
            else:  # Ball is to the right
                robot.turn(5)  # Turn right slightly
        else:
            # No balls detected, search by rotating
            robot.turn(10)
            
        # Check for exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Return to drop-off point
    ev3.speaker.say("Returning to drop off point")
    # This is a simplified return - in reality, you'd need more sophisticated 
    # navigation to return to the starting point
    robot.straight(-500)
    drop_ball()
    ev3.speaker.say("Task complete")

# Run the program
try:
    vision_based_collection()
finally:
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    robot.stop() 