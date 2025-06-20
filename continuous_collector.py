#!/usr/bin/env python3
"""
Continuous Ball Collection Client

This script implements a continuous ball collection strategy:
1. Detect balls using local YOLOv8 model
2. Track robot position using colored markers
3. Plan and execute paths to collect balls
4. Return to goal area to deliver balls
"""

import cv2
import math
import json
import socket
import logging
import numpy as np
from typing import List, Tuple, Optional
from ultralytics import YOLO
import time

# Configuration
EV3_IP = "172.20.10.6"
EV3_PORT = 12345

# Model configuration
MODEL_PATH = "weights(2).pt"  # Using local weights
CONFIDENCE_THRESHOLD = 0.5
MODEL_IMAGE_SIZE = 640  # YOLOv8 default size

# Color detection ranges (HSV)
GREEN_LOWER = np.array([35, 50, 50])   # Robot base marker
GREEN_UPPER = np.array([85, 255, 255])
PINK_LOWER = np.array([145, 50, 50])   # Direction marker
PINK_UPPER = np.array([175, 255, 255])

# Field dimensions (cm)
FIELD_WIDTH_CM = 180
FIELD_HEIGHT_CM = 120

# Robot configuration
ROBOT_WIDTH = 15   # cm
ROBOT_LENGTH = 20  # cm
ROBOT_START_X = 20  # cm from left edge
ROBOT_START_Y = 20  # cm from bottom edge
ROBOT_START_HEADING = 0  # degrees (0 = facing east)

# Collection parameters
MAX_BALLS_PER_TRIP = 3      # Maximum balls to collect before delivery
COLLECTION_DISTANCE_CM = 20  # Distance to move forward when collecting
APPROACH_DISTANCE_CM = 30    # Distance to keep from ball for approach
WALL_SAFETY_MARGIN = 15     # cm to keep away from walls

# Ignored area (center obstacle)
IGNORED_AREA = {
    "x_min": 50, "x_max": 100,
    "y_min": 50, "y_max": 100
}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BallCollector:
    def __init__(self):
        # Initialize YOLO model
        try:
            self.model = YOLO(MODEL_PATH)
            logger.info("Successfully loaded YOLO model from %s", MODEL_PATH)
        except Exception as e:
            logger.error("Failed to load YOLO model: %s", e)
            raise

        # Initialize camera
        self.cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # Use camera index 1 for USB camera
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera")

        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # Robot state
        self.robot_pos = (ROBOT_START_X, ROBOT_START_Y)
        self.robot_heading = ROBOT_START_HEADING
        
        # Calibration
        self.calibration_points = []
        self.homography_matrix = None

    def detect_markers(self, frame):
        """Detect green base and pink direction markers"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Detect green base marker
        green_mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
        green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Detect pink direction marker
        pink_mask = cv2.inRange(hsv, PINK_LOWER, PINK_UPPER)
        pink_contours, _ = cv2.findContours(pink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find largest green contour
        green_center = None
        if green_contours:
            largest_green = max(green_contours, key=cv2.contourArea)
            if cv2.contourArea(largest_green) > 100:
                M = cv2.moments(largest_green)
                if M["m00"] != 0:
                    green_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        
        # Find largest pink contour
        pink_center = None
        if pink_contours:
            largest_pink = max(pink_contours, key=cv2.contourArea)
            if cv2.contourArea(largest_pink) > 50:
                M = cv2.moments(largest_pink)
                if M["m00"] != 0:
                    pink_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        
        return green_center, pink_center

    def update_robot_position(self, frame):
        """Update robot position and heading based on visual markers"""
        green_center, pink_center = self.detect_markers(frame)
        
        if green_center and pink_center and self.homography_matrix is not None:
            # Convert green center (robot base) to cm coordinates
            green_px = np.array([[[float(green_center[0]), float(green_center[1])]]], dtype="float32")
            green_cm = cv2.perspectiveTransform(green_px, np.linalg.inv(self.homography_matrix))[0][0]
            
            # Convert pink center (direction marker) to cm coordinates
            pink_px = np.array([[[float(pink_center[0]), float(pink_center[1])]]], dtype="float32")
            pink_cm = cv2.perspectiveTransform(pink_px, np.linalg.inv(self.homography_matrix))[0][0]
            
            # Update robot position and heading
            self.robot_pos = (green_cm[0], green_cm[1])
            dx = pink_cm[0] - green_cm[0]
            dy = pink_cm[1] - green_cm[1]
            self.robot_heading = math.degrees(math.atan2(dy, dx))
            
            return True
        return False

    def detect_balls(self) -> List[Tuple[float, float, str]]:
        """Detect balls in current camera view, return list of (x_cm, y_cm, label)"""
        ret, frame = self.cap.read()
        if not ret:
            return []

        # Run YOLOv8 inference
        results = self.model(frame, conf=CONFIDENCE_THRESHOLD)[0]
        balls = []

        # Process detections
        for det in results.boxes.data.tolist():
            x, y, x2, y2, conf, cls = det
            
            # Calculate center point
            center_x = (x + x2) / 2
            center_y = (y + y2) / 2
            
            # Convert to cm using homography if available
            if self.homography_matrix is not None:
                pt_px = np.array([[[center_x, center_y]]], dtype="float32")
                pt_cm = cv2.perspectiveTransform(pt_px, np.linalg.inv(self.homography_matrix))[0][0]
                x_cm, y_cm = pt_cm
                
                # Skip if in ignored area
                if not (IGNORED_AREA["x_min"] <= x_cm <= IGNORED_AREA["x_max"] and
                       IGNORED_AREA["y_min"] <= y_cm <= IGNORED_AREA["y_max"]):
                    class_name = results.names[int(cls)]
                    balls.append((x_cm, y_cm, class_name))
                    
                    # Draw detection for debugging
                    cv2.rectangle(frame, (int(x), int(y)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f"{class_name} {conf:.2f}",
                              (int(x), int(y)-10), cv2.FONT_HERSHEY_SIMPLEX,
                              0.5, (0, 255, 0), 2)

        return balls

    def calibrate(self):
        """Perform camera calibration by clicking 4 corners"""
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(self.calibration_points) < 4:
                    self.calibration_points.append((x, y))
                    logger.info(f"Corner {len(self.calibration_points)} set: ({x}, {y})")

                if len(self.calibration_points) == 4:
                    # Calculate homography matrix
                    dst_pts = np.array([
                        [0, 0],
                        [FIELD_WIDTH_CM, 0],
                        [FIELD_WIDTH_CM, FIELD_HEIGHT_CM],
                        [0, FIELD_HEIGHT_CM]
                    ], dtype="float32")
                    src_pts = np.array(self.calibration_points, dtype="float32")
                    self.homography_matrix = cv2.getPerspectiveTransform(dst_pts, src_pts)
                    logger.info("✅ Calibration complete")

        cv2.namedWindow("Calibration")
        cv2.setMouseCallback("Calibration", mouse_callback)
        
        while len(self.calibration_points) < 4:
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            # Draw existing points
            for i, pt in enumerate(self.calibration_points):
                cv2.circle(frame, pt, 5, (0, 255, 0), -1)
                cv2.putText(frame, str(i+1), (pt[0]+10, pt[1]+10),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("Calibration", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyWindow("Calibration")

    def send_command(self, command: str, **params) -> bool:
        """Send a command to the EV3 server"""
        try:
            with socket.create_connection((EV3_IP, EV3_PORT), timeout=5) as s:
                message = {
                    "command": command,
                    **params
                }
                s.sendall((json.dumps(message) + "\n").encode("utf-8"))
                response = s.recv(1024).decode("utf-8").strip()
                return response == "OK"
        except Exception as e:
            logger.error(f"Failed to send command: {e}")
            return False

    def move(self, distance_cm: float) -> bool:
        """Move forward/backward by distance_cm"""
        return self.send_command("MOVE", distance=distance_cm)

    def turn(self, angle_deg: float) -> bool:
        """Turn by angle_deg (positive = CCW)"""
        return self.send_command("TURN", angle=angle_deg)

    def collect(self, distance_cm: float) -> bool:
        """Move forward while collecting"""
        return self.send_command("COLLECT", distance=distance_cm)

    def deliver_balls(self) -> bool:
        """Run collector in reverse to deliver balls"""
        return self.send_command("COLLECT_REVERSE", duration=2.0)

    def stop(self) -> bool:
        """Stop all motors"""
        return self.send_command("STOP")

    def get_approach_vector(self, target_pos: Tuple[float, float]) -> Tuple[Tuple[float, float], float]:
        """Calculate approach position and angle for a target position"""
        dx = target_pos[0] - self.robot_pos[0]
        dy = target_pos[1] - self.robot_pos[1]
        distance = math.hypot(dx, dy)
        
        # Calculate approach point APPROACH_DISTANCE_CM before target
        ratio = (distance - APPROACH_DISTANCE_CM) / distance if distance > APPROACH_DISTANCE_CM else 0
        approach_x = self.robot_pos[0] + dx * ratio
        approach_y = self.robot_pos[1] + dy * ratio
        
        # Calculate angle to target
        target_angle = math.degrees(math.atan2(dy, dx))
        
        return (approach_x, approach_y), target_angle

    def collect_balls(self):
        """Main ball collection loop"""
        cv2.namedWindow("Collection")
        last_plan_time = 0
        last_plan_positions = []
        
        while True:
            try:
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    continue

                # Update robot position
                if not self.update_robot_position(frame):
                    cv2.putText(frame, "Robot markers not detected!", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                              0.7, (0, 0, 255), 2)
                    cv2.imshow("Collection", frame)
                    cv2.waitKey(1)
                    continue

                # Detect balls
                balls = self.detect_balls()
                current_time = time.time()

                # Only replan if we have balls and enough time has passed
                if balls and current_time - last_plan_time >= 3:
                    last_plan_time = current_time
                    last_plan_positions = [(b[0], b[1]) for b in balls]

                    # Sort balls by distance from robot
                    balls.sort(key=lambda b: math.hypot(
                        b[0] - self.robot_pos[0],
                        b[1] - self.robot_pos[1]
                    ))

                    # Take closest balls up to MAX_BALLS_PER_TRIP
                    current_batch = balls[:MAX_BALLS_PER_TRIP]
                    logger.info(f"Planning to collect {len(current_batch)} balls")

                    # Process each ball
                    for ball in current_batch:
                        ball_pos = (ball[0], ball[1])
                        
                        # Get approach position
                        approach_pos, target_angle = self.get_approach_vector(ball_pos)
                        
                        # Turn to face target
                        angle_diff = (target_angle - self.robot_heading + 180) % 360 - 180
                        if abs(angle_diff) > 5:
                            logger.info(f"Turning {angle_diff:.1f} degrees")
                            if not self.turn(angle_diff):
                                raise Exception("Turn failed")
                            time.sleep(0.5)
                        
                        # Move to approach position
                        distance = math.hypot(
                            approach_pos[0] - self.robot_pos[0],
                            approach_pos[1] - self.robot_pos[1]
                        )
                        logger.info(f"Moving {distance:.1f} cm to approach")
                        if not self.move(distance):
                            raise Exception("Move failed")
                        time.sleep(0.5)
                        
                        # Collect ball
                        logger.info("Collecting ball")
                        if not self.collect(COLLECTION_DISTANCE_CM):
                            raise Exception("Collection failed")
                        time.sleep(0.5)

                    # After collecting batch, return to goal
                    if current_batch:
                        # Calculate goal position (right side, center)
                        goal_x = FIELD_WIDTH_CM - 30  # 30cm from right edge
                        goal_y = FIELD_HEIGHT_CM / 2
                        
                        # Turn to face goal
                        dx = goal_x - self.robot_pos[0]
                        dy = goal_y - self.robot_pos[1]
                        goal_angle = math.degrees(math.atan2(dy, dx))
                        
                        angle_diff = (goal_angle - self.robot_heading + 180) % 360 - 180
                        if abs(angle_diff) > 5:
                            logger.info(f"Turning {angle_diff:.1f} degrees to goal")
                            if not self.turn(angle_diff):
                                raise Exception("Turn to goal failed")
                            time.sleep(0.5)
                        
                        # Move to goal
                        distance = math.hypot(dx, dy)
                        logger.info(f"Moving {distance:.1f} cm to goal")
                        if not self.move(distance):
                            raise Exception("Move to goal failed")
                        time.sleep(0.5)
                        
                        # Deliver balls
                        logger.info("Delivering balls")
                        if not self.deliver_balls():
                            raise Exception("Ball delivery failed")
                        time.sleep(0.5)

                # Show frame
                cv2.imshow("Collection", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

            except Exception as e:
                logger.error(f"Error in collection loop: {e}")
                self.stop()
                time.sleep(1)

        cv2.destroyWindow("Collection")

    def run(self):
        """Main run loop"""
        try:
            # First calibrate the camera
            logger.info("Starting camera calibration...")
            self.calibrate()
            
            if self.homography_matrix is None:
                logger.error("Calibration failed")
                return
            
            # Start continuous collection
            logger.info("Starting ball collection...")
            self.collect_balls()
            
        except KeyboardInterrupt:
            logger.info("Stopping...")
        finally:
            self.stop()
            self.cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    collector = BallCollector()
    collector.run() 