#!/usr/bin/env python3
"""
Robot Controller for Ball Collection
This script handles:
1. Ball detection using Roboflow models
2. Robot position tracking
3. Path planning and execution
4. Ball collection and delivery
"""

import cv2
import math
import json
import socket
import logging
import numpy as np
from typing import List, Tuple, Dict, Optional
from roboflow import Roboflow
import time

# Configuration
class Config:
    # Network settings
    EV3_IP = "172.20.10.6"
    EV3_PORT = 12345
    
    # Roboflow settings
    ROBOFLOW_API_KEY = "LdvRakmEpZizttEFtQap"
    RF_WORKSPACE = "legoms3"
    RF_PROJECT = "golfbot-fyxfe-etdz0"
    
    # Field dimensions (cm)
    FIELD_WIDTH = 180
    FIELD_HEIGHT = 120
    
    # Robot dimensions and parameters
    ROBOT_WIDTH = 15
    ROBOT_LENGTH = 20
    COLLECTION_DISTANCE = 20
    APPROACH_DISTANCE = 30
    MAX_BALLS_PER_TRIP = 3
    
    # Camera settings
    CAMERA_INDEX = 1
    CAMERA_WIDTH = 1280
    CAMERA_HEIGHT = 720
    CAMERA_FPS = 30
    
    # Color detection ranges (HSV)
    GREEN_LOWER = np.array([35, 50, 50])
    GREEN_UPPER = np.array([85, 255, 255])
    PINK_LOWER = np.array([145, 50, 50])
    PINK_UPPER = np.array([175, 255, 255])
    
    # Safety margins
    WALL_MARGIN = 10
    OBSTACLE_MARGIN = 15

class RobotCommander:
    """Handles communication with the EV3 robot"""
    
    def __init__(self):
        self.config = Config()
        self.logger = logging.getLogger("RobotCommander")
    
    def send_command(self, command: str, **params) -> bool:
        """Send a command to the EV3"""
        try:
            with socket.create_connection((self.config.EV3_IP, self.config.EV3_PORT), timeout=15) as s:
                message = {"command": command, **params}
                s.sendall((json.dumps(message) + "\n").encode("utf-8"))
                response = s.recv(1024).decode("utf-8").strip()
                return response == "OK"
        except Exception as e:
            self.logger.error(f"Command failed: {command} - {str(e)}")
            return False
    
    def move(self, distance: float) -> bool:
        """Move forward/backward by distance (cm)"""
        return self.send_command("MOVE", distance=distance)
    
    def turn(self, angle: float) -> bool:
        """Turn by angle (degrees, positive = CCW)"""
        return self.send_command("TURN", angle=angle)
    
    def collect(self, distance: float) -> bool:
        """Move forward while collecting"""
        return self.send_command("COLLECT", distance=distance)
    
    def deliver(self, duration: float = 2.0) -> bool:
        """Run collector in reverse to deliver balls"""
        return self.send_command("COLLECT_REVERSE", duration=duration)
    
    def stop(self) -> bool:
        """Stop all motors"""
        return self.send_command("STOP")

class VisionSystem:
    """Handles computer vision tasks"""
    
    def __init__(self, weight_file: str):
        self.config = Config()
        self.logger = logging.getLogger("VisionSystem")
        
        # Initialize camera
        self.cap = cv2.VideoCapture(self.config.CAMERA_INDEX, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, self.config.CAMERA_FPS)
        
        # Initialize Roboflow
        rf = Roboflow(api_key=self.config.ROBOFLOW_API_KEY)
        self.project = rf.workspace(self.config.RF_WORKSPACE).project(self.config.RF_PROJECT)
        self.model = self.project.version(2).model
        
        # Calibration matrix
        self.homography_matrix = None
        self.calibration_points = []
    
    def calibrate(self):
        """Perform camera calibration"""
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(self.calibration_points) < 4:
                self.calibration_points.append((x, y))
                if len(self.calibration_points) == 4:
                    # Calculate homography matrix
                    dst_pts = np.array([
                        [0, 0],
                        [self.config.FIELD_WIDTH, 0],
                        [self.config.FIELD_WIDTH, self.config.FIELD_HEIGHT],
                        [0, self.config.FIELD_HEIGHT]
                    ], dtype="float32")
                    src_pts = np.array(self.calibration_points, dtype="float32")
                    self.homography_matrix = cv2.getPerspectiveTransform(dst_pts, src_pts)
        
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
        return self.homography_matrix is not None
    
    def detect_markers(self, frame) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
        """Detect green base and pink direction markers"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Detect green base marker
        green_mask = cv2.inRange(hsv, self.config.GREEN_LOWER, self.config.GREEN_UPPER)
        green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Detect pink direction marker
        pink_mask = cv2.inRange(hsv, self.config.PINK_LOWER, self.config.PINK_UPPER)
        pink_contours, _ = cv2.findContours(pink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        green_center = None
        pink_center = None
        
        if green_contours:
            largest_green = max(green_contours, key=cv2.contourArea)
            if cv2.contourArea(largest_green) > 100:
                M = cv2.moments(largest_green)
                if M["m00"] != 0:
                    green_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        
        if pink_contours:
            largest_pink = max(pink_contours, key=cv2.contourArea)
            if cv2.contourArea(largest_pink) > 50:
                M = cv2.moments(largest_pink)
                if M["m00"] != 0:
                    pink_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        
        return green_center, pink_center
    
    def detect_objects(self, frame) -> Tuple[List[Dict], List[Dict]]:
        """Detect balls and obstacles using Roboflow model"""
        # Resize for model
        small = cv2.resize(frame, (416, 416))
        
        # Get predictions
        predictions = self.model.predict(small, confidence=30, overlap=20).json()
        
        balls = []
        obstacles = []
        
        scale_x = frame.shape[1] / 416
        scale_y = frame.shape[0] / 416
        
        for pred in predictions.get('predictions', []):
            x_px = int(pred['x'] * scale_x)
            y_px = int(pred['y'] * scale_y)
            class_name = pred['class']
            
            # Convert to field coordinates
            if self.homography_matrix is not None:
                pt_px = np.array([[[x_px, y_px]]], dtype="float32")
                pt_cm = cv2.perspectiveTransform(pt_px, np.linalg.inv(self.homography_matrix))[0][0]
                
                obj = {
                    'type': class_name,
                    'position': (pt_cm[0], pt_cm[1]),
                    'confidence': pred['confidence']
                }
                
                if class_name in ['egg', 'cross']:
                    obstacles.append(obj)
                else:
                    balls.append(obj)
        
        return balls, obstacles

class RobotController:
    """Main robot control class"""
    
    def __init__(self, weight_file: str):
        self.config = Config()
        self.logger = logging.getLogger("RobotController")
        
        self.commander = RobotCommander()
        self.vision = VisionSystem(weight_file)
        
        # Robot state
        self.position = None
        self.heading = None
    
    def run(self):
        """Main control loop"""
        # First calibrate
        if not self.vision.calibrate():
            self.logger.error("Calibration failed")
            return
        
        cv2.namedWindow("Robot Control")
        
        try:
            while True:
                ret, frame = self.vision.cap.read()
                if not ret:
                    continue
                
                # Update robot position
                green_center, pink_center = self.vision.detect_markers(frame)
                if green_center and pink_center:
                    # Convert to field coordinates and update state
                    green_px = np.array([[[float(green_center[0]), float(green_center[1])]]], dtype="float32")
                    green_cm = cv2.perspectiveTransform(green_px, np.linalg.inv(self.vision.homography_matrix))[0][0]
                    
                    self.position = (green_cm[0], green_cm[1])
                    
                    # Calculate heading
                    pink_px = np.array([[[float(pink_center[0]), float(pink_center[1])]]], dtype="float32")
                    pink_cm = cv2.perspectiveTransform(pink_px, np.linalg.inv(self.vision.homography_matrix))[0][0]
                    
                    dx = pink_cm[0] - green_cm[0]
                    dy = pink_cm[1] - green_cm[1]
                    self.heading = math.degrees(math.atan2(dy, dx))
                
                # Detect balls and obstacles
                balls, obstacles = self.vision.detect_objects(frame)
                
                # Draw visualization
                self.draw_visualization(frame, balls, obstacles)
                
                cv2.imshow("Robot Control", frame)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    # Plan and execute collection sequence
                    self.collect_balls(balls, obstacles)
        
        finally:
            self.cleanup()
    
    def collect_balls(self, balls: List[Dict], obstacles: List[Dict]):
        """Plan and execute ball collection sequence"""
        if not balls or not self.position:
            return
        
        # Sort balls by distance
        balls.sort(key=lambda b: math.hypot(
            b['position'][0] - self.position[0],
            b['position'][1] - self.position[1]
        ))
        
        # Take closest balls up to limit
        current_batch = balls[:self.config.MAX_BALLS_PER_TRIP]
        
        for ball in current_batch:
            # Calculate approach position
            dx = ball['position'][0] - self.position[0]
            dy = ball['position'][1] - self.position[1]
            angle = math.degrees(math.atan2(dy, dx))
            
            # Turn to face ball
            angle_diff = (angle - self.heading + 180) % 360 - 180
            if abs(angle_diff) > 5:
                self.commander.turn(angle_diff)
            
            # Move to ball and collect
            distance = math.hypot(dx, dy)
            self.commander.collect(distance)
        
        # After collecting, deliver to goal
        self.deliver_to_goal()
    
    def deliver_to_goal(self):
        """Navigate to goal and deliver balls"""
        if not self.position:
            return
        
        # Calculate path to goal
        goal_x = self.config.FIELD_WIDTH - 30  # 30cm from right wall
        goal_y = self.config.FIELD_HEIGHT / 2   # Center of field
        
        dx = goal_x - self.position[0]
        dy = goal_y - self.position[1]
        
        angle = math.degrees(math.atan2(dy, dx))
        angle_diff = (angle - self.heading + 180) % 360 - 180
        
        # Turn to face goal
        if abs(angle_diff) > 5:
            self.commander.turn(angle_diff)
        
        # Move to goal
        distance = math.hypot(dx, dy)
        self.commander.move(distance)
        
        # Deliver balls
        self.commander.deliver()
    
    def draw_visualization(self, frame, balls, obstacles):
        """Draw visualization overlays"""
        # Draw robot position and heading
        if self.position and self.vision.homography_matrix is not None:
            pos_cm = np.array([[[self.position[0], self.position[1]]]], dtype="float32")
            pos_px = cv2.perspectiveTransform(pos_cm, self.vision.homography_matrix)[0][0].astype(int)
            
            cv2.circle(frame, tuple(pos_px), 8, (255, 0, 0), -1)
            
            # Draw heading
            angle_rad = math.radians(self.heading)
            end_x = pos_px[0] + int(25 * math.cos(angle_rad))
            end_y = pos_px[1] + int(25 * math.sin(angle_rad))
            cv2.arrowedLine(frame, tuple(pos_px), (end_x, end_y), (255, 0, 0), 2)
        
        # Draw balls and obstacles
        for ball in balls:
            pos_cm = np.array([[[ball['position'][0], ball['position'][1]]]], dtype="float32")
            pos_px = cv2.perspectiveTransform(pos_cm, self.vision.homography_matrix)[0][0].astype(int)
            
            cv2.circle(frame, tuple(pos_px), 5, (0, 255, 0), -1)
            cv2.putText(frame, f"{ball['type']} ({ball['confidence']:.1f})",
                       (pos_px[0] + 10, pos_px[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        for obstacle in obstacles:
            pos_cm = np.array([[[obstacle['position'][0], obstacle['position'][1]]]], dtype="float32")
            pos_px = cv2.perspectiveTransform(pos_cm, self.vision.homography_matrix)[0][0].astype(int)
            
            cv2.circle(frame, tuple(pos_px), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"{obstacle['type']} ({obstacle['confidence']:.1f})",
                       (pos_px[0] + 10, pos_px[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    def cleanup(self):
        """Clean up resources"""
        self.commander.stop()
        self.vision.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create controller with first weight file
    controller = RobotController("weights(1).pt")
    controller.run() 