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
import random

# ArUco marker configuration
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()
ARUCO_MARKER_ID = 0  # ID of the marker to track

# Configuration
EV3_IP = "172.20.10.6"
EV3_PORT = 12345

# Model configuration
MODEL_PATH = "weights(2).pt"  # Using local weights
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45  # Non-maximum suppression threshold
MODEL_IMAGE_SIZE = 640  # YOLOv8 default size

# Camera settings
CAMERA_WIDTH = 640   # Match model input size
CAMERA_HEIGHT = 640  # Square aspect ratio for better detection
CAMERA_FPS = 30

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

# Obstacle detection parameters
OBSTACLE_GRID_SIZE = 10     # cm per grid cell
MIN_OBSTACLE_AREA = 100     # minimum pixel area for obstacles
MAX_OBSTACLE_AREA = 400     # maximum cm² area for obstacles

# Color detection ranges (HSV)
GREEN_LOWER = np.array([35, 50, 50])   # Robot base marker
GREEN_UPPER = np.array([85, 255, 255])
PINK_LOWER = np.array([145, 50, 50])   # Direction marker
PINK_UPPER = np.array([175, 255, 255])

# Ignored area (center obstacle)
IGNORED_AREA = {
    "x_min": 50, "x_max": 100,
    "y_min": 50, "y_max": 100
}

# Ball tracking parameters
BALL_TRACKING_HISTORY = 5  # Number of frames to keep ball history
MIN_DETECTION_CONF = 0.5   # Minimum confidence for ball detection
TRACKING_DISTANCE_THRESHOLD = 30  # Maximum cm between tracked positions

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
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

        # Robot state
        self.robot_pos = (ROBOT_START_X, ROBOT_START_Y)
        self.robot_heading = ROBOT_START_HEADING
        
        # Calibration
        self.calibration_points = []
        self.homography_matrix = None
        
        # Ball tracking
        self.tracked_balls = {}  # Dictionary to store ball tracking history
        self.ball_history = []   # List to store recent ball positions
        self.frame_count = 0     # Counter for tracking frames
        
        # Initialize class colors for visualization
        self.class_colors = {}

    def detect_markers(self, frame):
        """Detect ArUco markers for robot position tracking"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMS)
        
        if ids is not None:
            for idx, marker_id in enumerate(ids.flatten()):
                if marker_id != ARUCO_MARKER_ID:
                    continue
                    
                # Get marker corners and center
                pts = corners[idx][0]
                center_x = int(pts[:, 0].mean())
                center_y = int(pts[:, 1].mean())
                
                # Calculate heading from marker orientation
                # Using first two points to determine direction
                dx = pts[1][0] - pts[0][0]
                dy = pts[1][1] - pts[0][1]
                heading = math.degrees(math.atan2(dy, dx))
                
                # Convert to cm coordinates if homography matrix exists
                if self.homography_matrix is not None:
                    pt_px = np.array([[[center_x, center_y]]], dtype="float32")
                    pt_cm = cv2.perspectiveTransform(pt_px, np.linalg.inv(self.homography_matrix))[0][0]
                    return pt_cm[0], pt_cm[1], heading
                    
        return None, None, None

    def update_robot_position(self, frame):
        """Update robot position and heading based on ArUco markers"""
        x_cm, y_cm, heading = self.detect_markers(frame)
        
        if x_cm is not None and y_cm is not None:
            self.robot_pos = (x_cm, y_cm)
            self.robot_heading = heading
            
            # Draw marker visualization
            if self.homography_matrix is not None:
                pt_cm = np.array([[[x_cm, y_cm]]], dtype="float32")
                pt_px = cv2.perspectiveTransform(pt_cm, self.homography_matrix)[0][0].astype(int)
                cv2.circle(frame, tuple(pt_px), 5, (0, 255, 0), -1)
                
                # Draw heading line
                angle_rad = math.radians(heading)
                end_x = pt_px[0] + int(30 * math.cos(angle_rad))
                end_y = pt_px[1] + int(30 * math.sin(angle_rad))
                cv2.line(frame, tuple(pt_px), (end_x, end_y), (0, 255, 0), 2)
            
            return True
            
        return False

    def detect_balls(self) -> List[Tuple[float, float, str]]:
        """Detect balls in current camera view, return list of (x_cm, y_cm, label)"""
        ret, frame = self.cap.read()
        if not ret:
            return []

        # Run YOLOv8 inference with optimized settings
        results = self.model(
            frame,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False
        )[0]
        
        current_balls = []

        # Process detections
        for det in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = det
            
            # Skip low confidence detections
            if conf < MIN_DETECTION_CONF:
                continue
                
            # Calculate center point
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Get class name
            class_name = results.names[int(cls)]
            
            # Convert to cm using homography if available
            if self.homography_matrix is not None:
                pt_px = np.array([[[center_x, center_y]]], dtype="float32")
                pt_cm = cv2.perspectiveTransform(pt_px, np.linalg.inv(self.homography_matrix))[0][0]
                x_cm, y_cm = pt_cm
                
                # Skip if in ignored area
                if (IGNORED_AREA["x_min"] <= x_cm <= IGNORED_AREA["x_max"] and
                    IGNORED_AREA["y_min"] <= y_cm <= IGNORED_AREA["y_max"]):
                    continue
                
                # Only process ball detections
                if 'ball' in class_name.lower():
                    current_balls.append((x_cm, y_cm, class_name))
                    
                    # Get color for visualization
                    color = self.class_colors.setdefault(class_name, (
                        random.randint(0,255),
                        random.randint(0,255),
                        random.randint(0,255)
                    ))
                    
                    # Draw detection box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    
                    # Draw confidence and position
                    label = f"{class_name} ({conf:.2f})"
                    cv2.putText(frame, label, (int(x1), int(y1)-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Draw position in cm
                    pos_text = f"({x_cm:.1f}, {y_cm:.1f})"
                    cv2.putText(frame, pos_text, (int(x1), int(y1)+20),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Update ball tracking
        self.update_ball_tracking(current_balls)
        
        # Get stable ball positions
        stable_balls = self.get_stable_balls()
        
        # Draw tracking visualization
        if self.homography_matrix is not None:
            for ball in stable_balls:
                x_cm, y_cm = ball[0], ball[1]
                pt_cm = np.array([[[x_cm, y_cm]]], dtype="float32")
                pt_px = cv2.perspectiveTransform(pt_cm, self.homography_matrix)[0][0].astype(int)
                
                # Draw circle for stable balls
                cv2.circle(frame, tuple(pt_px), 15, (0, 255, 0), 2)
                cv2.circle(frame, tuple(pt_px), 2, (0, 255, 0), -1)

        return stable_balls

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

    def draw_path(self, frame, path):
        """Draw the planned path on the frame"""
        if self.homography_matrix is None:
            return frame

        overlay = frame.copy()
        
        # Colors for different point types
        colors = {
            'approach': (0, 255, 255),  # Yellow
            'collect': (0, 255, 0),     # Green
            'goal': (0, 0, 255)         # Red
        }
        
        # Draw lines between points
        for i in range(len(path)):
            # Convert current point from cm to pixels
            pt_cm = np.array([[[path[i]['pos'][0], path[i]['pos'][1]]]], dtype="float32")
            pt_px = cv2.perspectiveTransform(pt_cm, self.homography_matrix)[0][0].astype(int)
            
            # Draw point
            color = colors[path[i]['type']]
            cv2.circle(overlay, tuple(pt_px), 5, color, -1)
            
            # Draw line to next point
            if i < len(path) - 1:
                next_cm = np.array([[[path[i+1]['pos'][0], path[i+1]['pos'][1]]]], dtype="float32")
                next_px = cv2.perspectiveTransform(next_cm, self.homography_matrix)[0][0].astype(int)
                cv2.line(overlay, tuple(pt_px), tuple(next_px), (150, 150, 150), 2)
            
            # Draw direction arrow for approach and goal points
            if path[i]['type'] in ['approach', 'goal']:
                angle_rad = math.radians(path[i]['angle'])
                end_x = pt_px[0] + int(20 * math.cos(angle_rad))
                end_y = pt_px[1] + int(20 * math.sin(angle_rad))
                cv2.arrowedLine(overlay, tuple(pt_px), (end_x, end_y), color, 2)
            
            # Add labels
            label = ""
            if path[i]['type'] == 'collect':
                label = path[i]['ball_type']
            elif path[i]['type'] == 'approach':
                label = 'A'
            elif path[i]['type'] == 'goal':
                label = 'G'
            
            if label:
                cv2.putText(overlay, label, 
                          (pt_px[0] + 10, pt_px[1] + 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw robot position and heading
        if self.robot_pos:
            robot_cm = np.array([[[self.robot_pos[0], self.robot_pos[1]]]], dtype="float32")
            robot_px = cv2.perspectiveTransform(robot_cm, self.homography_matrix)[0][0].astype(int)
            
            # Robot circle
            cv2.circle(overlay, tuple(robot_px), 8, (255, 0, 0), -1)
            
            # Robot heading
            angle_rad = math.radians(self.robot_heading)
            end_x = robot_px[0] + int(25 * math.cos(angle_rad))
            end_y = robot_px[1] + int(25 * math.sin(angle_rad))
            cv2.arrowedLine(overlay, tuple(robot_px), (end_x, end_y), (255, 0, 0), 2)
        
        # Blend overlay with original frame
        alpha = 0.7
        return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    def draw_status(self, frame):
        """Draw status information on frame"""
        # Create semi-transparent overlay for status
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 120), (0, 0, 0), -1)
        alpha = 0.7
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # Add status text
        y = 30
        cv2.putText(frame, f"Robot Position: ({self.robot_pos[0]:.1f}, {self.robot_pos[1]:.1f})",
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y += 20
        cv2.putText(frame, f"Robot Heading: {self.robot_heading:.1f}°",
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y += 20
        cv2.putText(frame, "Commands:",
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y += 20
        cv2.putText(frame, "SPACE - Execute Path | Q - Quit",
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame

    def collect_balls(self):
        """Main ball collection loop"""
        cv2.namedWindow("Collection")
        last_plan_time = 0
        last_plan_positions = []
        current_path = []
        
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

                    # Create path plan
                    current_path = []
                    current_pos = self.robot_pos
                    current_heading = self.robot_heading

                    # Add each ball to the path
                    for ball in current_batch:
                        ball_pos = (ball[0], ball[1])
                        
                        # Get approach position
                        approach_pos, target_angle = self.get_approach_vector(ball_pos)
                        
                        # Add approach point
                        current_path.append({
                            'type': 'approach',
                            'pos': approach_pos,
                            'angle': target_angle
                        })
                        
                        # Add collection point
                        current_path.append({
                            'type': 'collect',
                            'pos': ball_pos,
                            'ball_type': ball[2],
                            'angle': target_angle
                        })
                        
                        current_pos = ball_pos
                        current_heading = target_angle

                    # Add goal point if we have any balls
                    if current_batch:
                        goal_x = FIELD_WIDTH_CM - 30  # 30cm from right edge
                        goal_y = FIELD_HEIGHT_CM / 2
                        goal_angle = 0  # Face right at goal
                        
                        current_path.append({
                            'type': 'goal',
                            'pos': (goal_x, goal_y),
                            'angle': goal_angle
                        })

                # Draw current path and status
                frame_with_path = frame.copy()
                if current_path:
                    frame_with_path = self.draw_path(frame_with_path, current_path)
                frame_with_path = self.draw_status(frame_with_path)
                cv2.imshow("Collection", frame_with_path)

                # Handle key commands
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' ') and current_path:
                    logger.info("Executing planned path...")
                    
                    try:
                        # Execute each point in the path
                        for point in current_path:
                            # Calculate turn angle
                            angle_diff = (point['angle'] - self.robot_heading + 180) % 360 - 180
                            if abs(angle_diff) > 5:
                                logger.info(f"Turning {angle_diff:.1f} degrees")
                                if not self.turn(angle_diff):
                                    raise Exception("Turn failed")
                                time.sleep(0.5)
                            
                            # Calculate movement distance
                            dx = point['pos'][0] - self.robot_pos[0]
                            dy = point['pos'][1] - self.robot_pos[1]
                            distance = math.hypot(dx, dy)
                            
                            # Execute movement based on point type
                            if point['type'] == 'collect':
                                logger.info(f"Collecting {point['ball_type']} ball")
                                if not self.collect(COLLECTION_DISTANCE_CM):
                                    raise Exception("Collection failed")
                            elif point['type'] == 'goal':
                                logger.info("Moving to goal")
                                if not self.move(distance):
                                    raise Exception("Move to goal failed")
                                # Deliver balls at goal
                                logger.info("Delivering balls")
                                if not self.deliver_balls():
                                    raise Exception("Ball delivery failed")
                            else:  # approach
                                logger.info(f"Moving {distance:.1f} cm")
                                if not self.move(distance):
                                    raise Exception("Move failed")
                            
                            time.sleep(0.5)
                            
                            # Update position and visualization
                            ret, frame = self.cap.read()
                            if ret:
                                self.update_robot_position(frame)
                                frame_with_path = self.draw_path(frame, current_path)
                                frame_with_path = self.draw_status(frame_with_path)
                                cv2.imshow("Collection", frame_with_path)
                                cv2.waitKey(1)
                        
                        # Clear path after successful execution
                        current_path = []
                        
                    except Exception as e:
                        logger.error(f"Path execution failed: {e}")
                        self.stop()
                        time.sleep(1)

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

    def detect_red_cross_obstacles(self, frame):
        """Detect red cross obstacles in the frame"""
        if self.homography_matrix is None:
            return set()
            
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Red color range in HSV (handles both red ranges)
        mask1 = cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        obstacles = set()
        min_area_px = 100  # Minimum area to consider
        max_area_cm2 = 400  # Maximum area in cm²
        
        # Calculate pixels per cm
        px_per_x = frame.shape[1] / FIELD_WIDTH_CM
        px_per_y = frame.shape[0] / FIELD_HEIGHT_CM
        
        for cnt in contours:
            area_px = cv2.contourArea(cnt)
            if area_px < min_area_px:
                continue
                
            x, y, w, h = cv2.boundingRect(cnt)
            area_cm2 = (w / px_per_x) * (h / px_per_y)
            
            if area_cm2 > max_area_cm2:
                continue
                
            # Sample points within contour
            for sx in range(x, x + w, 10):
                for sy in range(y, y + h, 10):
                    if cv2.pointPolygonTest(cnt, (sx, sy), False) >= 0:
                        # Convert to cm coordinates
                        pt_px = np.array([[[float(sx), float(sy)]]], dtype="float32")
                        pt_cm = cv2.perspectiveTransform(pt_px, np.linalg.inv(self.homography_matrix))[0][0]
                        
                        # Add to obstacles if within field bounds
                        if (0 <= pt_cm[0] <= FIELD_WIDTH_CM and 
                            0 <= pt_cm[1] <= FIELD_HEIGHT_CM):
                            obstacles.add((pt_cm[0], pt_cm[1]))
                            
            # Draw obstacle visualization
            cv2.drawContours(frame, [cnt], -1, (0, 0, 255), 2)
            
        return obstacles

    def update_ball_tracking(self, current_balls):
        """Update ball tracking with new detections"""
        self.frame_count += 1
        
        # Convert current balls to dictionary for easier lookup
        current_dict = {(x, y): label for x, y, label in current_balls}
        
        # Update tracked balls
        new_tracked = {}
        for pos, label in current_dict.items():
            matched = False
            
            # Try to match with existing tracked balls
            for tracked_pos in self.tracked_balls:
                dist = math.hypot(pos[0] - tracked_pos[0], pos[1] - tracked_pos[1])
                if dist < TRACKING_DISTANCE_THRESHOLD:
                    # Update tracking history
                    history = self.tracked_balls[tracked_pos]
                    history.append((self.frame_count, pos))
                    # Keep only recent history
                    while len(history) > BALL_TRACKING_HISTORY:
                        history.pop(0)
                    new_tracked[pos] = history
                    matched = True
                    break
            
            # If no match found, start new track
            if not matched:
                new_tracked[pos] = [(self.frame_count, pos)]
        
        self.tracked_balls = new_tracked
        
        # Clean up old tracks
        for pos in list(self.tracked_balls.keys()):
            history = self.tracked_balls[pos]
            if self.frame_count - history[-1][0] > BALL_TRACKING_HISTORY:
                del self.tracked_balls[pos]

    def get_stable_balls(self):
        """Get balls that have been consistently tracked"""
        stable_balls = []
        
        for pos, history in self.tracked_balls.items():
            # Only consider balls with enough history
            if len(history) >= BALL_TRACKING_HISTORY - 1:
                # Calculate average position
                recent_positions = [p for _, p in history[-3:]]  # Use last 3 positions
                avg_x = sum(x for x, _ in recent_positions) / len(recent_positions)
                avg_y = sum(y for y, _ in recent_positions) / len(recent_positions)
                
                # Calculate position stability
                max_deviation = max(
                    math.hypot(x - avg_x, y - avg_y)
                    for x, y in recent_positions
                )
                
                # If position is stable, add to result
                if max_deviation < TRACKING_DISTANCE_THRESHOLD / 2:
                    stable_balls.append((avg_x, avg_y, "ball"))
        
        return stable_balls

if __name__ == "__main__":
    collector = BallCollector()
    collector.run() 