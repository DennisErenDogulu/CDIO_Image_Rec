#!/usr/bin/env python3
"""
Continuous Ball Collection Client

This script implements a continuous ball collection strategy using visual markers:
1. Track robot position using green base marker
2. Track robot heading using pink direction marker
3. Detect balls in the camera view
4. Navigate to and collect balls based on visual position tracking
"""

import cv2
import math
import json
import socket
import logging
import numpy as np
from typing import List, Tuple, Optional
from roboflow import Roboflow
import time

# Configuration
EV3_IP = "172.20.10.6"
EV3_PORT = 12345
ROBOFLOW_API_KEY = "LdvRakmEpZizttEFtQap"
RF_WORKSPACE = "legoms3"
RF_PROJECT = "golfbot-fyxfe-etdz0" 
RF_VERSION = 3

# Color detection ranges (HSV)
GREEN_LOWER = np.array([35, 50, 50])
GREEN_UPPER = np.array([85, 255, 255])

# Multiple pink ranges to handle different lighting conditions
PINK_RANGES = [
    # Primary pink range (original)
    (np.array([145, 50, 50]), np.array([175, 255, 255])),
    # Brighter pink (higher value, lower saturation for bright light)
    (np.array([140, 30, 100]), np.array([180, 200, 255])),
    # Darker pink (for shadows or dim lighting)
    (np.array([150, 80, 30]), np.array([175, 255, 200])),
    # Extended hue range for color variations
    (np.array([135, 40, 40]), np.array([185, 255, 255]))
]

# Marker dimensions
GREEN_MARKER_WIDTH_CM = 20  # Width of green base sheet
PINK_MARKER_WIDTH_CM = 5    # Width of pink direction marker

# Wall configuration
WALL_SAFETY_MARGIN = 1  # cm, minimum distance to keep from walls
GOAL_WIDTH = 30  # cm, width of the goal area to keep clear

# Robot configuration
ROBOT_START_X = 20  # cm from left edge
ROBOT_START_Y = 20  # cm from bottom edge
ROBOT_WIDTH = 15   # cm
ROBOT_LENGTH = 20  # cm
ROBOT_START_HEADING = 0  # degrees (0 = facing east)

# Physical constraints
FIELD_WIDTH_CM = 180
FIELD_HEIGHT_CM = 120
COLLECTION_DISTANCE_CM = 20  # Distance to move forward when collecting
APPROACH_DISTANCE_CM = 30    # Distance to keep from ball for approach
MAX_BALLS_PER_TRIP = 3

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

def point_to_line_distance(point, line_start, line_end):
    """Calculate the shortest distance from a point to a line segment"""
    x, y = point
    x1, y1 = line_start
    x2, y2 = line_end
    
    # Vector from line start to end
    line_vec = (x2 - x1, y2 - y1)
    # Vector from line start to point
    point_vec = (x - x1, y - y1)
    # Length of line
    line_len = math.hypot(line_vec[0], line_vec[1])
    
    if line_len == 0:
        return math.hypot(point_vec[0], point_vec[1])
    
    # Project point vector onto line vector to get distance along line
    t = max(0, min(1, (point_vec[0] * line_vec[0] + point_vec[1] * line_vec[1]) / (line_len * line_len)))
    
    # Calculate projection point
    proj_x = x1 + t * line_vec[0]
    proj_y = y1 + t * line_vec[1]
    
    return math.hypot(x - proj_x, y - proj_y)

def check_wall_collision(start_pos, end_pos, walls, safety_margin):
    """Check if a path between two points collides with any walls"""
    for wall_start_x, wall_start_y, wall_end_x, wall_end_y in walls:
        # Check if either endpoint is too close to the wall
        if (point_to_line_distance(start_pos, (wall_start_x, wall_start_y), (wall_end_x, wall_end_y)) < safety_margin or
            point_to_line_distance(end_pos, (wall_start_x, wall_start_y), (wall_end_x, wall_end_y)) < safety_margin):
            return True
            
        # Check if path intersects with wall
        # Using line segment intersection formula
        x1, y1 = start_pos
        x2, y2 = end_pos
        x3, y3 = wall_start_x, wall_start_y
        x4, y4 = wall_end_x, wall_end_y
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denom == 0:  # Lines are parallel
            continue
            
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
        
        if 0 <= t <= 1 and 0 <= u <= 1:
            return True
            
    return False

class BallCollector:
    def __init__(self):
        # Initialize Roboflow
        self.rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        self.project = self.rf.workspace(RF_WORKSPACE).project(RF_PROJECT)
        self.model = self.project.version(RF_VERSION).model
        
        # Initialize camera
        self.cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # Use camera index 1 for USB camera
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera")

        # Set camera properties for reduced lag - use lower resolution for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Reduced from 1280
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Reduced from 720
        self.cap.set(cv2.CAP_PROP_FPS, 15)            # Reduced from 30 for smoother processing
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)      # Minimize buffer to reduce lag
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)       # Disable autofocus for consistent performance

        # Additional lag reduction settings
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Disable auto exposure
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))  # Use MJPG codec
        
        # Robot state
        self.robot_pos = (ROBOT_START_X, ROBOT_START_Y)  # Starting position
        self.robot_heading = ROBOT_START_HEADING  # Starting heading
        
        # Calibration points for homography
        self.calibration_points = []
        self.homography_matrix = None
        
        # Wall configuration
        self.walls = []  # Will be set after calibration
        
        # Goal dimensions and positions
        self.goal_y_center = FIELD_HEIGHT_CM / 2  # Center Y coordinate of goal
        self.goal_approach_distance = 20  # Distance to stop in front of goal
        self.delivery_time = 2.0  # Seconds to run collector in reverse
        
        # Frame skipping for lag reduction
        self.frame_skip_counter = 0
        self.frame_skip_interval = 2  # Process every 3rd frame for better performance

    def flush_camera_buffer(self):
        """Flush camera buffer to get the most recent frame"""
        # Read and discard multiple frames to clear buffer
        for _ in range(3):
            self.cap.grab()

    def get_fresh_frame(self):
        """Get the most recent frame with minimal lag"""
        # Flush buffer first
        self.flush_camera_buffer()
        
        # Get the latest frame
        ret, frame = self.cap.read()
        return ret, frame

    def detect_markers(self, frame):
        """Detect green base and pink direction markers with improved lighting robustness"""
        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Detect green base marker
        green_mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
        
        # Apply morphological operations to clean up green mask
        kernel = np.ones((3,3), np.uint8)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
        
        green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Detect pink direction marker using multiple color ranges
        pink_mask = np.zeros_like(hsv[:,:,0])
        
        # Try each pink range and combine the results
        for pink_lower, pink_upper in PINK_RANGES:
            range_mask = cv2.inRange(hsv, pink_lower, pink_upper)
            pink_mask = cv2.bitwise_or(pink_mask, range_mask)
        
        # Apply morphological operations to clean up pink mask
        kernel_small = np.ones((2,2), np.uint8)
        pink_mask = cv2.morphologyEx(pink_mask, cv2.MORPH_OPEN, kernel_small)
        pink_mask = cv2.morphologyEx(pink_mask, cv2.MORPH_CLOSE, kernel)
        
        # Apply Gaussian blur to reduce noise
        pink_mask = cv2.GaussianBlur(pink_mask, (3, 3), 0)
        
        pink_contours, _ = cv2.findContours(pink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find largest green contour (base marker)
        green_center = None
        if green_contours:
            # Filter contours by area and aspect ratio
            valid_green_contours = []
            for contour in green_contours:
                area = cv2.contourArea(contour)
                if area > 100:  # Minimum area threshold
                    # Check aspect ratio to filter out non-rectangular shapes
                    rect = cv2.minAreaRect(contour)
                    width, height = rect[1]
                    if width > 0 and height > 0:
                        aspect_ratio = max(width, height) / min(width, height)
                        if aspect_ratio < 3:  # Reasonable aspect ratio for marker
                            valid_green_contours.append(contour)
            
            if valid_green_contours:
                largest_green = max(valid_green_contours, key=cv2.contourArea)
                M = cv2.moments(largest_green)
                if M["m00"] != 0:
                    green_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        
        # Find best pink contour (direction marker) with improved filtering
        pink_center = None
        if pink_contours:
            # Filter and score pink contours
            scored_contours = []
            
            for contour in pink_contours:
                area = cv2.contourArea(contour)
                if area > 30:  # Lower minimum area for pink marker
                    # Calculate circularity (how round the contour is)
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        
                        # Calculate compactness
                        hull = cv2.convexHull(contour)
                        hull_area = cv2.contourArea(hull)
                        compactness = area / hull_area if hull_area > 0 else 0
                        
                        # Score based on area, circularity, and compactness
                        score = area * 0.1 + circularity * 100 + compactness * 50
                        
                        # Bonus for reasonable size (not too big, not too small)
                        if 30 < area < 500:
                            score += 20
                            
                        scored_contours.append((contour, score))
            
            if scored_contours:
                # Sort by score and take the best one
                scored_contours.sort(key=lambda x: x[1], reverse=True)
                best_contour = scored_contours[0][0]
                
                M = cv2.moments(best_contour)
                if M["m00"] != 0:
                    pink_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                    
                    # Additional validation: pink marker should be reasonably close to green marker
                    if green_center:
                        distance = math.hypot(pink_center[0] - green_center[0], 
                                           pink_center[1] - green_center[1])
                        # If pink marker is too far from green marker, ignore it
                        if distance > 200:  # Max pixel distance between markers
                            pink_center = None
        
        return green_center, pink_center

    def update_robot_position(self, frame=None):
        """Update robot position and heading based on visual markers"""
        # Capture fresh frame if none provided
        if frame is None:
            ret, frame = self.get_fresh_frame()
            if not ret:
                return False
        
        green_center, pink_center = self.detect_markers(frame)
        
        if green_center and pink_center and self.homography_matrix is not None:
            # Convert green center (robot base) to cm coordinates
            green_px = np.array([[[float(green_center[0]), float(green_center[1])]]], dtype="float32")
            green_cm = cv2.perspectiveTransform(green_px, np.linalg.inv(self.homography_matrix))[0][0]
            
            # Convert pink center (direction marker) to cm coordinates
            pink_px = np.array([[[float(pink_center[0]), float(pink_center[1])]]], dtype="float32")
            pink_cm = cv2.perspectiveTransform(pink_px, np.linalg.inv(self.homography_matrix))[0][0]
            
            # Update robot position (green marker center)
            self.robot_pos = (green_cm[0], green_cm[1])
            
            # Calculate heading from green to pink marker
            dx = pink_cm[0] - green_cm[0]
            dy = pink_cm[1] - green_cm[1]
            self.robot_heading = math.degrees(math.atan2(dy, dx))
            
            return True
        
        return False

    def draw_robot_markers(self, frame):
        """Draw detected robot markers on frame"""
        green_center, pink_center = self.detect_markers(frame)
        
        if green_center:
            # Draw green base marker
            cv2.circle(frame, green_center, 10, (0, 255, 0), -1)
            cv2.circle(frame, green_center, 12, (255, 255, 255), 2)
        
        if pink_center:
            # Draw pink direction marker
            cv2.circle(frame, pink_center, 8, (255, 192, 203), -1)
            cv2.circle(frame, pink_center, 10, (255, 255, 255), 2)
        
        if green_center and pink_center:
            # Draw line from base to direction marker
            cv2.line(frame, green_center, pink_center, (255, 255, 255), 2)
        
        return frame

    def show_detection_debug(self, frame):
        """Show debug view of marker detection masks"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create combined pink mask using all ranges
        pink_mask_combined = np.zeros_like(hsv[:,:,0])
        individual_masks = []
        
        for i, (pink_lower, pink_upper) in enumerate(PINK_RANGES):
            range_mask = cv2.inRange(hsv, pink_lower, pink_upper)
            individual_masks.append(range_mask)
            pink_mask_combined = cv2.bitwise_or(pink_mask_combined, range_mask)
        
        # Apply the same filtering as in detect_markers
        kernel_small = np.ones((2,2), np.uint8)
        kernel = np.ones((3,3), np.uint8)
        pink_mask_filtered = cv2.morphologyEx(pink_mask_combined, cv2.MORPH_OPEN, kernel_small)
        pink_mask_filtered = cv2.morphologyEx(pink_mask_filtered, cv2.MORPH_CLOSE, kernel)
        pink_mask_filtered = cv2.GaussianBlur(pink_mask_filtered, (3, 3), 0)
        
        # Create a visualization combining original frame and masks
        debug_frame = frame.copy()
        
        # Show pink detections as colored overlay
        pink_colored = cv2.applyColorMap(pink_mask_filtered, cv2.COLORMAP_HOT)
        debug_frame = cv2.addWeighted(debug_frame, 0.7, pink_colored, 0.3, 0)
        
        # Add text showing how many ranges detected something
        active_ranges = sum(1 for mask in individual_masks if cv2.countNonZero(mask) > 0)
        cv2.putText(debug_frame, f"Active pink ranges: {active_ranges}/{len(PINK_RANGES)}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show total pink pixels detected
        total_pink_pixels = cv2.countNonZero(pink_mask_filtered)
        cv2.putText(debug_frame, f"Pink pixels: {total_pink_pixels}", 
                   (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return debug_frame

    def setup_walls(self):
        """Set up wall segments based on calibration points, excluding goal areas"""
        if len(self.calibration_points) != 4:
            return

        # Calculate goal positions (in cm)
        goal_y_min = (FIELD_HEIGHT_CM / 2) - (GOAL_WIDTH / 2)
        goal_y_max = (FIELD_HEIGHT_CM / 2) + (GOAL_WIDTH / 2)

        # Create wall segments with small safety margin
        margin = WALL_SAFETY_MARGIN
        self.walls = [
            # Bottom wall (excluding goal)
            (margin, margin, FIELD_WIDTH_CM - margin, margin),
            
            # Top wall (excluding goal)
            (margin, FIELD_HEIGHT_CM - margin, FIELD_WIDTH_CM - margin, FIELD_HEIGHT_CM - margin),
            
            # Left wall (split into two parts to exclude goal)
            (margin, margin, margin, goal_y_min),
            (margin, goal_y_max, margin, FIELD_HEIGHT_CM - margin),
            
            # Right wall (split into two parts to exclude goal)
            (FIELD_WIDTH_CM - margin, margin, FIELD_WIDTH_CM - margin, goal_y_min),
            (FIELD_WIDTH_CM - margin, goal_y_max, FIELD_WIDTH_CM - margin, FIELD_HEIGHT_CM - margin)
        ]

    def draw_walls(self, frame):
        """Draw walls, safety margins, and starting box on the frame"""
        if not self.homography_matrix is None and self.walls:
           
            
            # Create a semi-transparent overlay for walls
            overlay = frame.copy()
            
            # Draw each wall segment
            for wall in self.walls:
                # Convert wall endpoints from cm to pixels
                start_cm = np.array([[[wall[0], wall[1]]]], dtype="float32")
                end_cm = np.array([[[wall[2], wall[3]]]], dtype="float32")
                
                start_px = cv2.perspectiveTransform(start_cm, self.homography_matrix)[0][0].astype(int)
                end_px = cv2.perspectiveTransform(end_cm, self.homography_matrix)[0][0].astype(int)
                
                # Draw the wall line
                cv2.line(overlay, tuple(start_px), tuple(end_px), (0, 0, 255), 2)
                
                # Draw safety margin area (semi-transparent)
                margin_pts = []
                angle = math.atan2(end_px[1] - start_px[1], end_px[0] - start_px[0])
                margin_px = int(WALL_SAFETY_MARGIN * 10)  # Convert cm to approximate pixels
                
                # Calculate margin points
                dx = int(margin_px * math.sin(angle))
                dy = int(margin_px * math.cos(angle))
                
                margin_pts = np.array([
                    [start_px[0] - dx, start_px[1] + dy],
                    [end_px[0] - dx, end_px[1] + dy],
                    [end_px[0] + dx, end_px[1] - dy],
                    [start_px[0] + dx, start_px[1] - dy]
                ], np.int32)
                
                # Draw safety margin area
                cv2.fillPoly(overlay, [margin_pts], (0, 0, 255, 128))
            
            # Blend the overlay with the original frame
            alpha = 0.3
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        return frame

    def send_command(self, command: str, **params) -> bool:
        """Send a command to the EV3 server"""
        try:
            with socket.create_connection((EV3_IP, EV3_PORT), timeout=15) as s:  # Increased timeout for longer movements
                # Prepare command
                message = {
                    "command": command,
                    **params
                }
                
                # Send command
                s.sendall((json.dumps(message) + "\n").encode("utf-8"))
                
                # Get response
                response = s.recv(1024).decode("utf-8").strip()
                return response == "OK"
                
        except socket.timeout:
            logger.error("Command timed out - robot might still be moving")
            return False
        except Exception as e:
            logger.error("Failed to send command: {}".format(e))
            return False

    def move(self, distance_cm: float) -> bool:
        """Move forward/backward by distance_cm (negative for backward)"""
        return self.send_command("MOVE", distance=distance_cm)

    def turn(self, angle_deg: float) -> bool:
        """Turn by angle_deg (positive = CCW)"""
        return self.send_command("TURN", angle=angle_deg)

    def collect(self, distance_cm: float) -> bool:
        """Move forward slowly while collecting"""
        return self.send_command("COLLECT", distance=distance_cm)

    def stop(self) -> bool:
        """Stop all motors"""
        return self.send_command("STOP")

    def calibrate(self):
        """Perform camera calibration by clicking 4 corners"""
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(self.calibration_points) < 4:
                    self.calibration_points.append((x, y))
                    logger.info(f"Corner {len(self.calibration_points)} set: ({x}, {y})")

                if len(self.calibration_points) == 4:
                    # Build homography matrix (cm) → (px)
                    dst_pts = np.array([
                        [0, 0],
                        [FIELD_WIDTH_CM, 0],
                        [FIELD_WIDTH_CM, FIELD_HEIGHT_CM],
                        [0, FIELD_HEIGHT_CM]
                    ], dtype="float32")
                    src_pts = np.array(self.calibration_points, dtype="float32")
                    self.homography_matrix = cv2.getPerspectiveTransform(dst_pts, src_pts)
                    
                    # Set up walls after calibration
                    self.setup_walls()
                    logger.info("✅ Calibration and wall setup complete")

        cv2.namedWindow("Calibration")
        cv2.setMouseCallback("Calibration", mouse_callback)
        
        while len(self.calibration_points) < 4:
            ret, frame = self.get_fresh_frame()
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

    def detect_balls(self) -> List[Tuple[float, float, str]]:
        """Detect balls in current camera view, return list of (x_cm, y_cm, label)"""
        ret, frame = self.get_fresh_frame()
        if not ret:
            return []

        # Resize for Roboflow model (optimized for our lower camera resolution)
        small = cv2.resize(frame, (416, 416))
        
        # Get predictions
        predictions = self.model.predict(small, confidence=30, overlap=20).json()
        
        balls = []
        scale_x = frame.shape[1] / 416
        scale_y = frame.shape[0] / 416
        
        # Target only white and orange balls
        target_ball_types = ['white_ball', 'orange_ball']
        
        for pred in predictions.get('predictions', []):
            ball_class = pred['class']
            
            # Only process white_ball and orange_ball
            if ball_class not in target_ball_types:
                continue
                
            x_px = int(pred['x'] * scale_x)
            y_px = int(pred['y'] * scale_y)
            
            # Convert to cm using homography
            if self.homography_matrix is not None:
                pt_px = np.array([[[x_px, y_px]]], dtype="float32")
                pt_cm = cv2.perspectiveTransform(pt_px, 
                                               np.linalg.inv(self.homography_matrix))[0][0]
                x_cm, y_cm = pt_cm
                
                # Check if ball is in ignored area
                if not (IGNORED_AREA["x_min"] <= x_cm <= IGNORED_AREA["x_max"] and
                       IGNORED_AREA["y_min"] <= y_cm <= IGNORED_AREA["y_max"]):
                    balls.append((x_cm, y_cm, ball_class))
        
        return balls

    def calculate_movement_efficiency(self, target_pos: Tuple[float, float]) -> dict:
        """Calculate efficiency scores for different movement approaches"""
        dx = target_pos[0] - self.robot_pos[0]
        dy = target_pos[1] - self.robot_pos[1]
        distance = math.hypot(dx, dy)
        target_angle = math.degrees(math.atan2(dy, dx))
        angle_diff = (target_angle - self.robot_heading + 180) % 360 - 180
        
        # Calculate efficiency for different approaches
        approaches = {}
        
        # 1. Direct forward approach
        turn_cost = abs(angle_diff) / 180.0  # Normalize to 0-1
        approaches['forward'] = {
            'total_cost': turn_cost + distance * 0.01,  # Small distance penalty
            'method': 'forward',
            'angle': target_angle,
            'backing': False
        }
        
        # 2. Backward movement (if target is behind us)
        backward_angle = target_angle + 180
        backward_angle_diff = (backward_angle - self.robot_heading + 180) % 360 - 180
        if abs(backward_angle_diff) < abs(angle_diff):  # Backing requires less turning
            approaches['backward'] = {
                'total_cost': abs(backward_angle_diff) / 180.0 + distance * 0.015,  # Slightly higher distance penalty
                'method': 'backward',
                'angle': backward_angle,
                'backing': True
            }
        
        # 3. Back away first, then forward (for nearby targets with sharp turns)
        if distance < 60 and abs(angle_diff) > 60:
            back_distance = min(30, distance * 0.5)
            back_angle = self.robot_heading + 180
            back_x = self.robot_pos[0] + back_distance * math.cos(math.radians(back_angle))
            back_y = self.robot_pos[1] + back_distance * math.sin(math.radians(back_angle))
            
            if not check_wall_collision(self.robot_pos, (back_x, back_y), self.walls, WALL_SAFETY_MARGIN):
                # Calculate new angle from backed position to target
                new_dx = target_pos[0] - back_x
                new_dy = target_pos[1] - back_y
                new_target_angle = math.degrees(math.atan2(new_dy, new_dx))
                new_angle_diff = (new_target_angle - back_angle + 180) % 360 - 180
                
                total_cost = (180 / 180.0) + (abs(new_angle_diff) / 180.0) + (back_distance + math.hypot(new_dx, new_dy)) * 0.01
                approaches['back_then_forward'] = {
                    'total_cost': total_cost,
                    'method': 'back_then_forward',
                    'back_pos': (back_x, back_y),
                    'angle': new_target_angle,
                    'backing': True
                }
        
        return approaches

    def get_smooth_approach_vector(self, target_pos: Tuple[float, float]) -> Tuple[Tuple[float, float], float, str]:
        """Calculate optimal approach using efficiency analysis"""
        approaches = self.calculate_movement_efficiency(target_pos)
        
        # Choose the most efficient approach
        best_approach = min(approaches.values(), key=lambda x: x['total_cost'])
        method = best_approach['method']
        
        dx = target_pos[0] - self.robot_pos[0]
        dy = target_pos[1] - self.robot_pos[1]
        distance = math.hypot(dx, dy)
        
        if method == 'backward':
            # Move backward toward target
            if distance > APPROACH_DISTANCE_CM:
                ratio = (distance - APPROACH_DISTANCE_CM) / distance
                approach_x = self.robot_pos[0] + dx * ratio
                approach_y = self.robot_pos[1] + dy * ratio
            else:
                approach_x, approach_y = self.robot_pos
            return (approach_x, approach_y), best_approach['angle'], 'backward'
            
        elif method == 'back_then_forward':
            # Return the backing position first
            return best_approach['back_pos'], self.robot_heading + 180, 'back_reposition'
            
        elif abs((best_approach['angle'] - self.robot_heading + 180) % 360 - 180) > 90:
            # Still a sharp turn - use curved approach
            curve_distance = min(40, distance * 0.6)
            angle_diff = (best_approach['angle'] - self.robot_heading + 180) % 360 - 180
            
            if angle_diff > 0:
                curve_angle = self.robot_heading + 45
            else:
                curve_angle = self.robot_heading - 45
            
            curve_x = self.robot_pos[0] + curve_distance * math.cos(math.radians(curve_angle))
            curve_y = self.robot_pos[1] + curve_distance * math.sin(math.radians(curve_angle))
            
            if not check_wall_collision(self.robot_pos, (curve_x, curve_y), self.walls, WALL_SAFETY_MARGIN):
                return (curve_x, curve_y), curve_angle, 'curve'
        
        # Standard forward approach
        if distance > APPROACH_DISTANCE_CM:
            ratio = (distance - APPROACH_DISTANCE_CM) / distance
            approach_x = self.robot_pos[0] + dx * ratio
            approach_y = self.robot_pos[1] + dy * ratio
        else:
            approach_x, approach_y = self.robot_pos
        
        return (approach_x, approach_y), best_approach['angle'], 'forward'

    def draw_path(self, frame, path):
        """Draw the planned path on the frame"""
        if not self.homography_matrix is None:
            overlay = frame.copy()
            
            # Colors for different point types
            colors = {
                'approach': (0, 255, 255),        # Yellow
                'collect': (0, 255, 0),           # Green
                'goal': (0, 0, 255),              # Red
                'backup': (255, 0, 255),          # Magenta
                'turn_to_ball': (128, 128, 255),  # Light purple
                'backward_approach': (255, 128, 0),  # Orange
                'backward_collect': (255, 165, 0),   # Orange red
                'curve_approach': (128, 255, 128)    # Light green
            }
            
            # Draw lines between points
            for i in range(len(path) - 1):
                # Convert cm to pixels for both points
                start_cm = np.array([[[path[i]['pos'][0], path[i]['pos'][1]]]], dtype="float32")
                end_cm = np.array([[[path[i+1]['pos'][0], path[i+1]['pos'][1]]]], dtype="float32")
                
                start_px = cv2.perspectiveTransform(start_cm, self.homography_matrix)[0][0].astype(int)
                end_px = cv2.perspectiveTransform(end_cm, self.homography_matrix)[0][0].astype(int)
                
                # Draw line
                cv2.line(overlay, tuple(start_px), tuple(end_px), (150, 150, 150), 2)
            
            # Draw points
            for point in path:
                pt_cm = np.array([[[point['pos'][0], point['pos'][1]]]], dtype="float32")
                pt_px = cv2.perspectiveTransform(pt_cm, self.homography_matrix)[0][0].astype(int)
                
                # Draw point
                color = colors[point['type']]
                cv2.circle(overlay, tuple(pt_px), 5, color, -1)
                
                # Draw direction for approach points
                if point['type'] in ['approach', 'goal']:
                    angle_rad = math.radians(point['angle'])
                    end_x = pt_px[0] + int(20 * math.cos(angle_rad))
                    end_y = pt_px[1] + int(20 * math.sin(angle_rad))
                    cv2.arrowedLine(overlay, tuple(pt_px), (end_x, end_y), color, 2)
                
                # Add labels
                if point['type'] == 'collect':
                    cv2.putText(overlay, point['ball_type'], 
                              (pt_px[0] + 10, pt_px[1] + 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Blend overlay with original frame
            alpha = 0.7
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            
            # Draw robot position and heading
            robot_cm = np.array([[[self.robot_pos[0], self.robot_pos[1]]]], dtype="float32")
            robot_px = cv2.perspectiveTransform(robot_cm, self.homography_matrix)[0][0].astype(int)
            
            # Robot circle
            cv2.circle(frame, tuple(robot_px), 8, (255, 0, 0), -1)
            
            # Robot heading
            angle_rad = math.radians(self.robot_heading)
            end_x = robot_px[0] + int(25 * math.cos(angle_rad))
            end_y = robot_px[1] + int(25 * math.sin(angle_rad))
            cv2.arrowedLine(frame, tuple(robot_px), (end_x, end_y), (255, 0, 0), 2)
        
        return frame

    def get_robot_status(self) -> dict:
        """Get current robot status"""
        try:
            with socket.create_connection((EV3_IP, EV3_PORT), timeout=5) as s:
                message = json.dumps({"command": "STATUS"}) + "\n"
                s.sendall(message.encode("utf-8"))
                response = s.recv(1024).decode("utf-8").strip()
                return json.loads(response)
        except Exception as e:
            logger.error(f"Failed to get robot status: {e}")
            return {}

    def draw_status(self, frame):
        """Draw robot status information on frame"""
        status = self.get_robot_status()
        if status:
            # Draw status box in top-left corner
            x, y = 10, 30
            line_height = 20
            
            # Background rectangle
            cv2.rectangle(frame, (5, 5), (200, 110), (0, 0, 0), -1)
            cv2.rectangle(frame, (5, 5), (200, 110), (255, 255, 255), 1)
            
            # Battery status
            battery_pct = status.get("battery_percentage", 0)
            battery_color = (0, 255, 0) if battery_pct > 20 else (0, 0, 255)
            cv2.putText(frame, f"Battery: {battery_pct:.1f}%",
                       (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, battery_color, 1)
            
            # Motor speeds
            left_speed = status.get("left_motor", {}).get("speed", 0)
            right_speed = status.get("right_motor", {}).get("speed", 0)
            collector_speed = status.get("collector_motor", {}).get("speed", 0)
            
            cv2.putText(frame, f"Left Motor: {left_speed}",
                       (x, y + line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Right Motor: {right_speed}",
                       (x, y + 2*line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Collector: {collector_speed}",
                       (x, y + 3*line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame

    def move_to_target_live(self, target_pos: Tuple[float, float], target_type: str = "approach", is_backing: bool = False) -> bool:
        """Move to target using continuous position updates and small adjustments"""
        # Different parameters based on movement type
        if target_type == "goal":
            max_distance_per_step = 30  # Larger steps for goal approach
            angle_tolerance = 25       # More lenient angle tolerance for long distance
            distance_tolerance = 10    # Goal can be reached with less precision
            heading_adjust_distance = 40  # Only adjust heading when within 40cm of goal
        else:
            max_distance_per_step = 10  # Smaller steps for precise ball collection
            angle_tolerance = 15       # Stricter angle tolerance
            distance_tolerance = 5     # Need to get closer for ball collection
            heading_adjust_distance = float('inf')  # Always adjust heading for ball collection
        
        step_count = 0
        max_steps = 50  # Prevent infinite loops
        
        while step_count < max_steps:
            # Update current position
            if not self.update_robot_position():
                logger.warning("Lost robot tracking during live movement")
                return False
            
            # Calculate distance and angle to target
            dx = target_pos[0] - self.robot_pos[0]
            dy = target_pos[1] - self.robot_pos[1]
            distance_to_target = math.hypot(dx, dy)
            target_angle = math.degrees(math.atan2(dy, dx))
            
            # Check if we've reached the target
            if distance_to_target <= distance_tolerance:
                logger.info(f"Reached target at {target_pos}")
                return True
            
            # Calculate angle difference
            angle_diff = (target_angle - self.robot_heading + 180) % 360 - 180
            
            # For backing moves, we want to face away from target
            if is_backing:
                target_angle = target_angle + 180  # Face opposite direction
                angle_diff = (target_angle - self.robot_heading + 180) % 360 - 180
            
            # Only adjust heading if within heading_adjust_distance or if angle is very wrong
            should_adjust_heading = (distance_to_target <= heading_adjust_distance or 
                                   abs(angle_diff) > 45)
            
            # If we need to turn significantly and should adjust heading
            if abs(angle_diff) > angle_tolerance and should_adjust_heading:
                # Limit turn to max 30 degrees per adjustment
                turn_amount = max(-30, min(30, angle_diff))
                action = "backing turn" if is_backing else "live adjustment"
                logger.info(f"{action}: turning {turn_amount:.1f} degrees (distance: {distance_to_target:.1f}cm)")
                if not self.turn(turn_amount):
                    return False
                self.update_robot_position()  # Update after turn
            else:
                # Move forward or backward in increments
                move_distance = min(max_distance_per_step, distance_to_target)
                
                if is_backing:
                    # Move backward
                    move_distance = -move_distance  # Negative for backward
                    logger.info(f"Backing up: {abs(move_distance):.1f} cm")
                    if not self.move(move_distance):
                        return False
                elif target_type == "collect" and distance_to_target <= COLLECTION_DISTANCE_CM:
                    # Final approach for collection
                    logger.info(f"Final collection approach: {move_distance:.1f} cm")
                    return self.collect(move_distance)
                else:
                    logger.info(f"Live movement ({target_type}): {move_distance:.1f} cm")
                    if not self.move(move_distance):
                        return False
                
                self.update_robot_position()  # Update after move
            
            step_count += 1
            
            # Brief pause to allow for processing
            time.sleep(0.1)
        
        logger.warning(f"Max steps reached trying to reach {target_pos}")
        return False

    def deliver_balls(self) -> bool:
        """Run collector in reverse to deliver balls"""
        try:
            return self.send_command("COLLECT_REVERSE", duration=self.delivery_time)
        except Exception as e:
            logger.error("Failed to deliver balls: {}".format(e))
            return False

    def calculate_goal_approach_path(self, current_pos, current_heading):
        """Calculate a smooth path to the goal using intermediate points"""
        goal_x = FIELD_WIDTH_CM - self.goal_approach_distance
        goal_y = self.goal_y_center
        
        # Calculate direct distance and angle to goal
        dx = goal_x - current_pos[0]
        dy = goal_y - current_pos[1]
        direct_distance = math.hypot(dx, dy)
        angle_to_goal = math.degrees(math.atan2(dy, dx))
        
        # If we're already well-aligned with the goal (within 15 degrees), just go straight
        angle_diff = (angle_to_goal - current_heading + 180) % 360 - 180
        if abs(angle_diff) < 15 and not check_wall_collision(current_pos, (goal_x, goal_y), self.walls, WALL_SAFETY_MARGIN):
            return [{
                'type': 'goal',
                'pos': (goal_x, goal_y),
                'angle': 0  # Face straight ahead at goal
            }]
        
        # Otherwise, calculate a curved approach path
        # First point: swing out to create better approach angle
        swing_distance = min(50, direct_distance * 0.4)  # Don't swing out too far
        
        # If we're to the left of the goal, swing right, and vice versa
        swing_direction = 1 if current_pos[1] < goal_y else -1
        swing_angle = current_heading + (45 * swing_direction)  # 45-degree swing
        
        # Calculate swing point
        swing_x = current_pos[0] + swing_distance * math.cos(math.radians(swing_angle))
        swing_y = current_pos[1] + swing_distance * math.sin(math.radians(swing_angle))
        
        # Second point: intermediate approach point
        approach_x = goal_x - (self.goal_approach_distance * 1.5)  # Further back from goal
        approach_y = goal_y  # Aligned with goal
        
        # Check if points are reachable
        path = []
        if not check_wall_collision(current_pos, (swing_x, swing_y), self.walls, WALL_SAFETY_MARGIN):
            path.append({
                'type': 'approach',
                'pos': (swing_x, swing_y),
                'angle': swing_angle
            })
        
        if not check_wall_collision((swing_x, swing_y), (approach_x, approach_y), self.walls, WALL_SAFETY_MARGIN):
            path.append({
                'type': 'approach',
                'pos': (approach_x, approach_y),
                'angle': 0  # Align with goal
            })
        
        if not check_wall_collision((approach_x, approach_y), (goal_x, goal_y), self.walls, WALL_SAFETY_MARGIN):
            path.append({
                'type': 'goal',
                'pos': (goal_x, goal_y),
                'angle': 0
            })
        
        return path

    def collect_balls(self):
        """Main ball collection loop"""
        cv2.namedWindow("Path Planning")
        last_plan_time = 0
        last_plan_positions = []
        path = []
        show_debug = False  # Toggle for showing detection debug view
        
        while True:
            try:
                # Get fresh frame with reduced lag
                ret, frame = self.get_fresh_frame()
                if not ret:
                    continue
                
                # Implement frame skipping for better performance
                self.frame_skip_counter += 1
                skip_heavy_processing = (self.frame_skip_counter % self.frame_skip_interval) != 0

                # Update robot position from visual markers
                if not self.update_robot_position(frame):
                    logger.warning("Could not detect robot markers")
                    cv2.putText(frame, "Robot markers not detected!", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.7, (0, 0, 255), 2)

                # Draw walls and safety margins
                frame = self.draw_walls(frame)

                # Draw robot markers
                frame = self.draw_robot_markers(frame)

                # Add status overlay
                frame = self.draw_status(frame)
                
                # Show debug view if enabled
                if show_debug:
                    frame = self.show_detection_debug(frame)

                # Detect balls (skip during heavy processing frames for better performance)
                balls = []
                if not skip_heavy_processing:
                    balls = self.detect_balls()
                current_time = time.time()
                
                # Only replan if:
                # 1. We have balls AND robot markers are detected AND
                # 2. Either:
                #    a) It's been at least 3 seconds since last plan OR
                #    b) Ball positions have changed significantly OR
                #    c) Robot has moved significantly
                should_replan = False
                if balls and self.robot_pos and not skip_heavy_processing:
                    if current_time - last_plan_time >= 3:  # Minimum 3 seconds between plans
                        should_replan = True
                    elif last_plan_positions:
                        # Check if any ball has moved more than 10cm
                        for ball in balls:
                            if not any(math.hypot(ball[0] - old[0], ball[1] - old[1]) < 10 
                                     for old in last_plan_positions):
                                should_replan = True
                                break
                    else:
                        should_replan = True

                if should_replan:
                    last_plan_time = current_time
                    last_plan_positions = [(b[0], b[1]) for b in balls]
                    logger.info(f"Planning to collect {len(balls)} balls from position {self.robot_pos}")

                    # Sort balls by distance from current robot position
                    balls.sort(key=lambda b: math.hypot(
                        b[0] - self.robot_pos[0],
                        b[1] - self.robot_pos[1]
                    ))

                    # Take up to MAX_BALLS_PER_TRIP closest balls
                    current_batch = balls[:MAX_BALLS_PER_TRIP]

                    # Plan path through all balls in the batch
                    path = []
                    current_pos = self.robot_pos
                    current_heading = self.robot_heading
                    remaining_balls = current_batch.copy()

                    # Add balls to path with smooth approach planning
                    while remaining_balls:
                        closest_ball = min(remaining_balls, 
                                         key=lambda b: math.hypot(
                                             b[0] - current_pos[0],
                                             b[1] - current_pos[1]
                                         ))
                        
                        ball_pos = (closest_ball[0], closest_ball[1])
                        approach_pos, target_angle, movement_type = self.get_smooth_approach_vector(ball_pos)
                        
                        # Check for wall collisions
                        if check_wall_collision(current_pos, approach_pos, self.walls, WALL_SAFETY_MARGIN):
                            logger.warning(f"Path to ball at {ball_pos} blocked by wall, skipping")
                            remaining_balls.remove(closest_ball)
                            continue
                        
                        if movement_type not in ['backward', 'back_reposition'] and check_wall_collision(approach_pos, ball_pos, self.walls, WALL_SAFETY_MARGIN):
                            logger.warning(f"Approach to ball at {ball_pos} blocked by wall, skipping")
                            remaining_balls.remove(closest_ball)
                            continue
                        
                        # Add appropriate waypoints based on movement type
                        if movement_type == 'backward':
                            path.append({
                                'type': 'backward_approach',
                                'pos': approach_pos,
                                'angle': target_angle,
                                'is_backing': True
                            })
                            path.append({
                                'type': 'backward_collect',
                                'pos': ball_pos,
                                'ball_type': closest_ball[2],
                                'is_backing': True
                            })
                        
                        elif movement_type == 'back_reposition':
                            path.append({
                                'type': 'backup',
                                'pos': approach_pos,
                                'angle': target_angle,
                                'is_backing': True
                            })
                            # After backing, turn to face the ball
                            face_ball_angle = math.degrees(math.atan2(
                                ball_pos[1] - approach_pos[1],
                                ball_pos[0] - approach_pos[0]
                            ))
                            path.append({
                                'type': 'turn_to_ball',
                                'pos': approach_pos,
                                'angle': face_ball_angle,
                                'target_ball': ball_pos
                            })
                            path.append({
                                'type': 'collect',
                                'pos': ball_pos,
                                'ball_type': closest_ball[2]
                            })
                        
                        elif movement_type == 'curve':
                            path.append({
                                'type': 'curve_approach',
                                'pos': approach_pos,
                                'angle': target_angle
                            })
                            # Add the final approach to ball
                            final_angle = math.degrees(math.atan2(
                                ball_pos[1] - approach_pos[1],
                                ball_pos[0] - approach_pos[0]
                            ))
                            path.append({
                                'type': 'approach',
                                'pos': ball_pos,
                                'angle': final_angle
                            })
                            path.append({
                                'type': 'collect',
                                'pos': ball_pos,
                                'ball_type': closest_ball[2]
                            })
                        
                        else:  # forward approach
                            path.append({
                                'type': 'approach',
                                'pos': approach_pos,
                                'angle': target_angle
                            })
                            path.append({
                                'type': 'collect',
                                'pos': ball_pos,
                                'ball_type': closest_ball[2]
                            })
                        
                        current_pos = ball_pos
                        current_heading = math.degrees(math.atan2(
                            ball_pos[1] - approach_pos[1],
                            ball_pos[0] - approach_pos[0]
                        ))
                        remaining_balls.remove(closest_ball)

                    # Add goal approach path if we have collected any balls
                    if path:
                        goal_path = self.calculate_goal_approach_path(current_pos, current_heading)
                        path.extend(goal_path)

                # Prepare the display frame
                display_frame = frame.copy()
                
                # Always show path preview when path exists
                if path:
                    display_frame = self.draw_path(display_frame, path)
                    # Add path ready indicator
                    cv2.putText(display_frame, "PATH READY - Press ENTER to execute", 
                              (10, display_frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.7, (0, 255, 0), 2)
                
                # Get key input
                key = cv2.waitKey(1) & 0xFF
                
                # Add key command help
                help_text = [
                    "Commands:",
                    "ENTER - Execute planned path",
                    "D - Toggle detection debug view",
                    "Q - Quit"
                ]
                y = 150
                for text in help_text:
                    cv2.putText(display_frame, text,
                              (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                              0.5, (255, 255, 255), 1)
                    y += 20
                
                # Show planned path count
                if path:
                    cv2.putText(display_frame, f"Path planned: {len([p for p in path if p['type'] == 'collect'])} balls",
                              (10, y + 10), cv2.FONT_HERSHEY_SIMPLEX,
                              0.5, (0, 255, 0), 1)
                
                cv2.imshow("Path Planning", display_frame)
                
                # Handle key commands
                if key == ord('q'):
                    break
                elif key == ord('d'):
                    show_debug = not show_debug
                    logger.info(f"Debug view {'enabled' if show_debug else 'disabled'}")
                elif key == 13:  # Enter key
                    if path:
                        # Execute the planned path
                        logger.info("Executing path with {} points".format(len(path)))
                        
                        try:
                            for point in path:
                                target_pos = point['pos']
                                
                                if point['type'] == 'collect':
                                    logger.info(f"Live approach to collect {point['ball_type']} ball")
                                    if not self.move_to_target_live(target_pos, "collect"):
                                        raise Exception("Live collection movement failed")
                                elif point['type'] == 'goal':
                                    logger.info(f"Live movement to goal")
                                    if not self.move_to_target_live(target_pos, "goal"):
                                        raise Exception("Live goal movement failed")
                                    # After reaching goal position, deliver balls
                                    if not self.deliver_balls():
                                        raise Exception("Ball delivery failed")
                                elif point['type'] == 'backup':
                                    logger.info(f"Backing up to improve approach angle")
                                    if not self.move_to_target_live(target_pos, "approach", is_backing=True):
                                        raise Exception("Backup movement failed")
                                elif point['type'] == 'backward_approach':
                                    logger.info(f"Moving backward to approach ball")
                                    if not self.move_to_target_live(target_pos, "approach", is_backing=True):
                                        raise Exception("Backward approach movement failed")
                                elif point['type'] == 'backward_collect':
                                    logger.info(f"Collecting {point['ball_type']} ball while moving backward")
                                    # For backward collection, we need to get close first then collect
                                    dx = target_pos[0] - self.robot_pos[0]
                                    dy = target_pos[1] - self.robot_pos[1]
                                    distance_to_ball = math.hypot(dx, dy)
                                    
                                    if distance_to_ball > COLLECTION_DISTANCE_CM:
                                        # Move backward to get closer
                                        move_distance = -(distance_to_ball - COLLECTION_DISTANCE_CM)
                                        if not self.move(move_distance):
                                            raise Exception("Backward movement to ball failed")
                                        self.update_robot_position()
                                    
                                    # Now collect while facing away from ball
                                    if not self.collect(COLLECTION_DISTANCE_CM):
                                        raise Exception("Backward collection failed")
                                elif point['type'] == 'curve_approach':
                                    logger.info(f"Moving through curve point for smooth approach")
                                    if not self.move_to_target_live(target_pos, "approach"):
                                        raise Exception("Curve approach movement failed")
                                elif point['type'] == 'turn_to_ball':
                                    logger.info(f"Turning to face ball after backup")
                                    # Calculate turn to face the target ball
                                    ball_pos = point['target_ball']
                                    dx = ball_pos[0] - self.robot_pos[0]
                                    dy = ball_pos[1] - self.robot_pos[1]
                                    target_angle = math.degrees(math.atan2(dy, dx))
                                    angle_diff = (target_angle - self.robot_heading + 180) % 360 - 180
                                    
                                    if abs(angle_diff) > 5:
                                        logger.info(f"Turning {angle_diff:.1f} degrees to face ball")
                                        if not self.turn(angle_diff):
                                            raise Exception("Turn to ball failed")
                                    self.update_robot_position()
                                else:  # approach point
                                    logger.info(f"Live movement to approach point")
                                    if not self.move_to_target_live(target_pos, "approach"):
                                        raise Exception("Live approach movement failed")
                                
                                # Update visualization during execution
                                ret, frame = self.get_fresh_frame()
                                if ret:
                                    self.update_robot_position(frame)
                                    frame = self.draw_status(frame)
                                    frame = self.draw_robot_markers(frame)
                                    frame_with_path = self.draw_path(frame, path)
                                    cv2.putText(frame_with_path, "EXECUTING PATH LIVE...", 
                                              (10, frame_with_path.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                              0.7, (0, 255, 0), 2)
                                    cv2.imshow("Path Planning", frame_with_path)
                                    cv2.waitKey(1)
                            
                            # Clear path after successful execution
                            path = []
                            logger.info("Path execution completed successfully")
                            cv2.waitKey(1000)
                            
                        except Exception as e:
                            logger.error("Path execution failed: {}".format(e))
                            self.stop()
                            cv2.waitKey(2000)
                    else:
                        logger.warning("No path planned to execute")
                    
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                cv2.waitKey(1000)

        cv2.destroyWindow("Path Planning")

    def run(self):
        """Main run loop"""
        try:
            # Calibrate the camera perspective
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