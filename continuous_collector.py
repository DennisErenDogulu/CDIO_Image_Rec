#!/usr/bin/env python3
"""
Continuous Ball Collection Client

This script implements a continuous ball collection strategy using visual markers:
1. Track robot position using green base marker
2. Track robot heading using pink direction marker
3. Detect balls using enhanced multi-method detection optimized for 1.8m aerial view
4. Navigate to and collect balls based on visual position tracking

Note: All detection parameters are optimized for a camera positioned 1.8 meters above the field
"""

import cv2
import math
import json
import socket
import logging
import numpy as np
from typing import List, Tuple, Optional
import time

# Configuration
EV3_IP = "172.20.10.6"
EV3_PORT = 12345

# Color detection ranges (HSV)
# Green paper detection - Optimized for bright green paper
GREEN_LOWER = np.array([40, 50, 50])    # Green hue range: 40-80, moderate saturation and value
GREEN_UPPER = np.array([80, 255, 255])
PINK_LOWER = np.array([145, 50, 50])
PINK_UPPER = np.array([175, 255, 255])

# Ball color detection ranges (HSV) - Optimized for white ball detection
# Orange detection disabled for now
# ORANGE_BALL_LOWER = np.array([8, 80, 40])    
# ORANGE_BALL_UPPER = np.array([22, 255, 255]) 

# Enhanced white ball detection ranges - multiple approaches
WHITE_BALL_LOWER_1 = np.array([0, 0, 140])   # Main range - bright whites, very low saturation
WHITE_BALL_UPPER_1 = np.array([180, 20, 255])
WHITE_BALL_LOWER_2 = np.array([0, 0, 100])   # Shadow range - medium brightness
WHITE_BALL_UPPER_2 = np.array([180, 35, 200])
WHITE_BALL_LOWER_3 = np.array([0, 0, 170])   # Highlight range - very bright, ultra low saturation
WHITE_BALL_UPPER_3 = np.array([180, 12, 255])
WHITE_BALL_LOWER_4 = np.array([0, 0, 80])    # Dark shadow range
WHITE_BALL_UPPER_4 = np.array([180, 50, 160])

# Ball colors mapping - Enhanced white ball detection
BALL_COLORS = {
    'white_main': (WHITE_BALL_LOWER_1, WHITE_BALL_UPPER_1),
    'white_shadow': (WHITE_BALL_LOWER_2, WHITE_BALL_UPPER_2),
    'white_bright': (WHITE_BALL_LOWER_3, WHITE_BALL_UPPER_3),
    'white_dark': (WHITE_BALL_LOWER_4, WHITE_BALL_UPPER_4)
}

# Marker dimensions
GREEN_MARKER_WIDTH_CM = 20  # Width of green base sheet (back to green!)
PINK_MARKER_WIDTH_CM = 5    # Width of pink direction marker

# Wall configuration
WALL_SAFETY_MARGIN = 1  # cm, minimum distance to keep from walls

# Goal configuration
SMALL_GOAL_SIDE = "right"  # "left" or "right" - where the small goal (A) is placed
GOAL_OFFSET_CM = 10  # cm, distance to stop before goal edge

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

# Detection parameters - Optimized for 1.8m height aerial view
MIN_BALL_AREA = 15   # Much smaller for aerial view (was 40)
MIN_ROUNDNESS = 0.4  # More lenient for small distant objects
MIN_WALL_AREA = 500  # Minimum contour area for wall detection
MIN_GOAL_AREA = 1000  # Minimum contour area for goal detection

# Advanced detection parameters - Adjusted for 1.8m aerial perspective
BALL_SIZE_RANGE = (3, 25)      # Much smaller pixel size from 1.8m height
BRIGHTNESS_PERCENTILE = 85      # Percentile threshold for brightness detection
HOUGH_CIRCLE_PARAMS = {         # Parameters optimized for aerial view
    'dp': 1,
    'minDist': 15,              # Smaller minimum distance between circles
    'param1': 40,               # Lower edge threshold for aerial view
    'param2': 18,               # Lower accumulator threshold for small objects
    'minRadius': 2,             # Much smaller minimum radius from height
    'maxRadius': 15             # Much smaller maximum radius from height
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
        # Initialize camera
        self.cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # Use camera index 1 for USB camera
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Robot state
        self.robot_pos = (ROBOT_START_X, ROBOT_START_Y)  # Starting position
        self.robot_heading = ROBOT_START_HEADING  # Starting heading
        
        # Calibration points for homography
        self.calibration_points = []
        self.homography_matrix = None
        
        # Wall configuration
        self.walls = []  # Will be set after calibration
        
        # Goal system
        self.small_goal_side = SMALL_GOAL_SIDE.lower()  # "left" or "right"
        self.selected_goal = 'A'  # 'A' (small goal) or 'B' (large goal)
        self.goal_ranges = self._build_goal_ranges()
        self.delivery_time = 2.0  # Seconds to run collector in reverse
        
        # Smart movement settings
        self.backward_driving_enabled = True  # Enable smart backward driving

    def is_point_in_field(self, x_cm: float, y_cm: float) -> bool:
        """Check if a point in cm coordinates is within the calibrated field bounds"""
        return (0 <= x_cm <= FIELD_WIDTH_CM and 0 <= y_cm <= FIELD_HEIGHT_CM)

    def _build_goal_ranges(self) -> dict:
        """
        Build goal ranges for both goals A and B based on small_goal_side setting.
        
        - Goal A (small goal): 8cm high, centered at 60cm vertically (Y: 56-64cm)
        - Goal B (large goal): 20cm high, centered at 60cm vertically (Y: 50-70cm)
        
        If small_goal_side == "left": A is on left edge (x=0), B on right edge (x=180cm)
        If small_goal_side == "right": A is on right edge (x=180cm), B on left edge (x=0)
        """
        ranges = {}
        
        # Y intervals for goals
        y_min_A = 56  # 60 - 4
        y_max_A = 64  # 60 + 4
        y_min_B = 50  # 60 - 10  
        y_max_B = 70  # 60 + 10
        
        if self.small_goal_side == "left":
            # Goal A (small) on left edge, Goal B (large) on right edge
            ranges['A'] = [(0, y_cm) for y_cm in range(y_min_A, y_max_A + 1)]
            ranges['B'] = [(FIELD_WIDTH_CM, y_cm) for y_cm in range(y_min_B, y_max_B + 1)]
        else:
            # Goal A (small) on right edge, Goal B (large) on left edge  
            ranges['A'] = [(FIELD_WIDTH_CM, y_cm) for y_cm in range(y_min_A, y_max_A + 1)]
            ranges['B'] = [(0, y_cm) for y_cm in range(y_min_B, y_max_B + 1)]
            
        return ranges

    def detect_markers(self, frame):
        """Detect green base paper and pink direction marker"""
        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Detect green base marker (paper) - use morphological operations to clean up
        green_mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
        # Clean up the mask for better paper detection
        kernel = np.ones((5,5), np.uint8)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)  # Fill holes
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)   # Remove noise
        
        green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Detect pink direction marker (clip)
        pink_mask = cv2.inRange(hsv, PINK_LOWER, PINK_UPPER)
        # Clean up pink detection
        kernel_small = np.ones((3,3), np.uint8)
        pink_mask = cv2.morphologyEx(pink_mask, cv2.MORPH_CLOSE, kernel_small)
        pink_mask = cv2.morphologyEx(pink_mask, cv2.MORPH_OPEN, kernel_small)
        
        pink_contours, _ = cv2.findContours(pink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find largest green contour (base paper) - lower threshold for larger areas
        green_center = None
        if green_contours:
            largest_green = max(green_contours, key=cv2.contourArea)
            green_area = cv2.contourArea(largest_green)
            if green_area > 500:  # Increased threshold for paper (was 100)
                # Use moments to find center of mass
                M = cv2.moments(largest_green)
                if M["m00"] != 0:
                    green_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        
        # Find largest pink contour (direction clip) - keep smaller threshold
        pink_center = None
        if pink_contours:
            largest_pink = max(pink_contours, key=cv2.contourArea)
            pink_area = cv2.contourArea(largest_pink)
            if pink_area > 20:  # Lower threshold for small clip (was 50)
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
            
            # Update robot position (green marker center)
            self.robot_pos = (green_cm[0], green_cm[1])
            
            # Calculate heading from green to pink marker (front direction)
            dx = pink_cm[0] - green_cm[0]
            dy = pink_cm[1] - green_cm[1]
            # Convert to degrees and normalize to [-180, 180]
            heading_raw = math.degrees(math.atan2(dy, dx))
            self.robot_heading = (heading_raw + 180) % 360 - 180
            
            return True
        
        return False

    def draw_robot_markers(self, frame):
        """Draw detected robot markers on frame with debugging info"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Get the masks for debugging
        green_mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
        kernel = np.ones((5,5), np.uint8)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
        green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        pink_mask = cv2.inRange(hsv, PINK_LOWER, PINK_UPPER)
        kernel_small = np.ones((3,3), np.uint8)
        pink_mask = cv2.morphologyEx(pink_mask, cv2.MORPH_CLOSE, kernel_small)
        pink_mask = cv2.morphologyEx(pink_mask, cv2.MORPH_OPEN, kernel_small)
        pink_contours, _ = cv2.findContours(pink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        green_center, pink_center = self.detect_markers(frame)
        
        # Draw all green contours for debugging
        if green_contours:
            largest_green = max(green_contours, key=cv2.contourArea)
            green_area = cv2.contourArea(largest_green)
            
            # Draw the green paper contour
            cv2.drawContours(frame, [largest_green], -1, (0, 255, 0), 2)  # Green color in BGR
            
            # Show area info
            cv2.putText(frame, f"Green area: {green_area:.0f}", 
                      (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            if green_center:
                # Draw center point
                cv2.circle(frame, green_center, 8, (0, 255, 0), -1)
                cv2.circle(frame, green_center, 12, (255, 255, 255), 2)
                cv2.putText(frame, "GREEN CENTER", 
                          (green_center[0] + 15, green_center[1] - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw all pink contours for debugging
        if pink_contours:
            largest_pink = max(pink_contours, key=cv2.contourArea)
            pink_area = cv2.contourArea(largest_pink)
            
            # Draw the pink clip contour
            cv2.drawContours(frame, [largest_pink], -1, (255, 192, 203), 2)
            
            # Show area info
            cv2.putText(frame, f"Pink area: {pink_area:.0f}", 
                      (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 192, 203), 2)
            
            if pink_center:
                # Draw center point
                cv2.circle(frame, pink_center, 6, (255, 192, 203), -1)
                cv2.circle(frame, pink_center, 10, (255, 255, 255), 2)
                cv2.putText(frame, "PINK DIRECTION", 
                          (pink_center[0] + 15, pink_center[1] - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 192, 203), 2)
        
        # Draw heading line if both markers detected
        if green_center and pink_center:
            # Draw thick heading line
            cv2.arrowedLine(frame, green_center, pink_center, (0, 255, 255), 4)  # Yellow arrow
            
            # Show distance between markers
            distance = math.hypot(pink_center[0] - green_center[0], pink_center[1] - green_center[1])
            cv2.putText(frame, f"Marker distance: {distance:.1f}px", 
                      (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return frame

    def setup_walls(self):
        """Set up wall segments based on calibration points, excluding goal areas"""
        if len(self.calibration_points) != 4:
            return

        # Create wall segments with small safety margin
        margin = WALL_SAFETY_MARGIN
        
        # Get goal Y ranges for both goals
        goal_a_y_vals = [y for (x, y) in self.goal_ranges['A']]
        goal_b_y_vals = [y for (x, y) in self.goal_ranges['B']]
        
        goal_a_y_min = min(goal_a_y_vals)
        goal_a_y_max = max(goal_a_y_vals)
        goal_b_y_min = min(goal_b_y_vals)
        goal_b_y_max = max(goal_b_y_vals)

        self.walls = [
            # Bottom wall (full width)
            (margin, margin, FIELD_WIDTH_CM - margin, margin),
            
            # Top wall (full width)
            (margin, FIELD_HEIGHT_CM - margin, FIELD_WIDTH_CM - margin, FIELD_HEIGHT_CM - margin),
        ]
        
        # Left wall segments (excluding goals)
        if self.small_goal_side == "left":
            # Goal A on left, so exclude its Y range
            if goal_a_y_min > margin:
                self.walls.append((margin, margin, margin, goal_a_y_min))
            if goal_a_y_max < FIELD_HEIGHT_CM - margin:
                self.walls.append((margin, goal_a_y_max, margin, FIELD_HEIGHT_CM - margin))
        else:
            # Goal B on left, so exclude its Y range
            if goal_b_y_min > margin:
                self.walls.append((margin, margin, margin, goal_b_y_min))
            if goal_b_y_max < FIELD_HEIGHT_CM - margin:
                self.walls.append((margin, goal_b_y_max, margin, FIELD_HEIGHT_CM - margin))
        
        # Right wall segments (excluding goals)
        if self.small_goal_side == "right":
            # Goal A on right, so exclude its Y range
            if goal_a_y_min > margin:
                self.walls.append((FIELD_WIDTH_CM - margin, margin, FIELD_WIDTH_CM - margin, goal_a_y_min))
            if goal_a_y_max < FIELD_HEIGHT_CM - margin:
                self.walls.append((FIELD_WIDTH_CM - margin, goal_a_y_max, FIELD_WIDTH_CM - margin, FIELD_HEIGHT_CM - margin))
        else:
            # Goal B on right, so exclude its Y range
            if goal_b_y_min > margin:
                self.walls.append((FIELD_WIDTH_CM - margin, margin, FIELD_WIDTH_CM - margin, goal_b_y_min))
            if goal_b_y_max < FIELD_HEIGHT_CM - margin:
                self.walls.append((FIELD_WIDTH_CM - margin, goal_b_y_max, FIELD_WIDTH_CM - margin, FIELD_HEIGHT_CM - margin))

    def draw_walls(self, frame):
        """Draw walls, safety margins, goals, and starting box on the frame"""
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
            
            # Draw goal areas
            for goal_label, goal_points in self.goal_ranges.items():
                # Color based on goal type and selection
                if goal_label == 'A':  # Small goal
                    if goal_label == self.selected_goal:
                        goal_color = (0, 0, 139)  # Magenta for selected small goal
                    else:
                        goal_color = (255, 0, 128)  # Pink for available small goal
                else:  # Goal B (large goal)
                    if goal_label == self.selected_goal:
                        goal_color = (0, 255, 0)  # Green for selected large goal
                    else:
                        goal_color = (0, 255, 255)  # Yellow for available large goal
                
                # Draw goal points
                for (x_cm, y_cm) in goal_points:
                    pt_cm = np.array([[[x_cm, y_cm]]], dtype="float32")
                    pt_px = cv2.perspectiveTransform(pt_cm, self.homography_matrix)[0][0].astype(int)
                    cv2.circle(overlay, tuple(pt_px), 4, goal_color, -1)
                
                # Draw goal label
                if goal_points:
                    mid_y = sum(y for (x, y) in goal_points) / len(goal_points)
                    label_x = goal_points[0][0]
                    
                    # Offset label position based on which side goal is on
                    if label_x == 0:  # Left side
                        label_x += 10
                    else:  # Right side
                        label_x -= 15
                    
                    label_cm = np.array([[[label_x, mid_y]]], dtype="float32")
                    label_px = cv2.perspectiveTransform(label_cm, self.homography_matrix)[0][0].astype(int)
                    cv2.putText(overlay, f"Goal {goal_label}", tuple(label_px),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, goal_color, 2)
            
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
        """Move forward/backward by distance_cm"""
        if distance_cm < 0:
            # Use explicit backward command for negative distances
            logger.info(f"🔄 Using MOVE_BACKWARD command for {abs(distance_cm):.1f}cm")
            return self.send_command("MOVE_BACKWARD", distance=abs(distance_cm))
        else:
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
                    logger.info("✅ Calibration complete")
                    
                    # Setup walls and goals after calibration
                    self.setup_walls()
                    logger.info("✅ Walls and goals initialized")

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

    def detect_balls(self, frame=None) -> List[Tuple[float, float, str]]:
        """Enhanced ball detection using multiple methods for robust white ball detection"""
        if frame is None:
            ret, frame = self.cap.read()
            if not ret:
                return []
        
        # Only detect if we have calibration
        if self.homography_matrix is None:
            return []
        
        # === ENHANCED PREPROCESSING PIPELINE ===
        
        # 1. Advanced denoising while preserving edges
        frame_denoised = cv2.fastNlMeansDenoisingColored(frame, None, 6, 6, 7, 21)
        
        # 2. Adaptive brightness enhancement
        lab = cv2.cvtColor(frame_denoised, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        lab[:, :, 0] = l_channel
        frame_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # 3. Adaptive gamma correction based on local statistics
        gray = cv2.cvtColor(frame_enhanced, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        
        # Adjust gamma based on image statistics
        if mean_brightness < 80:
            gamma = 0.7  # Brighten dark images
        elif mean_brightness > 180:
            gamma = 1.4  # Darken bright images
        else:
            gamma = 1.0  # Normal images
        
        # Apply gamma correction if needed
        if gamma != 1.0:
            frame_enhanced = np.power(frame_enhanced / 255.0, gamma)
            frame_enhanced = (frame_enhanced * 255).astype(np.uint8)
        
        # 4. Additional white balance correction for changing lighting
        frame_enhanced = self._apply_white_balance_correction(frame_enhanced)
        
        # 5. Convert to multiple color spaces for comprehensive detection
        hsv = cv2.cvtColor(frame_enhanced, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(frame_enhanced, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(frame_enhanced, cv2.COLOR_BGR2GRAY)
        
        # === MULTI-METHOD DETECTION WITH ADAPTIVE WEIGHTING ===
        
        # Method 1: Enhanced HSV color detection
        hsv_mask = self._detect_balls_hsv(hsv)
        
        # Method 2: Brightness-based detection
        brightness_mask = self._detect_balls_brightness(gray, lab)
        
        # Method 3: Enhanced Hough circles detection (now primary method)
        edge_mask = self._detect_balls_edges(gray)
        
        # === ADAPTIVE LIGHTING COMPENSATION ===
        
        # Analyze lighting conditions to adjust detection strategy
        lighting_condition = self._analyze_lighting_conditions(gray)
        
        # Combine methods with adaptive weighting based on lighting
        if lighting_condition == "low_light":
            # In low light, prioritize edge detection and brightness methods
            combined_mask = cv2.addWeighted(edge_mask, 0.6, brightness_mask, 0.3, 0)
            combined_mask = cv2.addWeighted(combined_mask, 1.0, hsv_mask, 0.2, 0)
        elif lighting_condition == "high_contrast":
            # In high contrast, prioritize HSV and edge detection
            combined_mask = cv2.addWeighted(edge_mask, 0.5, hsv_mask, 0.4, 0)
            combined_mask = cv2.addWeighted(combined_mask, 1.0, brightness_mask, 0.2, 0)
        elif lighting_condition == "changing_light":
            # In changing light, give highest weight to enhanced Hough circles
            combined_mask = cv2.addWeighted(edge_mask, 0.7, brightness_mask, 0.2, 0)
            combined_mask = cv2.addWeighted(combined_mask, 1.0, hsv_mask, 0.2, 0)
        else:  # normal lighting
            # Balanced approach with slight preference for enhanced Hough circles
            combined_mask = cv2.addWeighted(edge_mask, 0.5, hsv_mask, 0.3, 0)
            combined_mask = cv2.addWeighted(combined_mask, 1.0, brightness_mask, 0.3, 0)
        
        # Ensure the result is binary
        _, combined_mask = cv2.threshold(combined_mask, 127, 255, cv2.THRESH_BINARY)
        
        # === ADVANCED MORPHOLOGICAL PROCESSING ===
        
        # Adaptive kernel size based on image resolution
        base_kernel_size = max(3, min(7, int(frame.shape[0] / 200)))
        kernel_small = np.ones((base_kernel_size, base_kernel_size), np.uint8)
        kernel_medium = np.ones((base_kernel_size + 2, base_kernel_size + 2), np.uint8)
        
        # Progressive morphological operations
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_medium, iterations=2)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
        
        # Distance transform for better separation
        dist_transform = cv2.distanceTransform(combined_mask, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
        combined_mask = np.uint8(sure_fg)
        
        # === CONTOUR DETECTION AND VALIDATION ===
        
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        balls = []
        
        for contour in contours:
            ball_candidate = self._validate_ball_contour(contour, frame_enhanced, gray)
            if ball_candidate:
                x_px, y_px = ball_candidate
                
                # Convert to cm using homography
                pt_px = np.array([[[x_px, y_px]]], dtype="float32")
                pt_cm = cv2.perspectiveTransform(pt_px, np.linalg.inv(self.homography_matrix))[0][0]
                x_cm, y_cm = pt_cm
                
                # Final validation
                if self._final_ball_validation(x_cm, y_cm, balls):
                    balls.append((x_cm, y_cm, 'white'))
        
        return balls
    
    def _detect_balls_hsv(self, hsv):
        """HSV-based ball detection with weighted combination"""
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        
        # Weights for different color ranges (based on reliability)
        weights = {'white_main': 1.0, 'white_bright': 0.8, 'white_shadow': 0.6, 'white_dark': 0.4}
        
        for color_name, (lower, upper) in BALL_COLORS.items():
            mask = cv2.inRange(hsv, lower, upper)
            
            # Apply Gaussian blur to soften the mask
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            
            # Weight the mask based on reliability
            weight = weights.get(color_name, 0.5)
            mask = (mask * weight).astype(np.uint8)
            
            combined_mask = cv2.add(combined_mask, mask)
        
        # Threshold the weighted combination
        _, combined_mask = cv2.threshold(combined_mask, 128, 255, cv2.THRESH_BINARY)
        
        return combined_mask
    
    def _detect_balls_brightness(self, gray, lab):
        """Brightness and luminance-based detection for white objects"""
        # Method 1: High brightness detection in grayscale
        _, bright_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Method 2: High luminance in LAB space
        l_channel = lab[:, :, 0]
        _, lab_mask = cv2.threshold(l_channel, np.percentile(l_channel, 85), 255, cv2.THRESH_BINARY)
        
        # Method 3: Adaptive threshold for local brightness peaks
        adaptive_mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY, 15, -5)
        
        # Combine brightness-based masks
        brightness_mask = cv2.bitwise_or(bright_mask, lab_mask)
        brightness_mask = cv2.bitwise_and(brightness_mask, adaptive_mask)
        
        return brightness_mask
    
    def _analyze_lighting_conditions(self, gray):
        """Analyze the lighting conditions of the image to adapt detection strategy"""
        # Calculate basic statistics
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        
        # Calculate histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_norm = hist.flatten() / hist.sum()
        
        # Check for low light conditions
        if mean_brightness < 70:
            return "low_light"
        
        # Check for high contrast (wide brightness distribution)
        if std_brightness > 60:
            # Additional check: bimodal distribution indicates high contrast
            peaks = []
            for i in range(1, 255):
                if hist_norm[i] > hist_norm[i-1] and hist_norm[i] > hist_norm[i+1]:
                    if hist_norm[i] > 0.01:  # Significant peak
                        peaks.append(i)
            
            if len(peaks) >= 2 and abs(peaks[-1] - peaks[0]) > 100:
                return "high_contrast"
        
        # Check for changing light conditions (uneven brightness distribution)
        # Divide image into regions and check brightness variation
        h, w = gray.shape
        regions = []
        for i in range(3):
            for j in range(3):
                y1, y2 = i * h // 3, (i + 1) * h // 3
                x1, x2 = j * w // 3, (j + 1) * w // 3
                region_mean = np.mean(gray[y1:y2, x1:x2])
                regions.append(region_mean)
        
        region_std = np.std(regions)
        if region_std > 30:  # High variation across regions
            return "changing_light"
        
        # Default to normal lighting
        return "normal"
    
    def _apply_white_balance_correction(self, frame):
        """Apply automatic white balance correction to normalize for changing lighting conditions"""
        # Convert to LAB color space for better white balance
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        
        # Method 1: Gray World assumption
        gray_world = frame.copy().astype(np.float32)
        b_mean = np.mean(gray_world[:, :, 0])
        g_mean = np.mean(gray_world[:, :, 1])
        r_mean = np.mean(gray_world[:, :, 2])
        
        # Calculate scaling factors
        gray_mean = (b_mean + g_mean + r_mean) / 3
        
        if b_mean > 0 and g_mean > 0 and r_mean > 0:
            b_scale = gray_mean / b_mean
            g_scale = gray_mean / g_mean
            r_scale = gray_mean / r_mean
            
            # Apply gentle correction (not too aggressive)
            b_scale = 0.7 + 0.3 * b_scale  # Limit correction strength
            g_scale = 0.7 + 0.3 * g_scale
            r_scale = 0.7 + 0.3 * r_scale
            
            gray_world[:, :, 0] *= b_scale
            gray_world[:, :, 1] *= g_scale
            gray_world[:, :, 2] *= r_scale
            
            gray_world = np.clip(gray_world, 0, 255).astype(np.uint8)
        else:
            gray_world = frame
        
        # Method 2: White patch correction
        # Find brightest regions (likely to be white/light colored)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bright_threshold = np.percentile(gray, 95)
        bright_mask = gray > bright_threshold
        
        if np.sum(bright_mask) > 100:  # Enough bright pixels
            white_patch = frame.copy().astype(np.float32)
            
            # Calculate average color of bright regions
            bright_b = np.mean(frame[:, :, 0][bright_mask])
            bright_g = np.mean(frame[:, :, 1][bright_mask])
            bright_r = np.mean(frame[:, :, 2][bright_mask])
            
            # Target white point (slightly bluish to account for typical indoor lighting)
            target_white = 240
            
            if bright_b > 0 and bright_g > 0 and bright_r > 0:
                b_factor = target_white / bright_b
                g_factor = target_white / bright_g
                r_factor = target_white / bright_r
                
                # Limit correction factors to avoid over-correction
                b_factor = np.clip(b_factor, 0.5, 2.0)
                g_factor = np.clip(g_factor, 0.5, 2.0)
                r_factor = np.clip(r_factor, 0.5, 2.0)
                
                white_patch[:, :, 0] *= b_factor
                white_patch[:, :, 1] *= g_factor
                white_patch[:, :, 2] *= r_factor
                
                white_patch = np.clip(white_patch, 0, 255).astype(np.uint8)
            else:
                white_patch = frame
        else:
            white_patch = frame
        
        # Combine both methods with adaptive weighting
        alpha = 0.6  # Weight for gray world method
        beta = 0.4   # Weight for white patch method
        
        combined = cv2.addWeighted(gray_world, alpha, white_patch, beta, 0)
        
        # Final gentle enhancement for white objects
        lab_corrected = cv2.cvtColor(combined, cv2.COLOR_BGR2LAB)
        l_channel = lab_corrected[:, :, 0]
        
        # Slightly boost brightness in the upper range (where white balls would be)
        l_enhanced = l_channel.copy().astype(np.float32)
        bright_regions = l_enhanced > 150
        l_enhanced[bright_regions] = l_enhanced[bright_regions] * 1.1
        l_enhanced = np.clip(l_enhanced, 0, 255).astype(np.uint8)
        
        lab_corrected[:, :, 0] = l_enhanced
        final_corrected = cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)
        
        return final_corrected
    
    def _detect_balls_edges(self, gray):
        """Enhanced edge-based circular object detection with multiple Hough circle approaches"""
        edge_masks = []
        
        # === PREPROCESSING IMPROVEMENTS ===
        
        # Method 1: Standard preprocessing with bilateral filtering
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        blurred1 = cv2.GaussianBlur(bilateral, (5, 5), 1)
        
        # Method 2: Histogram equalization for better contrast
        equalized = cv2.equalizeHist(gray)
        blurred2 = cv2.GaussianBlur(equalized, (7, 7), 2)
        
        # Method 3: CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_enhanced = clahe.apply(gray)
        blurred3 = cv2.GaussianBlur(clahe_enhanced, (9, 9), 2)
        
        # Method 4: Edge enhancement using unsharp masking
        gaussian = cv2.GaussianBlur(gray, (0, 0), 2.0)
        unsharp_mask = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)
        blurred4 = cv2.GaussianBlur(unsharp_mask, (5, 5), 1)
        
        preprocessed_images = [
            (blurred1, "bilateral"),
            (blurred2, "equalized"), 
            (blurred3, "clahe"),
            (blurred4, "unsharp")
        ]
        
        # === MULTIPLE HOUGH CIRCLE PARAMETER SETS ===
        
        # Parameter sets optimized for 1.8m aerial view and different lighting conditions
        hough_params = [
            # Standard detection - optimized for 1.8m height
            {'dp': 1, 'minDist': 12, 'param1': 40, 'param2': 18, 'minRadius': 2, 'maxRadius': 15},
            
            # Sensitive detection - for low contrast from height
            {'dp': 1, 'minDist': 8, 'param1': 25, 'param2': 12, 'minRadius': 2, 'maxRadius': 18},
            
            # Conservative detection - for noisy aerial conditions
            {'dp': 1, 'minDist': 15, 'param1': 50, 'param2': 22, 'minRadius': 3, 'maxRadius': 12},
            
            # Ultra-sensitive - for very small/distant balls
            {'dp': 1, 'minDist': 6, 'param1': 20, 'param2': 8, 'minRadius': 1, 'maxRadius': 10},
            
            # High precision - for clearly visible balls from height
            {'dp': 1, 'minDist': 10, 'param1': 35, 'param2': 15, 'minRadius': 2, 'maxRadius': 20}
        ]
        
        all_circles = []
        
        # Apply each preprocessing method with each parameter set
        for img, img_type in preprocessed_images:
            for i, params in enumerate(hough_params):
                try:
                    circles = cv2.HoughCircles(
                        img, 
                        cv2.HOUGH_GRADIENT,
                        dp=params['dp'],
                        minDist=params['minDist'],
                        param1=params['param1'],
                        param2=params['param2'],
                        minRadius=params['minRadius'],
                        maxRadius=params['maxRadius']
                    )
                    
                    if circles is not None:
                        circles = np.round(circles[0, :]).astype("int")
                        for (x, y, r) in circles:
                            # Store circle with confidence score based on method
                            confidence = self._calculate_circle_confidence(gray, x, y, r, img_type, i)
                            all_circles.append((x, y, r, confidence, img_type, i))
                            
                except Exception as e:
                    # Continue if this parameter set fails
                    continue
        
        # === CIRCLE VALIDATION AND FILTERING ===
        
        # Remove duplicate circles using non-maximum suppression
        filtered_circles = self._non_maximum_suppression_circles(all_circles)
        
        # Additional validation for white ball characteristics
        validated_circles = []
        for (x, y, r, confidence, method, param_idx) in filtered_circles:
            if self._validate_circle_as_ball(gray, x, y, r):
                validated_circles.append((x, y, r, confidence))
        
        # === CREATE FINAL MASK ===
        
        edge_mask = np.zeros(gray.shape, dtype=np.uint8)
        
        # Sort circles by confidence and take the best ones
        validated_circles.sort(key=lambda x: x[3], reverse=True)
        max_circles = min(10, len(validated_circles))  # Limit to avoid false positives
        
        for i in range(max_circles):
            x, y, r, confidence = validated_circles[i]
            # Lower confidence threshold for small aerial objects
            if confidence > 0.2:  # More lenient for 1.8m height detection
                # Draw filled circle with radius slightly smaller than detected
                adjusted_radius = max(2, int(r * 0.8))
                cv2.circle(edge_mask, (x, y), adjusted_radius, 255, -1)
        
        return edge_mask
    
    def _calculate_circle_confidence(self, gray, x, y, r, method_type, param_idx):
        """Calculate confidence score for detected circle based on white ball characteristics"""
        if x - r < 0 or y - r < 0 or x + r >= gray.shape[1] or y + r >= gray.shape[0]:
            return 0.0
        
        # Extract circular region
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(mask, (x, y), r, 255, -1)
        
        # Get pixels within the circle
        circle_pixels = gray[mask > 0]
        if len(circle_pixels) == 0:
            return 0.0
        
        # Get surrounding pixels for comparison
        outer_mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(outer_mask, (x, y), int(r * 1.5), 255, -1)
        cv2.circle(outer_mask, (x, y), r, 0, -1)  # Remove inner circle
        surrounding_pixels = gray[outer_mask > 0]
        
        confidence = 0.0
        
        # Brightness score - white balls should be bright
        avg_brightness = np.mean(circle_pixels)
        max_brightness = np.max(circle_pixels)
        brightness_score = min(1.0, avg_brightness / 200.0) * 0.3
        brightness_score += min(1.0, max_brightness / 255.0) * 0.2
        confidence += brightness_score
        
        # Contrast score - ball should be brighter than surroundings
        if len(surrounding_pixels) > 0:
            contrast = avg_brightness - np.mean(surrounding_pixels)
            contrast_score = min(1.0, max(0.0, contrast / 50.0)) * 0.2
            confidence += contrast_score
        
        # Uniformity score - ball should have relatively uniform brightness
        brightness_std = np.std(circle_pixels)
        max_allowed_std = 30 + (r * 2)  # Allow more variation for larger circles
        uniformity_score = max(0.0, 1.0 - brightness_std / max_allowed_std) * 0.15
        confidence += uniformity_score
        
        # Size score - adjusted for 1.8m aerial view
        ideal_radius = 6  # Much smaller for aerial perspective (was 15)
        size_diff = abs(r - ideal_radius)
        size_score = max(0.0, 1.0 - size_diff / 8.0) * 0.1  # Tighter tolerance for small objects
        confidence += size_score
        
        # Method bonus - some methods are more reliable
        method_bonuses = {
            "bilateral": 0.05,
            "clahe": 0.03,
            "equalized": 0.02,
            "unsharp": 0.04
        }
        confidence += method_bonuses.get(method_type, 0.0)
        
        # Edge strength score
        roi = gray[max(0, y-r):min(gray.shape[0], y+r+1), 
                   max(0, x-r):min(gray.shape[1], x+r+1)]
        if roi.size > 0:
            edges = cv2.Canny(roi, 50, 150)
            edge_pixels = np.sum(edges > 0)
            expected_edge_pixels = 2 * np.pi * r * 0.7  # Approximate circle perimeter
            edge_score = min(1.0, edge_pixels / max(1, expected_edge_pixels)) * 0.05
            confidence += edge_score
        
        return min(1.0, confidence)
    
    def _non_maximum_suppression_circles(self, circles):
        """Remove overlapping circles, keeping the ones with highest confidence"""
        if len(circles) == 0:
            return []
        
        # Sort by confidence
        circles = sorted(circles, key=lambda x: x[3], reverse=True)
        
        kept_circles = []
        for circle in circles:
            x, y, r, conf, method, param_idx = circle
            
            # Check if this circle overlaps significantly with any kept circle
            should_keep = True
            for kept_x, kept_y, kept_r, kept_conf, kept_method, kept_param in kept_circles:
                distance = np.sqrt((x - kept_x)**2 + (y - kept_y)**2)
                overlap_threshold = min(r, kept_r) * 0.7
                
                if distance < overlap_threshold:
                    should_keep = False
                    break
            
            if should_keep:
                kept_circles.append(circle)
        
        return kept_circles
    
    def _validate_circle_as_ball(self, gray, x, y, r):
        """Additional validation to ensure detected circle represents a ball"""
        # Check bounds
        if x - r < 0 or y - r < 0 or x + r >= gray.shape[1] or y + r >= gray.shape[0]:
            return False
        
        # Size constraints - adjusted for 1.8m height
        if r < 1 or r > 20:  # Much smaller range for aerial view
            return False
        
        # Extract circular region
        roi = gray[y-r:y+r+1, x-r:x+r+1]
        if roi.size == 0:
            return False
        
        # Create circular mask for the ROI
        mask = np.zeros(roi.shape, dtype=np.uint8)
        cv2.circle(mask, (r, r), r, 255, -1)
        
        # Get pixels within the circle
        circle_pixels = roi[mask > 0]
        if len(circle_pixels) < 10:  # Too few pixels
            return False
        
        # Brightness validation - adjusted for aerial perspective
        avg_brightness = np.mean(circle_pixels)
        if avg_brightness < 50:  # Lower threshold for distant white balls
            return False
        
        # Check for reasonable brightness distribution
        brightness_std = np.std(circle_pixels)
        if brightness_std > 60:  # Too much variation
            return False
        
        # Gradient check - balls should have smooth gradients
        gradx = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
        grady = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(gradx**2 + grady**2)
        
        # Check if gradients are reasonable (not too high)
        avg_gradient = np.mean(magnitude[mask > 0])
        if avg_gradient > 50:  # Too much texture/noise
            return False
        
        return True
    
    def _validate_ball_contour(self, contour, frame_enhanced, gray):
        """Comprehensive validation of ball contours"""
        # Basic area check - adjusted for 1.8m aerial perspective
        area = cv2.contourArea(contour)
        if area < MIN_BALL_AREA or area > 400:  # Much smaller upper limit for aerial view
            return None
        
        # Enhanced circularity check
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return None
        
        circularity = 4 * math.pi * area / (perimeter * perimeter)
        if circularity < 0.5:  # More lenient for better detection
            return None
        
        # Bounding rectangle analysis
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio < 0.7 or aspect_ratio > 1.4:  # More lenient aspect ratio
            return None
        
        # Size consistency - adjusted for aerial view
        avg_size = (w + h) / 2
        if avg_size < 3 or avg_size > 25:  # Much smaller range for 1.8m height
            return None
        
        # Minimum enclosing circle analysis
        (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
        circle_area = math.pi * radius * radius
        area_ratio = area / circle_area if circle_area > 0 else 0
        if area_ratio < 0.5:  # More lenient area ratio
            return None
        
        # Convexity check - balls should be mostly convex
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        convexity = area / hull_area if hull_area > 0 else 0
        if convexity < 0.8:  # Should be quite convex
            return None
        
        # Enhanced brightness validation
        roi = frame_enhanced[y:y+h, x:x+w]
        if roi.size > 0:
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Create mask for the contour within ROI
            contour_shifted = contour - [x, y]  # Shift contour to ROI coordinates
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [contour_shifted], 255)
            
            # Calculate statistics only within the contour
            masked_pixels = roi_gray[mask > 0]
            if len(masked_pixels) == 0:
                return None
            
            avg_brightness = np.mean(masked_pixels)
            max_brightness = np.max(masked_pixels)
            brightness_std = np.std(masked_pixels)
            
            # Adaptive brightness thresholds - adjusted for aerial perspective
            min_avg_brightness = 60 if np.mean(gray) < 120 else 80  # Lower thresholds for distant objects
            min_max_brightness = 90 if np.mean(gray) < 120 else 120
            
            if avg_brightness < min_avg_brightness or max_brightness < min_max_brightness:
                return None
            
            # Check for reasonable brightness variation (not too flat)
            if brightness_std < 5:
                return None
        
        # Get center of mass for more accurate center
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return None
        
        center_x = M["m10"] / M["m00"]
        center_y = M["m01"] / M["m00"]
        
        return (center_x, center_y)
    
    def _final_ball_validation(self, x_cm, y_cm, existing_balls):
        """Final validation checks for ball position"""
        # Check if ball is within the calibrated field bounds
        if not self.is_point_in_field(x_cm, y_cm):
            return False
        
        # Check if ball is in ignored area (center obstacle)
        if (IGNORED_AREA["x_min"] <= x_cm <= IGNORED_AREA["x_max"] and
           IGNORED_AREA["y_min"] <= y_cm <= IGNORED_AREA["y_max"]):
            return False
        
        # Avoid duplicate detections with adaptive distance threshold
        min_distance = 4  # Slightly increased for better duplicate removal
        for existing_x, existing_y, _ in existing_balls:
            if math.hypot(x_cm - existing_x, y_cm - existing_y) < min_distance:
                return False
        
        return True

    def draw_detected_balls(self, frame, balls):
        """Draw detected balls on the frame with enhanced visualization"""
        ball_colors_bgr = {
            'orange': (0, 140, 255),  # Orange in BGR
            'white': (255, 255, 255)  # White in BGR
        }
        
        for i, (x_cm, y_cm, color_name) in enumerate(balls):
            if self.homography_matrix is not None:
                # Convert cm position back to pixels for drawing
                pt_cm = np.array([[[x_cm, y_cm]]], dtype="float32")
                pt_px = cv2.perspectiveTransform(pt_cm, self.homography_matrix)[0][0].astype(int)
                
                # Draw ball with enhanced visualization
                bgr_color = ball_colors_bgr.get(color_name, (255, 255, 255))
                
                # Main ball circle
                cv2.circle(frame, tuple(pt_px), 15, bgr_color, -1)
                cv2.circle(frame, tuple(pt_px), 17, (0, 255, 0), 3)  # Green outline
                
                # Ball number
                cv2.putText(frame, str(i+1), 
                          (pt_px[0] - 5, pt_px[1] + 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                
                # Position info
                cv2.putText(frame, f"({x_cm:.1f},{y_cm:.1f})", 
                          (pt_px[0] + 20, pt_px[1] - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Distance from robot
                if hasattr(self, 'robot_pos') and self.robot_pos:
                    distance = math.hypot(x_cm - self.robot_pos[0], y_cm - self.robot_pos[1])
                    cv2.putText(frame, f"{distance:.1f}cm", 
                              (pt_px[0] + 20, pt_px[1] + 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        return frame
    
    def show_detection_debug(self, frame):
        """Show debug visualization of the detection process"""
        if self.homography_matrix is None:
            return frame
        
        # Get enhanced frame
        frame_denoised = cv2.fastNlMeansDenoisingColored(frame, None, 6, 6, 7, 21)
        lab = cv2.cvtColor(frame_denoised, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        lab[:, :, 0] = l_channel
        frame_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Convert to color spaces
        hsv = cv2.cvtColor(frame_enhanced, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame_enhanced, cv2.COLOR_BGR2GRAY)
        
        # Get detection masks
        hsv_mask = self._detect_balls_hsv(hsv)
        brightness_mask = self._detect_balls_brightness(gray, lab)
        edge_mask = self._detect_balls_edges(gray)
        
        # Create debug visualization
        debug_frame = np.zeros((frame.shape[0] * 2, frame.shape[1] * 2, 3), dtype=np.uint8)
        
        # Top-left: Original enhanced frame
        debug_frame[0:frame.shape[0], 0:frame.shape[1]] = frame_enhanced
        
        # Top-right: HSV mask
        hsv_colored = cv2.applyColorMap(hsv_mask, cv2.COLORMAP_JET)
        debug_frame[0:frame.shape[0], frame.shape[1]:frame.shape[1]*2] = hsv_colored
        
        # Bottom-left: Brightness mask
        brightness_colored = cv2.applyColorMap(brightness_mask, cv2.COLORMAP_HOT)
        debug_frame[frame.shape[0]:frame.shape[0]*2, 0:frame.shape[1]] = brightness_colored
        
        # Bottom-right: Edge mask
        edge_colored = cv2.applyColorMap(edge_mask, cv2.COLORMAP_COOL)
        debug_frame[frame.shape[0]:frame.shape[0]*2, frame.shape[1]:frame.shape[1]*2] = edge_colored
        
        # Add labels
        cv2.putText(debug_frame, "Enhanced Frame", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(debug_frame, "HSV Detection", (frame.shape[1] + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(debug_frame, "Brightness Detection", (10, frame.shape[0] + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(debug_frame, "Edge Detection", (frame.shape[1] + 10, frame.shape[0] + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return debug_frame

    def get_approach_vector(self, target_pos: Tuple[float, float]) -> Tuple[Tuple[float, float], float]:
        """Calculate approach position and angle for a target position"""
        # Get vector from robot to target
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
        """Draw the planned path on the frame with enhanced preview"""
        if not self.homography_matrix is None:
            overlay = frame.copy()
            
            # Colors for different point types
            colors = {
                'approach': (0, 255, 255),  # Yellow
                'collect': (0, 255, 0),     # Green
                'goal': (0, 0, 255),        # Red
                'backward': (255, 0, 255)   # Magenta for backward movements
            }
            
            # Draw grid overlay (light grid lines every 10cm)
            grid_color = (100, 100, 100)
            for x_cm in range(0, FIELD_WIDTH_CM + 1, 10):
                start_cm = np.array([[[x_cm, 0]]], dtype="float32")
                end_cm = np.array([[[x_cm, FIELD_HEIGHT_CM]]], dtype="float32")
                start_px = cv2.perspectiveTransform(start_cm, self.homography_matrix)[0][0].astype(int)
                end_px = cv2.perspectiveTransform(end_cm, self.homography_matrix)[0][0].astype(int)
                cv2.line(overlay, tuple(start_px), tuple(end_px), grid_color, 1)
            
            for y_cm in range(0, FIELD_HEIGHT_CM + 1, 10):
                start_cm = np.array([[[0, y_cm]]], dtype="float32")
                end_cm = np.array([[[FIELD_WIDTH_CM, y_cm]]], dtype="float32")
                start_px = cv2.perspectiveTransform(start_cm, self.homography_matrix)[0][0].astype(int)
                end_px = cv2.perspectiveTransform(end_cm, self.homography_matrix)[0][0].astype(int)
                cv2.line(overlay, tuple(start_px), tuple(end_px), grid_color, 1)
            
            # Convert path to grid for display
            grid_path, ball_cells = self.convert_path_to_grid(path)
            
            # Draw thick path lines between consecutive points
            for i in range(len(path) - 1):
                start_cm = np.array([[[path[i]['pos'][0], path[i]['pos'][1]]]], dtype="float32")
                end_cm = np.array([[[path[i+1]['pos'][0], path[i+1]['pos'][1]]]], dtype="float32")
                
                start_px = cv2.perspectiveTransform(start_cm, self.homography_matrix)[0][0].astype(int)
                end_px = cv2.perspectiveTransform(end_cm, self.homography_matrix)[0][0].astype(int)
                
                # Draw thick path line
                cv2.line(overlay, tuple(start_px), tuple(end_px), (255, 255, 255), 4)
                cv2.line(overlay, tuple(start_px), tuple(end_px), (150, 150, 150), 2)
            
            # Draw waypoints with numbers
            for i, point in enumerate(path):
                pt_cm = np.array([[[point['pos'][0], point['pos'][1]]]], dtype="float32")
                pt_px = cv2.perspectiveTransform(pt_cm, self.homography_matrix)[0][0].astype(int)
                
                # Draw point with larger circle
                color = colors[point['type']]
                cv2.circle(overlay, tuple(pt_px), 8, color, -1)
                cv2.circle(overlay, tuple(pt_px), 10, (255, 255, 255), 2)
                
                # Add waypoint number
                cv2.putText(overlay, str(i+1), 
                          (pt_px[0] - 5, pt_px[1] + 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Draw direction arrows for approach points
                if point['type'] in ['approach', 'goal']:
                    angle_rad = math.radians(point['angle'])
                    end_x = pt_px[0] + int(25 * math.cos(angle_rad))
                    end_y = pt_px[1] + int(25 * math.sin(angle_rad))
                    cv2.arrowedLine(overlay, tuple(pt_px), (end_x, end_y), color, 3)
                
                # Add grid coordinates
                if i < len(grid_path):
                    gx, gy = grid_path[i]
                    grid_text = f"({gx},{gy})"
                    cv2.putText(overlay, grid_text, 
                              (pt_px[0] + 15, pt_px[1] - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Draw path statistics
            total_distance = 0
            for i in range(len(path) - 1):
                dx = path[i+1]['pos'][0] - path[i]['pos'][0]
                dy = path[i+1]['pos'][1] - path[i]['pos'][1]
                total_distance += math.hypot(dx, dy)
            
            # Path info overlay
            info_y = 30
            cv2.putText(overlay, f"Path: {len(path)} waypoints", 
                      (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(overlay, f"Distance: {total_distance:.1f}cm", 
                      (10, info_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(overlay, f"Balls: {len(ball_cells)}", 
                      (10, info_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(overlay, f"Grid points: {len(grid_path)}", 
                      (10, info_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Blend overlay with original frame
            alpha = 0.8
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            
            # Draw robot position and heading
            robot_cm = np.array([[[self.robot_pos[0], self.robot_pos[1]]]], dtype="float32")
            robot_px = cv2.perspectiveTransform(robot_cm, self.homography_matrix)[0][0].astype(int)
            
            # Robot circle
            cv2.circle(frame, tuple(robot_px), 10, (255, 0, 0), -1)
            cv2.circle(frame, tuple(robot_px), 12, (255, 255, 255), 2)
            
            # Robot heading
            angle_rad = math.radians(self.robot_heading)
            end_x = robot_px[0] + int(30 * math.cos(angle_rad))
            end_y = robot_px[1] + int(30 * math.sin(angle_rad))
            cv2.arrowedLine(frame, tuple(robot_px), (end_x, end_y), (255, 0, 0), 3)
        
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
        
        # Draw status box in top-left corner
        x, y = 10, 30
        line_height = 20
        
        # Background rectangle (make it taller for goal info)
        cv2.rectangle(frame, (5, 5), (220, 150), (0, 0, 0), -1)
        cv2.rectangle(frame, (5, 5), (220, 150), (255, 255, 255), 1)
        
        # Goal status (always show)
        goal_type = "small" if self.selected_goal == 'A' else "large"
        cv2.putText(frame, f"Selected: Goal {self.selected_goal} ({goal_type})",
                   (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(frame, f"Small goal side: {self.small_goal_side}",
                   (x, y + line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        if status:
            # Battery status
            battery_pct = status.get("battery_percentage", 0)
            battery_color = (0, 255, 0) if battery_pct > 20 else (0, 0, 255)
            cv2.putText(frame, f"Battery: {battery_pct:.1f}%",
                       (x, y + 2*line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, battery_color, 1)
            
            # Motor speeds
            left_speed = status.get("left_motor", {}).get("speed", 0)
            right_speed = status.get("right_motor", {}).get("speed", 0)
            collector_speed = status.get("collector_motor", {}).get("speed", 0)
            
            cv2.putText(frame, f"Left Motor: {left_speed}",
                       (x, y + 3*line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Right Motor: {right_speed}",
                       (x, y + 4*line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Collector: {collector_speed}",
                       (x, y + 5*line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame

    def deliver_balls(self) -> bool:
        """Run collector in reverse to deliver balls"""
        try:
            return self.send_command("COLLECT_REVERSE", duration=self.delivery_time)
        except Exception as e:
            logger.error("Failed to deliver balls: {}".format(e))
            return False

    def should_drive_backward(self, target_pos: Tuple[float, float], distance: float) -> Tuple[bool, float]:
        """
        Determine if it's more efficient to drive backward instead of turning around.
        
        Returns:
            (should_reverse, backward_angle): Whether to reverse and the angle for backward driving
        """
        # Get current position and heading
        current_pos = self.robot_pos
        current_heading = self.robot_heading
        
        # Calculate angle to target (forward direction)
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        forward_angle = math.degrees(math.atan2(dy, dx))
        forward_angle_diff = abs((forward_angle - current_heading + 180) % 360 - 180)
        
        # Calculate angle for backward driving (opposite direction)
        backward_angle = (forward_angle + 180) % 360
        backward_angle_diff = abs((backward_angle - current_heading + 180) % 360 - 180)
        
        # Scenarios where backward driving makes sense:
        should_reverse = False
        reason = ""
        
        # 1. Short distances (<20cm) and large turn angle (>90°)
        if distance < 20 and forward_angle_diff > 90:
            if backward_angle_diff < forward_angle_diff - 30:  # At least 30° advantage
                should_reverse = True
                reason = f"short distance ({distance:.1f}cm) with large turn angle"
        
        # 2. Backward angle is significantly smaller (>60° difference)
        elif backward_angle_diff < forward_angle_diff - 60:
            should_reverse = True
            reason = f"much smaller turn angle ({backward_angle_diff:.1f}° vs {forward_angle_diff:.1f}°)"
        
        # 3. Very close to 180° turn (between 150° and 210°)
        elif 150 <= forward_angle_diff <= 210:
            should_reverse = True
            reason = f"near-180° turn ({forward_angle_diff:.1f}°) - backing up instead"
        
        # 4. Robot is facing almost opposite direction and distance is reasonable
        elif forward_angle_diff > 135 and distance < 50:
            should_reverse = True
            reason = f"facing opposite direction with reasonable distance"
        
        # Additional safety checks for backward driving
        if should_reverse and self.backward_driving_enabled:
            # Check if backward path is clear
            # Calculate backward endpoint
            backward_distance = min(distance, 30)  # Limit backward distance to 30cm max
            back_dx = -backward_distance * math.cos(math.radians(current_heading))
            back_dy = -backward_distance * math.sin(math.radians(current_heading))
            back_end_pos = (current_pos[0] + back_dx, current_pos[1] + back_dy)
            
            # Check for wall collisions on backward path
            if check_wall_collision(current_pos, back_end_pos, self.walls, WALL_SAFETY_MARGIN + 2):
                should_reverse = False
                reason = "backward path blocked by walls"
            # Check if we'd go out of bounds
            elif not self.is_point_in_field(back_end_pos[0], back_end_pos[1]):
                should_reverse = False
                reason = "backward path goes out of field"
        elif should_reverse and not self.backward_driving_enabled:
            should_reverse = False
            reason = "backward driving disabled"
        
        if should_reverse:
            logger.info(f"🔄 Backward driving recommended: {reason}")
            return True, backward_angle
        else:
            return False, forward_angle
    
    def execute_smart_movement(self, target_pos: Tuple[float, float], movement_type: str = "approach") -> bool:
        """
        Execute movement to target with smart forward/backward decision making.
        
        Args:
            target_pos: Target position (x, y) in cm
            movement_type: Type of movement ("approach", "goal", "collect")
        
        Returns:
            True if movement was successful
        """
        current_pos = self.robot_pos
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        distance = math.hypot(dx, dy)
        
        # Skip if we're already very close
        if distance < 2:
            logger.info(f"Already at target position (distance: {distance:.1f}cm)")
            return True
        
        # Determine if we should drive backward
        should_reverse, target_angle = self.should_drive_backward(target_pos, distance)
        
        if should_reverse:
            # Backward driving strategy
            
            # First, align for backward driving (face away from target)
            current_heading = self.robot_heading
            backward_heading = (target_angle + 180) % 360  # Face opposite to target
            turn_angle = (backward_heading - current_heading + 180) % 360 - 180
            
            if abs(turn_angle) > 5:
                logger.info(f"🔄 Turning {turn_angle:.1f}° to align for backward driving")
                if not self.turn(turn_angle):
                    logger.error("❌ Turn for backward alignment failed")
                    return False
                self.robot_heading = backward_heading
            
            # Drive backward (negative distance)
            max_backward_distance = min(distance, 25)  # Limit backward driving
            logger.info(f"⬅️ Driving backward {max_backward_distance:.1f}cm")
            if not self.move(-max_backward_distance):
                logger.error("❌ Backward movement failed")
                return False
            
            # If we didn't reach the target, turn and move forward for the rest
            remaining_distance = distance - max_backward_distance
            if remaining_distance > 2:
                # Turn to face target
                forward_angle = math.degrees(math.atan2(dy, dx))
                final_turn = (forward_angle - self.robot_heading + 180) % 360 - 180
                
                if abs(final_turn) > 5:
                    logger.info(f"🔄 Final turn {final_turn:.1f}° to face target")
                    if not self.turn(final_turn):
                        logger.error("❌ Final turn failed")
                        return False
                    self.robot_heading = forward_angle
                
                # Move remaining distance forward
                logger.info(f"➡️ Moving remaining {remaining_distance:.1f}cm forward")
                if not self.move(remaining_distance):
                    logger.error("❌ Final forward movement failed")
                    return False
        
        else:
            # Standard forward driving strategy
            
            # Calculate angle to target
            target_angle = math.degrees(math.atan2(dy, dx))
            angle_diff = (target_angle - self.robot_heading + 180) % 360 - 180
            
            # Turn to face target if significant difference
            if abs(angle_diff) > 5:
                logger.info(f"🔄 Turning {angle_diff:.1f}° to face target")
                if not self.turn(angle_diff):
                    logger.error("❌ Turn command failed")
                    return False
                self.robot_heading = target_angle
            
            # Move forward to target
            logger.info(f"➡️ Moving {distance:.1f}cm to {movement_type} position")
            if not self.move(distance):
                logger.error(f"❌ {movement_type} movement failed")
                return False
        
        return True

    def convert_path_to_grid(self, path):
        """Convert continuous path to grid coordinates for client_automated.py"""
        if not path:
            return [], []
        
        grid_path = []
        ball_cells = []
        
        # Convert each waypoint to grid coordinates
        for point in path:
            x_cm, y_cm = point['pos']
            # Convert cm to grid coordinates (2cm spacing)
            gx = int(round(x_cm / 2))
            gy = int(round(y_cm / 2))
            
            grid_path.append([gx, gy])
            
            # Mark ball collection points
            if point['type'] == 'collect':
                ball_cells.append([gx, gy])
        
        return grid_path, ball_cells

    def execute_autonomous_collection(self, initial_path=None):
        """Execute true live-tracking autonomous ball collection - continuously replans based on current conditions"""
        logger.info("🤖 Starting TRUE live-tracking autonomous collection...")
        
        collected_balls = 0
        max_balls_per_session = 10  # Prevent infinite loops
        
        try:
            while collected_balls < max_balls_per_session:
                # Get fresh frame and update robot position
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Failed to get camera frame")
                    time.sleep(0.1)
                    continue
                
                # Update robot position from visual markers
                max_retries = 5
                retry_count = 0
                
                while retry_count < max_retries:
                    if self.update_robot_position(frame):
                        break
                    logger.warning(f"Lost robot tracking, retry {retry_count+1}/{max_retries}")
                    ret, frame = self.cap.read()
                    if not ret:
                        break
                    retry_count += 1
                    time.sleep(0.2)
                
                if retry_count >= max_retries:
                    logger.error("❌ Lost robot tracking - aborting autonomous collection")
                    return False
                
                # Detect balls in real-time
                balls = self.detect_balls(frame)
                
                if not balls:
                    logger.info("✅ No more balls detected - autonomous collection complete!")
                    
                    # Show completion status
                    frame = self.draw_walls(frame)
                    frame = self.draw_robot_markers(frame)
                    frame = self.draw_status(frame)
                    cv2.putText(frame, f"🎉 Collection Complete! Collected {collected_balls} balls", 
                              (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.7, (0, 255, 0), 2)
                    cv2.putText(frame, "No more balls detected", 
                              (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.7, (255, 255, 255), 2)
                    cv2.imshow("Live Autonomous Collection", frame)
                    cv2.waitKey(3000)  # Show completion for 3 seconds
                    return True
                
                # Find closest reachable ball
                closest_ball = None
                min_distance = float('inf')
                
                for ball in balls:
                    ball_pos = (ball[0], ball[1])
                    distance = math.hypot(ball_pos[0] - self.robot_pos[0], ball_pos[1] - self.robot_pos[1])
                    
                    # Check if ball is reachable (not blocked by walls)
                    if not check_wall_collision(self.robot_pos, ball_pos, self.walls, WALL_SAFETY_MARGIN):
                        if distance < min_distance:
                            min_distance = distance
                            closest_ball = ball
                
                if not closest_ball:
                    logger.warning("All remaining balls are blocked by walls - ending collection")
                    return True
                
                ball_pos = (closest_ball[0], closest_ball[1])
                logger.info(f"🎯 Target: {closest_ball[2]} ball at ({ball_pos[0]:.1f}, {ball_pos[1]:.1f}), distance: {min_distance:.1f}cm")
                
                # Calculate approach vector
                approach_pos, target_angle = self.get_approach_vector(ball_pos)
                
                # Check if approach position is reachable
                if check_wall_collision(self.robot_pos, approach_pos, self.walls, WALL_SAFETY_MARGIN):
                    logger.warning("Approach to ball blocked by wall, skipping...")
                    continue
                
                # Create immediate path to this ball
                immediate_path = [
                    {
                        'type': 'approach',
                        'pos': approach_pos,
                        'angle': target_angle
                    },
                    {
                        'type': 'collect',
                        'pos': ball_pos,
                        'ball_type': closest_ball[2]
                    }
                ]
                
                # Add goal delivery path
                goal_path = self.calculate_goal_approach_path(ball_pos, target_angle)
                immediate_path.extend(goal_path)
                
                # Show current plan with live visualization
                frame = self.draw_walls(frame)
                frame = self.draw_robot_markers(frame)
                frame = self.draw_detected_balls(frame, balls)
                frame = self.draw_path(frame, immediate_path)
                frame = self.draw_status(frame)
                
                cv2.putText(frame, f"🔄 LIVE AUTONOMOUS - Ball {collected_balls+1}", 
                          (10, frame.shape[0] - 90), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"Target: {closest_ball[2]} ball at ({ball_pos[0]:.1f}, {ball_pos[1]:.1f})", 
                          (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.7, (0, 255, 0), 2)
                cv2.putText(frame, "Executing live-planned path...", 
                          (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.7, (255, 255, 255), 2)
                
                cv2.imshow("Live Autonomous Collection", frame)
                cv2.waitKey(1000)  # Show plan for 1 second
                
                # Execute each step with live position updates
                for i, target_point in enumerate(immediate_path):
                    logger.info(f"Executing step {i+1}/{len(immediate_path)}: {target_point['type']}")
                    
                    # Update robot position before each movement
                    for retry in range(3):
                        ret, frame = self.cap.read()
                        if ret and self.update_robot_position(frame):
                            break
                        time.sleep(0.1)
                    
                    # Calculate movement needed from current live position
                    target_pos = target_point['pos']
                    current_pos = self.robot_pos
                    
                    dx = target_pos[0] - current_pos[0]
                    dy = target_pos[1] - current_pos[1]
                    distance = math.hypot(dx, dy)
                    
                    # Skip if we're already very close
                    if distance < 2:
                        logger.info(f"Already at target position (distance: {distance:.1f}cm)")
                        continue
                    
                    # Execute movement based on waypoint type
                    if target_point['type'] == 'collect':
                        # For collection, use standard movement (need to face the ball)
                        target_angle = math.degrees(math.atan2(dy, dx))
                        angle_diff = (target_angle - self.robot_heading + 180) % 360 - 180
                        
                        if abs(angle_diff) > 5:
                            logger.info(f"Turning {angle_diff:.1f} degrees to face ball")
                            if not self.turn(angle_diff):
                                logger.error("❌ Turn command failed")
                                return False
                            self.robot_heading = target_angle
                        
                        logger.info(f"🏐 Collecting {target_point['ball_type']} ball")
                        if not self.collect(COLLECTION_DISTANCE_CM):
                            logger.error("❌ Ball collection failed")
                            return False
                        collected_balls += 1
                        logger.info(f"✅ Ball collected! Total: {collected_balls}")
                        
                    elif target_point['type'] == 'goal':
                        # Use smart movement for goal approach
                        if not self.execute_smart_movement(target_pos, "goal"):
                            logger.error("❌ Smart goal movement failed")
                            return False
                        
                        # Deliver balls at goal
                        logger.info("🎁 Delivering balls...")
                        if not self.deliver_balls():
                            logger.error("❌ Ball delivery failed")
                            return False
                        logger.info(f"✅ Delivered {collected_balls} balls to goal!")
                        
                    else:  # 'approach' waypoint
                        # Use smart movement for approach
                        if not self.execute_smart_movement(target_pos, "approach"):
                            logger.error("❌ Smart approach movement failed")
                            return False
                    
                    # Update visualization after each step
                    ret, frame = self.cap.read()
                    if ret:
                        self.update_robot_position(frame)
                        frame = self.draw_status(frame)
                        frame = self.draw_robot_markers(frame)
                        frame = self.draw_walls(frame)
                        
                        cv2.putText(frame, f"Step {i+1}/{len(immediate_path)} completed - Ball {collected_balls}", 
                                  (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.7, (0, 255, 0), 2)
                        cv2.imshow("Live Autonomous Collection", frame)
                        cv2.waitKey(500)  # Brief pause to show progress
                
                # After completing this ball's collection, immediately scan for next ball
                logger.info(f"🔄 Ball {collected_balls} complete! Scanning for next target...")
                time.sleep(0.5)  # Brief pause before next scan
            
            logger.info(f"✅ Autonomous collection session complete! Collected {collected_balls} balls")
            return True
            
        except Exception as e:
            logger.error(f"❌ Live autonomous collection failed: {e}")
            self.stop()  # Emergency stop
            return False

    def _execute_immediate_path(self, path):
        """Execute a specific immediate path (usually for one ball) with live position tracking"""
        if not path:
            logger.warning("No immediate path to execute")
            return False
        
        logger.info(f"🎯 Executing immediate path with {len(path)} steps...")
        
        try:
            # Execute each step with live position updates
            for i, target_point in enumerate(path):
                logger.info(f"Executing step {i+1}/{len(path)}: {target_point['type']}")
                
                # Update robot position before each movement
                for retry in range(3):
                    ret, frame = self.cap.read()
                    if ret and self.update_robot_position(frame):
                        break
                    time.sleep(0.1)
                
                # Calculate movement needed from current live position
                target_pos = target_point['pos']
                current_pos = self.robot_pos
                
                dx = target_pos[0] - current_pos[0]
                dy = target_pos[1] - current_pos[1]
                distance = math.hypot(dx, dy)
                
                # Skip if we're already very close
                if distance < 2:
                    logger.info(f"Already at target position (distance: {distance:.1f}cm)")
                    continue
                
                # Calculate turn needed
                target_angle = math.degrees(math.atan2(dy, dx))
                angle_diff = (target_angle - self.robot_heading + 180) % 360 - 180
                
                # Turn to face target if significant difference
                if abs(angle_diff) > 5:
                    logger.info(f"Turning {angle_diff:.1f} degrees to face target")
                    if not self.turn(angle_diff):
                        logger.error("❌ Turn command failed")
                        return False
                    
                    # Update heading after turn
                    self.robot_heading = target_angle
                
                # Execute movement based on waypoint type
                if target_point['type'] == 'collect':
                    logger.info(f"🏐 Collecting {target_point.get('ball_type', 'ball')}")
                    if not self.collect(COLLECTION_DISTANCE_CM):
                        logger.error("❌ Ball collection failed")
                        return False
                    logger.info("✅ Ball collected!")
                    
                elif target_point['type'] == 'goal':
                    logger.info(f"🎯 Moving {distance:.1f}cm to goal")
                    if not self.move(distance):
                        logger.error("❌ Goal movement failed")
                        return False
                    
                    # Deliver balls at goal
                    logger.info("🎁 Delivering balls...")
                    if not self.deliver_balls():
                        logger.error("❌ Ball delivery failed")
                        return False
                    logger.info("✅ Balls delivered to goal!")
                    
                else:  # 'approach' waypoint
                    logger.info(f"➡️ Moving {distance:.1f}cm to approach position")
                    if not self.move(distance):
                        logger.error("❌ Approach movement failed")
                        return False
                
                # Brief pause between steps
                time.sleep(0.2)
            
            logger.info("✅ Immediate path execution complete!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Immediate path execution failed: {e}")
            self.stop()  # Emergency stop
            return False

    def run_adaptive_autonomous(self):
        """Simple live tracking - just move toward balls and collect them"""
        logger.info("🔄 Starting simple live tracking...")
        
        cv2.namedWindow("Live Tracking")
        
        while True:
            try:
                # Get current frame and robot position
                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                # Update robot position
                if not self.update_robot_position(frame):
                    cv2.putText(frame, "❌ Robot tracking lost!", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow("Live Tracking", frame)
                    if cv2.waitKey(100) & 0xFF == ord('q'):
                        break
                    continue
                
                # Draw everything
                frame = self.draw_walls(frame)
                frame = self.draw_robot_markers(frame)
                frame = self.draw_status(frame)
                
                # Show robot position and heading
                cv2.putText(frame, f"Robot: ({self.robot_pos[0]:.1f}, {self.robot_pos[1]:.1f})", 
                          (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"Heading: {self.robot_heading:.1f}°", 
                          (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Show backward driving status
                backward_status = "ON" if self.backward_driving_enabled else "OFF"
                backward_color = (0, 255, 0) if self.backward_driving_enabled else (0, 0, 255)
                cv2.putText(frame, f"Backward driving: {backward_status}", 
                          (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, backward_color, 2)
                
                # Find balls
                balls = self.detect_balls(frame)
                frame = self.draw_detected_balls(frame, balls)
                
                if balls:
                    # Find closest ball
                    closest_ball = min(balls, key=lambda b: math.hypot(
                        b[0] - self.robot_pos[0], b[1] - self.robot_pos[1]
                    ))
                    
                    ball_x, ball_y = closest_ball[0], closest_ball[1]
                    
                    # Calculate distance and angle to ball
                    dx = ball_x - self.robot_pos[0]
                    dy = ball_y - self.robot_pos[1]
                    distance = math.hypot(dx, dy)
                    angle_to_ball = math.degrees(math.atan2(dy, dx))
                    # Normalize angle to [-180, 180]
                    angle_to_ball = (angle_to_ball + 180) % 360 - 180
                    
                    # Calculate turn needed (shortest angular distance)
                    angle_diff = angle_to_ball - self.robot_heading
                    # Normalize to [-180, 180] for shortest turn
                    while angle_diff > 180:
                        angle_diff -= 360
                    while angle_diff < -180:
                        angle_diff += 360
                    
                    cv2.putText(frame, f"Target: ({ball_x:.1f}, {ball_y:.1f})", 
                              (10, frame.shape[0] - 120), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.6, (0, 255, 0), 2)
                    cv2.putText(frame, f"Distance: {distance:.1f}cm", 
                              (10, frame.shape[0] - 90), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.6, (0, 255, 255), 2)
                    cv2.putText(frame, f"Ball angle: {angle_to_ball:.1f}°", 
                              (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.6, (255, 255, 0), 2)
                    cv2.putText(frame, f"Turn needed: {angle_diff:.1f}°", 
                              (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.6, (255, 0, 255), 2)
                    
                    # Simple logic: if close enough, collect. Otherwise move toward it.
                    if distance < 20:  # Close enough to collect
                        logger.info(f"🏐 Collecting ball at {distance:.1f}cm")
                        if self.collect(15):  # Move forward 15cm while collecting
                            logger.info("✅ Ball collected, moving to goal...")
                            
                            # After collecting, move to goal and deliver
                            goal_candidates = self.goal_ranges.get(self.selected_goal, [])
                            if goal_candidates:
                                goal_x = goal_candidates[0][0]  # X coordinate of goal
                                goal_y = sum(y for (x, y) in goal_candidates) / len(goal_candidates)  # Average Y
                                
                                # Offset goal position to stay away from wall
                                if goal_x == 0:  # Left side goal
                                    goal_x = 15  # Stay 15cm from left wall
                                else:  # Right side goal
                                    goal_x = FIELD_WIDTH_CM - 15  # Stay 15cm from right wall
                                
                                # Calculate path to goal
                                goal_dx = goal_x - self.robot_pos[0]
                                goal_dy = goal_y - self.robot_pos[1]
                                goal_distance = math.hypot(goal_dx, goal_dy)
                                goal_angle = math.degrees(math.atan2(goal_dy, goal_dx))
                                goal_angle = (goal_angle + 180) % 360 - 180  # Normalize
                                
                                # Turn toward goal
                                goal_angle_diff = goal_angle - self.robot_heading
                                while goal_angle_diff > 180:
                                    goal_angle_diff -= 360
                                while goal_angle_diff < -180:
                                    goal_angle_diff += 360
                                
                                if abs(goal_angle_diff) > 5:
                                    logger.info(f"Turning {goal_angle_diff:.1f}° toward goal")
                                    self.turn(goal_angle_diff)
                                    # Update heading after turn
                                    self.robot_heading = goal_angle
                                
                                # Move toward goal (in smaller steps)
                                if goal_distance > 20:
                                    move_distance = min(goal_distance - 15, 40)  # Max 40cm, leave 15cm buffer
                                    if move_distance > 5:
                                        logger.info(f"Moving {move_distance:.1f}cm toward goal")
                                        self.move(move_distance)
                                
                                # Deliver balls
                                logger.info("🎁 Delivering balls...")
                                self.deliver_balls()
                        
                    else:
                        # Use smart movement toward ball
                        ball_approach_pos = (ball_x - 15 * math.cos(math.radians(angle_to_ball)), 
                                           ball_y - 15 * math.sin(math.radians(angle_to_ball)))
                        
                        # Check if we need to move at all
                        approach_distance = math.hypot(ball_approach_pos[0] - self.robot_pos[0], 
                                                     ball_approach_pos[1] - self.robot_pos[1])
                        
                        if approach_distance > 5:  # Only move if we're not close enough
                            logger.info(f"Smart movement toward ball (distance: {approach_distance:.1f}cm)")
                            self.execute_smart_movement(ball_approach_pos, "ball_approach")
                
                else:
                    cv2.putText(frame, "🔍 No balls detected", 
                              (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.6, (255, 255, 255), 2)
                
                cv2.imshow("Live Tracking", frame)
                
                # Small delay between decisions
                if cv2.waitKey(800) & 0xFF == ord('q'):  # Reduced to 0.8 second
                    break
                    
            except Exception as e:
                logger.error(f"Error: {e}")
                time.sleep(1)
        
        cv2.destroyWindow("Live Tracking")
        logger.info("Live tracking ended")

    def send_route_to_robot(self, path):
        """Legacy method - now redirects to autonomous execution"""
        logger.info("🔄 Redirecting to autonomous execution with live tracking...")
        return self.execute_autonomous_collection()

    def calculate_goal_approach_path(self, current_pos, current_heading):
        """Calculate a smooth path to the selected goal using the new goal system"""
        # Get goal candidates for selected goal
        goal_candidates = self.goal_ranges.get(self.selected_goal, [])
        if not goal_candidates:
            logger.warning("No goal candidates for %s", self.selected_goal)
            return []
        
        # Calculate target point with offset from goal edge
        y_vals = [y for (x, y) in goal_candidates]
        goal_y = sum(y_vals) / len(y_vals)  # Middle Y of goal
        
        # Determine goal side and calculate X with offset
        goal_x_edge = goal_candidates[0][0]  # X coordinate of goal (0 or FIELD_WIDTH_CM)
        
        if goal_x_edge == 0:  # Left side goal
            goal_x = GOAL_OFFSET_CM  # Stop before left edge
        else:  # Right side goal
            goal_x = FIELD_WIDTH_CM - GOAL_OFFSET_CM  # Stop before right edge
        
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
        if goal_x_edge == 0:  # Left side goal
            approach_x = goal_x + (GOAL_OFFSET_CM * 1.5)  # Further from left edge
        else:  # Right side goal  
            approach_x = goal_x - (GOAL_OFFSET_CM * 1.5)  # Further from right edge
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
                'angle': angle_to_goal  # Face toward goal
            })
        
        if not check_wall_collision((approach_x, approach_y), (goal_x, goal_y), self.walls, WALL_SAFETY_MARGIN):
            path.append({
                'type': 'goal',
                'pos': (goal_x, goal_y),
                'angle': 0
            })
        
        return path

    def collect_balls(self):
        """Main ball collection loop with preview and manual control"""
        cv2.namedWindow("Path Planning")
        last_plan_time = 0
        last_plan_positions = []
        
        while True:
            try:
                # Capture frame for visualization
                ret, frame = self.cap.read()
                if not ret:
                    continue

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

                # Detect balls
                balls = self.detect_balls(frame)
                current_time = time.time()
                
                # Draw detected balls on frame
                frame = self.draw_detected_balls(frame, balls)
                
                # Only replan if:
                # 1. We have balls AND robot markers are detected AND
                # 2. Either:
                #    a) It's been at least 3 seconds since last plan OR
                #    b) Ball positions have changed significantly OR
                #    c) Robot has moved significantly
                should_replan = False
                if balls and self.robot_pos:
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

                    # Add balls to path in nearest-neighbor order
                    while remaining_balls:
                        closest_ball = min(remaining_balls, 
                                         key=lambda b: math.hypot(
                                             b[0] - current_pos[0],
                                             b[1] - current_pos[1]
                                         ))
                        
                        ball_pos = (closest_ball[0], closest_ball[1])
                        approach_pos, target_angle = self.get_approach_vector(ball_pos)
                        
                        # Check for wall collisions
                        if check_wall_collision(current_pos, approach_pos, self.walls, WALL_SAFETY_MARGIN):
                            logger.warning(f"Path to ball at {ball_pos} blocked by wall, skipping")
                            remaining_balls.remove(closest_ball)
                            continue
                        
                        if check_wall_collision(approach_pos, ball_pos, self.walls, WALL_SAFETY_MARGIN):
                            logger.warning(f"Approach to ball at {ball_pos} blocked by wall, skipping")
                            remaining_balls.remove(closest_ball)
                            continue
                        
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
                        current_heading = target_angle
                        remaining_balls.remove(closest_ball)

                    # Add goal approach path if we have collected any balls
                    if path:
                        goal_path = self.calculate_goal_approach_path(current_pos, current_heading)
                        path.extend(goal_path)

                    # Draw path on frame
                    frame_with_path = self.draw_path(frame, path)
                    
                    # Add key command help
                    help_text = [
                        "Commands:",
                        "SPACE - Show preview",
                        "ENTER - Execute LIVE autonomous",
                        "S - Start LIVE autonomous collection", 
                        "A - Adaptive real-time mode",
                        "D - Debug detection view",
                        "1 - Select Goal A (small)",
                        "2 - Select Goal B (large)",
                        "3 - Toggle goal sides",
                        "B - Toggle backward driving",
                        "T - Test backward movement",
                        "Q - Quit"
                    ]
                    y = 150
                    for text in help_text:
                        cv2.putText(frame_with_path, text,
                                  (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                                  0.5, (255, 255, 255), 1)
                        y += 20
                    
                    # Store path for later execution
                    self.current_path = path
                    cv2.imshow("Path Planning", frame_with_path)
                
                # Show frame even when not replanning
                cv2.imshow("Path Planning", frame)
                
                # Handle key commands
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('1'):
                    # Select Goal A (small goal)
                    self.selected_goal = 'A'
                    logger.info("✅ Selected Goal A (small goal)")
                elif key == ord('2'):
                    # Select Goal B (large goal)
                    self.selected_goal = 'B'
                    logger.info("✅ Selected Goal B (large goal)")
                elif key == ord('3'):
                    # Toggle small goal side between left and right
                    self.small_goal_side = "right" if self.small_goal_side == "left" else "left"
                    self.goal_ranges = self._build_goal_ranges()
                    self.setup_walls()  # Rebuild walls to exclude new goal positions
                    logger.info(f"🔄 Toggled small goal side to {self.small_goal_side}")
                elif key == ord('b') or key == ord('B'):
                    # Toggle backward driving
                    self.backward_driving_enabled = not self.backward_driving_enabled
                    status = "enabled" if self.backward_driving_enabled else "disabled"
                    logger.info(f"🔄 Backward driving {status}")
                elif key == ord('t') or key == ord('T'):
                    # Test backward movement
                    logger.info("🧪 Testing backward movement...")
                    logger.info("Moving backward 10cm...")
                    if self.move(-10):
                        logger.info("✅ Backward movement successful!")
                        time.sleep(1)
                        logger.info("Moving forward 10cm...")
                        if self.move(10):
                            logger.info("✅ Forward movement successful!")
                        else:
                            logger.error("❌ Forward movement failed!")
                    else:
                        logger.error("❌ Backward movement failed!")
                elif key == ord('s') or key == ord('S'):
                    # Start LIVE autonomous collection
                    logger.info("🤖 Starting LIVE autonomous collection...")
                    if self.execute_autonomous_collection():
                        cv2.putText(frame, "LIVE autonomous collection completed!", 
                                  (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, "LIVE autonomous collection failed!", 
                                  (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow("Path Planning", frame)
                    cv2.waitKey(2000)  # Show message for 2 seconds
                elif key == ord('a') or key == ord('A'):
                    # Adaptive real-time autonomous mode
                    logger.info("🔄 Starting adaptive real-time autonomous mode...")
                    self.run_adaptive_autonomous()
                    break  # Exit main loop after adaptive mode
                elif key == ord('d') or key == ord('D'):
                    # Show debug detection view
                    logger.info("🔍 Showing detection debug view...")
                    while True:
                        ret, debug_frame = self.cap.read()
                        if not ret:
                            continue
                        
                        # Update robot position
                        self.update_robot_position(debug_frame)
                        
                        # Show debug visualization
                        debug_view = self.show_detection_debug(debug_frame)
                        
                        # Detect and draw balls
                        balls = self.detect_balls(debug_frame)
                        debug_frame = self.draw_detected_balls(debug_frame, balls)
                        
                        # Show both normal and debug view
                        combined_view = np.hstack([debug_frame, cv2.resize(debug_view, (debug_frame.shape[1], debug_frame.shape[0]))])
                        
                        cv2.putText(combined_view, f"Detected {len(balls)} balls - Press 'Q' to exit debug", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        cv2.imshow("Detection Debug", combined_view)
                        
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            cv2.destroyWindow("Detection Debug")
                            break
                elif key == ord(' ') and hasattr(self, 'current_path'):
                    # Show preview
                    preview_frame = frame.copy()
                    preview_frame = self.draw_path(preview_frame, self.current_path)
                    cv2.putText(preview_frame, "PREVIEW MODE - Press ENTER to execute LIVE autonomous", 
                              (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.imshow("Path Planning", preview_frame)
                    cv2.waitKey(1000)  # Show preview for 1 second
                elif key == 13:  # Enter key
                    # Execute LIVE autonomous collection
                    logger.info("🤖 Starting LIVE autonomous collection (from ENTER key)...")
                    if self.execute_autonomous_collection():
                        cv2.putText(frame, "LIVE autonomous collection completed!", 
                                  (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, "LIVE autonomous collection failed!", 
                                  (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow("Path Planning", frame)
                    cv2.waitKey(2000)  # Show message for 2 seconds
                    
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                cv2.waitKey(1000)

        cv2.destroyWindow("Path Planning")

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