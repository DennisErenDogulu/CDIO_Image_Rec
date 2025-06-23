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
RF_VERSION = 4

# Color detection ranges (HSV)
GREEN_LOWER = np.array([35, 50, 50])
GREEN_UPPER = np.array([85, 255, 255])

# Multiple red ranges to handle different lighting conditions
RED_RANGES = [
    # Primary red range (lower hue values)
    (np.array([0, 50, 50]), np.array([10, 255, 255])),
    # Upper red range (higher hue values - red wraps around in HSV)
    (np.array([170, 50, 50]), np.array([180, 255, 255])),
    # Brighter red (higher value, lower saturation for bright light)
    (np.array([0, 30, 100]), np.array([15, 200, 255])),
    # Darker red (for shadows or dim lighting)
    (np.array([165, 80, 30]), np.array([180, 255, 200]))
]

# Marker dimensions
GREEN_MARKER_WIDTH_CM = 20  # Width of green base sheet
RED_MARKER_WIDTH_CM = 5     # Width of red direction marker

# Wall configuration  
WALL_SAFETY_MARGIN = MIN_CLEARANCE_CM  # cm, minimum distance to keep from walls (robot radius + safety)

# Goal configuration
SMALL_GOAL_SIDE = "right"  # "left" or "right" - where the small goal (A) is placed
GOAL_OFFSET_CM = 6   # cm, distance to stop before goal edge (closer to goal)

# Robot configuration
ROBOT_START_X = 20  # cm from left edge
ROBOT_START_Y = 20  # cm from bottom edge
ROBOT_WIDTH = 26.5   # cm (actual measured width)
ROBOT_LENGTH = 20  # cm
ROBOT_START_HEADING = 0  # degrees (0 = facing east)

# Safety margins based on robot size
ROBOT_RADIUS = ROBOT_WIDTH / 2  # 13.25cm radius
MIN_CLEARANCE_CM = ROBOT_RADIUS + 5  # Robot radius + 5cm safety = ~18cm

# Physical constraints
FIELD_WIDTH_CM = 180
FIELD_HEIGHT_CM = 120
COLLECTION_DISTANCE_CM = 20  # Distance to move forward when collecting
APPROACH_DISTANCE_CM = MIN_CLEARANCE_CM + 15    # Distance to keep from ball for approach (robot clearance + extra)
MAX_BALLS_PER_TRIP = 3

# Ignored area (center obstacle) - will be set during calibration
IGNORED_AREA = {
    "x_min": 0, "x_max": 0,
    "y_min": 0, "y_max": 0
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
    """Check if a path between two points collides with any walls or obstacles"""
    # Check wall collisions
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
    
    # Check obstacle collisions
    if (IGNORED_AREA["x_max"] > IGNORED_AREA["x_min"] and 
        IGNORED_AREA["y_max"] > IGNORED_AREA["y_min"]):
        
        # Expand obstacle bounds by safety margin
        expanded_x_min = IGNORED_AREA["x_min"] - safety_margin
        expanded_x_max = IGNORED_AREA["x_max"] + safety_margin
        expanded_y_min = IGNORED_AREA["y_min"] - safety_margin
        expanded_y_max = IGNORED_AREA["y_max"] + safety_margin
        
        # Check if either endpoint is inside expanded obstacle
        for pos in [start_pos, end_pos]:
            if (expanded_x_min <= pos[0] <= expanded_x_max and 
                expanded_y_min <= pos[1] <= expanded_y_max):
                return True
        
        # Check if path intersects with obstacle edges (treat as 4 wall segments)
        obstacle_walls = [
            (expanded_x_min, expanded_y_min, expanded_x_max, expanded_y_min),  # bottom
            (expanded_x_max, expanded_y_min, expanded_x_max, expanded_y_max),  # right
            (expanded_x_max, expanded_y_max, expanded_x_min, expanded_y_max),  # top
            (expanded_x_min, expanded_y_max, expanded_x_min, expanded_y_min),  # left
        ]
        
        x1, y1 = start_pos
        x2, y2 = end_pos
        
        for wall_start_x, wall_start_y, wall_end_x, wall_end_y in obstacle_walls:
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
        
        # Obstacle calibration points
        self.obstacle_points = []
        
        # Wall configuration
        self.walls = []  # Will be set after calibration
        
        # Goal system
        self.small_goal_side = SMALL_GOAL_SIDE.lower()  # "left" or "right"
        self.selected_goal = 'A'  # 'A' (small goal) or 'B' (large goal)
        self.goal_ranges = self._build_goal_ranges()
        self.delivery_time = 15.0  # Seconds to run collector in reverse
        
        # Frame skipping for lag reduction
        self.frame_skip_counter = 0
        self.frame_skip_interval = 2  # Process every 3rd frame for better performance
        
        # Ball tracking
        self.ignored_balls = []  # Balls ignored because they're in obstacle area

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
        """Detect green base and red direction markers with improved lighting robustness"""
        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Detect green base marker
        green_mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
        
        # Apply morphological operations to clean up green mask
        kernel = np.ones((3,3), np.uint8)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
        
        green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
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
        
        # Find red direction marker within 40 pixels of green marker
        red_center = None
        if green_center:  # Only search for red if green marker is found
            # Create a mask for the search area (40 pixel radius around green marker)
            search_mask = np.zeros_like(hsv[:,:,0])
            cv2.circle(search_mask, green_center, 40, 255, -1)
            
            # Detect red direction marker using multiple color ranges
            red_mask = np.zeros_like(hsv[:,:,0])
            
            # Try each red range and combine the results
            for red_lower, red_upper in RED_RANGES:
                range_mask = cv2.inRange(hsv, red_lower, red_upper)
                red_mask = cv2.bitwise_or(red_mask, range_mask)
            
            # Apply the search area mask to limit detection to within 100 pixels of green
            red_mask = cv2.bitwise_and(red_mask, search_mask)
            
            # Apply morphological operations to clean up red mask
            kernel_small = np.ones((2,2), np.uint8)
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel_small)
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
            
            # Apply Gaussian blur to reduce noise
            red_mask = cv2.GaussianBlur(red_mask, (3, 3), 0)
            
            red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if red_contours:
                # Filter and score red contours
                scored_contours = []
                
                for contour in red_contours:
                    area = cv2.contourArea(contour)
                    if area > 30:  # Lower minimum area for red marker
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
                        red_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        
        return green_center, red_center

    def update_robot_position(self, frame=None):
        """Update robot position and heading based on visual markers"""
        # Capture fresh frame if none provided
        if frame is None:
            ret, frame = self.get_fresh_frame()
            if not ret:
                return False
        
        green_center, red_center = self.detect_markers(frame)
        
        if green_center and red_center and self.homography_matrix is not None:
            # Convert green center (robot base) to cm coordinates
            green_px = np.array([[[float(green_center[0]), float(green_center[1])]]], dtype="float32")
            green_cm = cv2.perspectiveTransform(green_px, np.linalg.inv(self.homography_matrix))[0][0]
            
            # Convert red center (direction marker) to cm coordinates
            red_px = np.array([[[float(red_center[0]), float(red_center[1])]]], dtype="float32")
            red_cm = cv2.perspectiveTransform(red_px, np.linalg.inv(self.homography_matrix))[0][0]
            
            # Update robot position (green marker center)
            self.robot_pos = (green_cm[0], green_cm[1])
            
            # Calculate heading from green to red marker
            dx = red_cm[0] - green_cm[0]
            dy = red_cm[1] - green_cm[1]
            self.robot_heading = math.degrees(math.atan2(dy, dx))
            
            return True
        
        return False

    def draw_robot_markers(self, frame):
        """Draw detected robot markers on frame"""
        green_center, red_center = self.detect_markers(frame)
        
        if green_center:
            # Draw green base marker
            cv2.circle(frame, green_center, 10, (0, 255, 0), -1)
            cv2.circle(frame, green_center, 12, (255, 255, 255), 2)
            # Draw search area for red marker (40 pixel radius)
            cv2.circle(frame, green_center, 40, (0, 255, 255), 1)  # Yellow circle to show search area
        
        if red_center:
            # Draw red direction marker
            cv2.circle(frame, red_center, 8, (0, 0, 255), -1)  # Red color in BGR
            cv2.circle(frame, red_center, 10, (255, 255, 255), 2)
        
        if green_center and red_center:
            # Draw line from base to direction marker
            cv2.line(frame, green_center, red_center, (255, 255, 255), 2)
        
        return frame

    def show_detection_debug(self, frame):
        """Show debug view of marker detection masks"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # First detect green to get search area
        green_center, _ = self.detect_markers(frame)
        
        # Create combined red mask using all ranges
        red_mask_combined = np.zeros_like(hsv[:,:,0])
        individual_masks = []
        
        for i, (red_lower, red_upper) in enumerate(RED_RANGES):
            range_mask = cv2.inRange(hsv, red_lower, red_upper)
            individual_masks.append(range_mask)
            red_mask_combined = cv2.bitwise_or(red_mask_combined, range_mask)
        
        # Apply search area restriction if green marker is found
        if green_center:
            search_mask = np.zeros_like(hsv[:,:,0])
            cv2.circle(search_mask, green_center, 40, 255, -1)
            red_mask_combined = cv2.bitwise_and(red_mask_combined, search_mask)
        
        # Apply the same filtering as in detect_markers
        kernel_small = np.ones((2,2), np.uint8)
        kernel = np.ones((3,3), np.uint8)
        red_mask_filtered = cv2.morphologyEx(red_mask_combined, cv2.MORPH_OPEN, kernel_small)
        red_mask_filtered = cv2.morphologyEx(red_mask_filtered, cv2.MORPH_CLOSE, kernel)
        red_mask_filtered = cv2.GaussianBlur(red_mask_filtered, (3, 3), 0)
        
        # Create a visualization combining original frame and masks
        debug_frame = frame.copy()
        
        # Show red detections as colored overlay
        red_colored = cv2.applyColorMap(red_mask_filtered, cv2.COLORMAP_HOT)
        debug_frame = cv2.addWeighted(debug_frame, 0.7, red_colored, 0.3, 0)
        
        # Draw search area if green marker found
        if green_center:
            cv2.circle(debug_frame, green_center, 40, (0, 255, 255), 2)
            cv2.putText(debug_frame, "Red search area", 
                       (green_center[0] + 10, green_center[1] - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Add text showing how many ranges detected something
        active_ranges = sum(1 for mask in individual_masks if cv2.countNonZero(mask) > 0)
        cv2.putText(debug_frame, f"Active red ranges: {active_ranges}/{len(RED_RANGES)}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show total red pixels detected
        total_red_pixels = cv2.countNonZero(red_mask_filtered)
        cv2.putText(debug_frame, f"Red pixels: {total_red_pixels}", 
                   (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return debug_frame

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
            
            # Draw goal areas
            for goal_label, goal_points in self.goal_ranges.items():
                # Color based on selection
                if goal_label == self.selected_goal:
                    goal_color = (0, 255, 0)  # Green for selected goal
                else:
                    goal_color = (0, 255, 255)  # Yellow for available goal
                
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
            
            # Draw obstacle area if it's defined
            if (IGNORED_AREA["x_max"] > IGNORED_AREA["x_min"] and 
                IGNORED_AREA["y_max"] > IGNORED_AREA["y_min"]):
                
                # Create obstacle rectangle points
                obstacle_corners = [
                    [IGNORED_AREA["x_min"], IGNORED_AREA["y_min"]],
                    [IGNORED_AREA["x_max"], IGNORED_AREA["y_min"]],
                    [IGNORED_AREA["x_max"], IGNORED_AREA["y_max"]],
                    [IGNORED_AREA["x_min"], IGNORED_AREA["y_max"]]
                ]
                
                # Convert obstacle corners to pixels
                obstacle_px_points = []
                for corner in obstacle_corners:
                    corner_cm = np.array([[[corner[0], corner[1]]]], dtype="float32")
                    corner_px = cv2.perspectiveTransform(corner_cm, self.homography_matrix)[0][0].astype(int)
                    obstacle_px_points.append(corner_px)
                
                # Draw obstacle area with orange color and transparency
                obstacle_pts = np.array(obstacle_px_points, np.int32)
                cv2.fillPoly(overlay, [obstacle_pts], (0, 165, 255))  # Orange
                cv2.polylines(overlay, [obstacle_pts], True, (0, 100, 200), 3)  # Darker orange border
                
                # Add obstacle label
                center_x = sum(pt[0] for pt in obstacle_px_points) // 4
                center_y = sum(pt[1] for pt in obstacle_px_points) // 4
                cv2.putText(overlay, "OBSTACLE", (center_x - 40, center_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Draw ignored balls within obstacle area
                if hasattr(self, 'ignored_balls') and self.ignored_balls:
                    for ball_x, ball_y, ball_type in self.ignored_balls:
                        # Convert ball position to pixels
                        ball_cm = np.array([[[ball_x, ball_y]]], dtype="float32")
                        ball_px = cv2.perspectiveTransform(ball_cm, self.homography_matrix)[0][0].astype(int)
                        
                        # Draw ignored ball with gray color and X mark
                        cv2.circle(overlay, tuple(ball_px), 8, (128, 128, 128), -1)  # Gray filled circle
                        cv2.circle(overlay, tuple(ball_px), 10, (64, 64, 64), 2)     # Darker gray border
                        
                        # Draw X mark to indicate ignored
                        cv2.line(overlay, (ball_px[0]-6, ball_px[1]-6), (ball_px[0]+6, ball_px[1]+6), (255, 255, 255), 2)
                        cv2.line(overlay, (ball_px[0]-6, ball_px[1]+6), (ball_px[0]+6, ball_px[1]-6), (255, 255, 255), 2)
                        
                        # Add label
                        cv2.putText(overlay, f"IGNORED", (ball_px[0] + 12, ball_px[1] - 8),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
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
        """Perform camera calibration by clicking 4 corners, then mark obstacle area"""
        calibration_phase = "corners"  # "corners" or "obstacle"
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal calibration_phase
            
            if event == cv2.EVENT_LBUTTONDOWN:
                if calibration_phase == "corners" and len(self.calibration_points) < 4:
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
                        calibration_phase = "obstacle"
                        logger.info("✅ Field corners calibrated. Now click 2 points to mark obstacle area:")
                        logger.info("   1. Click top-left corner of obstacle")
                        logger.info("   2. Click bottom-right corner of obstacle")
                
                elif calibration_phase == "obstacle" and len(self.obstacle_points) < 2:
                    self.obstacle_points.append((x, y))
                    if len(self.obstacle_points) == 1:
                        logger.info(f"Obstacle corner 1 set: ({x}, {y}) - now click bottom-right corner")
                    elif len(self.obstacle_points) == 2:
                        logger.info(f"Obstacle corner 2 set: ({x}, {y})")
                        self.setup_obstacle_area()

        cv2.namedWindow("Calibration")
        cv2.setMouseCallback("Calibration", mouse_callback)
        
        while calibration_phase == "corners" or len(self.obstacle_points) < 2:
            ret, frame = self.get_fresh_frame()
            if not ret:
                continue
            
            # Draw instruction text
            if calibration_phase == "corners":
                instruction = f"Click field corners in order: 1=top-left, 2=top-right, 3=bottom-right, 4=bottom-left ({len(self.calibration_points)}/4)"
                cv2.putText(frame, instruction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Draw existing corner points
                for i, pt in enumerate(self.calibration_points):
                    cv2.circle(frame, pt, 8, (0, 255, 0), -1)
                    cv2.putText(frame, str(i+1), (pt[0]+15, pt[1]+15),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            elif calibration_phase == "obstacle":
                instruction = f"Mark obstacle area: click top-left, then bottom-right corner ({len(self.obstacle_points)}/2)"
                cv2.putText(frame, instruction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Draw field corners in green
                for i, pt in enumerate(self.calibration_points):
                    cv2.circle(frame, pt, 6, (0, 255, 0), -1)
                
                # Draw obstacle points in red
                for i, pt in enumerate(self.obstacle_points):
                    cv2.circle(frame, pt, 8, (0, 0, 255), -1)
                    cv2.putText(frame, f"OBS{i+1}", (pt[0]+15, pt[1]+15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Draw rectangle preview if we have one obstacle point
                if len(self.obstacle_points) == 1:
                    # Show current mouse position as potential second corner
                    mouse_pos = cv2.getWindowProperty("Calibration", cv2.WND_PROP_AUTOSIZE)
                    # This doesn't get mouse position, but we can show the first point at least
                    cv2.rectangle(frame, self.obstacle_points[0], self.obstacle_points[0], (0, 0, 255), 2)
            
            cv2.imshow("Calibration", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and calibration_phase == "obstacle" and len(self.obstacle_points) == 0:
                # Skip obstacle marking by setting a default small area
                logger.info("Skipping obstacle marking - no obstacle will be ignored")
                self.obstacle_points = [(0, 0), (1, 1)]  # Minimal area
                self.setup_obstacle_area()
        
        cv2.destroyWindow("Calibration")
    
    def setup_obstacle_area(self):
        """Convert obstacle pixel coordinates to cm and update IGNORED_AREA"""
        global IGNORED_AREA
        
        if len(self.obstacle_points) == 2 and self.homography_matrix is not None:
            # Convert obstacle corners from pixels to cm
            pt1_px = np.array([[[float(self.obstacle_points[0][0]), float(self.obstacle_points[0][1])]]], dtype="float32")
            pt2_px = np.array([[[float(self.obstacle_points[1][0]), float(self.obstacle_points[1][1])]]], dtype="float32")
            
            pt1_cm = cv2.perspectiveTransform(pt1_px, np.linalg.inv(self.homography_matrix))[0][0]
            pt2_cm = cv2.perspectiveTransform(pt2_px, np.linalg.inv(self.homography_matrix))[0][0]
            
            # Ensure we have min/max coordinates regardless of click order
            x_min = min(pt1_cm[0], pt2_cm[0])
            x_max = max(pt1_cm[0], pt2_cm[0])
            y_min = min(pt1_cm[1], pt2_cm[1])
            y_max = max(pt1_cm[1], pt2_cm[1])
            
            # Update the global IGNORED_AREA
            IGNORED_AREA["x_min"] = x_min
            IGNORED_AREA["x_max"] = x_max
            IGNORED_AREA["y_min"] = y_min
            IGNORED_AREA["y_max"] = y_max
            
            logger.info(f"✅ Obstacle area set: ({x_min:.1f}, {y_min:.1f}) to ({x_max:.1f}, {y_max:.1f}) cm")
            
            # Set up walls after obstacle calibration
            self.setup_walls()
            logger.info("✅ Calibration and wall setup complete")

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
        ignored_balls = []  # Track ignored balls for visualization
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
                
                # Check if ball is in ignored area (obstacle)
                ball_in_obstacle = False
                if (IGNORED_AREA["x_max"] > IGNORED_AREA["x_min"] and 
                    IGNORED_AREA["y_max"] > IGNORED_AREA["y_min"]):
                    # Ball is inside obstacle if within the defined boundaries
                    if (IGNORED_AREA["x_min"] <= x_cm <= IGNORED_AREA["x_max"] and
                        IGNORED_AREA["y_min"] <= y_cm <= IGNORED_AREA["y_max"]):
                        ball_in_obstacle = True
                        ignored_balls.append((x_cm, y_cm, ball_class))
                        logger.debug(f"Ignoring {ball_class} ball at ({x_cm:.1f}, {y_cm:.1f}) - inside obstacle area")
                
                # Only add ball if it's not in the obstacle area
                if not ball_in_obstacle:
                    balls.append((x_cm, y_cm, ball_class))
        
        # Store ignored balls for visualization (as instance variable)
        self.ignored_balls = ignored_balls
        
        return balls

    def calculate_movement_efficiency(self, target_pos: Tuple[float, float]) -> dict:
        """Calculate efficiency scores for different movement approaches - optimized for tank turns"""
        dx = target_pos[0] - self.robot_pos[0]
        dy = target_pos[1] - self.robot_pos[1]
        distance = math.hypot(dx, dy)
        target_angle = math.degrees(math.atan2(dy, dx))
        angle_diff = (target_angle - self.robot_heading + 180) % 360 - 180
        
        # Calculate efficiency for different approaches
        approaches = {}
        
        # 1. Direct tank turn + forward approach (now very efficient with tank turns)
        # Tank turns have minimal cost since we can turn on the spot
        turn_cost = abs(angle_diff) / 360.0  # Reduced penalty for turns (was /180.0)
        approaches['tank_turn_forward'] = {
            'total_cost': turn_cost + distance * 0.01,
            'method': 'tank_turn_forward',
            'angle': target_angle,
            'backing': False
        }
        
        # 2. Backward movement (only if it's significantly better)
        backward_angle = target_angle + 180
        backward_angle_diff = (backward_angle - self.robot_heading + 180) % 360 - 180
        # Only consider backward if it requires much less turning
        if abs(backward_angle_diff) < abs(angle_diff) * 0.5:  # Must be significantly better
            approaches['backward'] = {
                'total_cost': abs(backward_angle_diff) / 360.0 + distance * 0.02,  # Higher distance penalty for backing
                'method': 'backward',
                'angle': backward_angle,
                'backing': True
            }
        
        # Remove complex repositioning approaches since tank turns make them unnecessary
        
        return approaches

    def get_smooth_approach_vector(self, target_pos: Tuple[float, float]) -> Tuple[Tuple[float, float], float, str]:
        """Calculate optimal approach using tank turn capabilities"""
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
        
        # For tank turn forward (the most common case now)
        # Since we can turn on the spot, we can approach directly
        if distance > APPROACH_DISTANCE_CM:
            ratio = (distance - APPROACH_DISTANCE_CM) / distance
            approach_x = self.robot_pos[0] + dx * ratio
            approach_y = self.robot_pos[1] + dy * ratio
        else:
            approach_x, approach_y = self.robot_pos
        
        return (approach_x, approach_y), best_approach['angle'], 'tank_turn_forward'

    def draw_path(self, frame, path):
        """Draw the planned path on the frame"""
        if not self.homography_matrix is None:
            overlay = frame.copy()
            
            # Colors for different point types
            colors = {
                'approach': (0, 255, 255),         # Yellow (legacy)
                'tank_approach': (0, 255, 255),    # Yellow (tank turn approach)
                'collect': (0, 255, 0),            # Green
                'goal': (0, 0, 255),               # Red
                'backward_approach': (255, 128, 0), # Orange
                'backward_collect': (255, 165, 0),  # Orange red
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
                if point['type'] in ['approach', 'tank_approach', 'goal', 'backward_approach']:
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
            logger.info(f"Sending COLLECT_REVERSE command for {self.delivery_time} seconds")
            result = self.send_command("COLLECT_REVERSE", duration=self.delivery_time)
            if result:
                logger.info("COLLECT_REVERSE command executed successfully")
            else:
                logger.warning("COLLECT_REVERSE command returned failure")
            return result
        except Exception as e:
            logger.error("Exception during ball delivery: {}".format(e))
            return False

    def check_clearance_and_escape(self, required_clearance_cm: float = None) -> bool:
        """Check if robot is too close to walls/obstacles and move to safety"""
        if not self.robot_pos or not self.walls:
            return True
        
        if required_clearance_cm is None:
            required_clearance_cm = MIN_CLEARANCE_CM
        
        current_x, current_y = self.robot_pos
        escape_vectors = []  # List of (direction_angle, distance_needed, reason)
        
        # Check distance to walls
        for wall in self.walls:
            wall_start = (wall[0], wall[1])
            wall_end = (wall[2], wall[3])
            distance = point_to_line_distance(self.robot_pos, wall_start, wall_end)
            
            if distance < required_clearance_cm:
                # Calculate direction away from wall
                # Find closest point on wall to robot
                wall_vec = (wall_end[0] - wall_start[0], wall_end[1] - wall_start[1])
                robot_vec = (current_x - wall_start[0], current_y - wall_start[1])
                wall_length_sq = wall_vec[0]**2 + wall_vec[1]**2
                
                if wall_length_sq > 0:
                    t = max(0, min(1, (robot_vec[0] * wall_vec[0] + robot_vec[1] * wall_vec[1]) / wall_length_sq))
                    closest_x = wall_start[0] + t * wall_vec[0]
                    closest_y = wall_start[1] + t * wall_vec[1]
                    
                    # Direction away from wall
                    away_x = current_x - closest_x
                    away_y = current_y - closest_y
                    away_angle = math.degrees(math.atan2(away_y, away_x))
                    
                    distance_needed = required_clearance_cm - distance + 5  # Extra 5cm safety
                    escape_vectors.append((away_angle, distance_needed, f"wall (distance: {distance:.1f}cm)"))
        
        # Check distance to obstacle
        if (IGNORED_AREA["x_max"] > IGNORED_AREA["x_min"] and 
            IGNORED_AREA["y_max"] > IGNORED_AREA["y_min"]):
            
            # Check if robot is inside or too close to obstacle
            obstacle_margin = required_clearance_cm
            
            # Expand obstacle bounds by the required clearance
            expanded_x_min = IGNORED_AREA["x_min"] - obstacle_margin
            expanded_x_max = IGNORED_AREA["x_max"] + obstacle_margin
            expanded_y_min = IGNORED_AREA["y_min"] - obstacle_margin
            expanded_y_max = IGNORED_AREA["y_max"] + obstacle_margin
            
            if (expanded_x_min <= current_x <= expanded_x_max and 
                expanded_y_min <= current_y <= expanded_y_max):
                
                # Calculate direction away from obstacle center
                obstacle_center_x = (IGNORED_AREA["x_min"] + IGNORED_AREA["x_max"]) / 2
                obstacle_center_y = (IGNORED_AREA["y_min"] + IGNORED_AREA["y_max"]) / 2
                
                away_x = current_x - obstacle_center_x
                away_y = current_y - obstacle_center_y
                away_angle = math.degrees(math.atan2(away_y, away_x))
                
                # Calculate minimum distance to get clear
                dist_to_left = abs(current_x - IGNORED_AREA["x_min"])
                dist_to_right = abs(current_x - IGNORED_AREA["x_max"])
                dist_to_bottom = abs(current_y - IGNORED_AREA["y_min"])
                dist_to_top = abs(current_y - IGNORED_AREA["y_max"])
                
                min_dist = min(dist_to_left, dist_to_right, dist_to_bottom, dist_to_top)
                distance_needed = required_clearance_cm - min_dist + 8  # Extra 8cm safety for obstacles
                
                escape_vectors.append((away_angle, distance_needed, f"obstacle (distance: {min_dist:.1f}cm)"))
        
        if escape_vectors:
            # Choose the escape vector that requires the least movement
            best_escape = min(escape_vectors, key=lambda x: x[1])
            escape_angle, escape_distance, reason = best_escape
            
            logger.info(f"⚠️ Robot too close to {reason} - moving {escape_distance:.1f}cm at {escape_angle:.1f}° for safety")
            
            # Turn to escape direction and move
            angle_diff = (escape_angle - self.robot_heading + 180) % 360 - 180
            
            # Turn towards escape direction
            if abs(angle_diff) > 5:  # Only turn if significantly off
                if not self.turn(angle_diff):
                    logger.error("Failed to turn for escape maneuver")
                    return False
                self.update_robot_position()  # Update after turn
            
            # Move in escape direction
            if not self.move(escape_distance):
                logger.error("Failed to move for escape maneuver")
                return False
            
            self.update_robot_position()  # Update after move
            
            # Verify we achieved clearance
            return self.check_clearance_and_escape(required_clearance_cm - 3)  # Slightly less strict on retry
        
        return True

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
                #    a) It's been at least X seconds since last plan OR
                #    b) Ball positions have changed significantly OR
                #    c) Robot has moved significantly OR
                #    d) No current path exists
                should_replan = False
                if balls and self.robot_pos and not skip_heavy_processing:
                    # Reduce wait time if no path is currently planned (for continuous operation)
                    min_plan_interval = 1 if not path else 3  # 1 second if no path, 3 seconds if path exists
                    
                    if current_time - last_plan_time >= min_plan_interval:
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
                        
                        # Add waypoints based on movement type (simplified for tank turns)
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
                        
                        else:  # tank_turn_forward (most common case)
                            # With tank turns, we can do direct approaches
                            path.append({
                                'type': 'tank_approach',
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
                    "1 - Select Goal A (small)",
                    "2 - Select Goal B (large)",
                    "3 - Toggle goal sides",
                    "R - Recalibrate (field + obstacle)",
                    "Q - Quit"
                ]
                y = 150
                for text in help_text:
                    cv2.putText(display_frame, text,
                              (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                              0.5, (255, 255, 255), 1)
                    y += 20
                
                # Show planned path count and continuous operation status
                if path:
                    ball_count = len([p for p in path if p['type'] == 'collect'])
                    cv2.putText(display_frame, f"Path planned: {ball_count} balls",
                              (10, y + 10), cv2.FONT_HERSHEY_SIMPLEX,
                              0.5, (0, 255, 0), 1)
                    cv2.putText(display_frame, "CONTINUOUS MODE: Auto-continues after delivery",
                              (10, y + 30), cv2.FONT_HERSHEY_SIMPLEX,
                              0.4, (0, 255, 255), 1)
                else:
                    cv2.putText(display_frame, "Scanning for balls...",
                              (10, y + 10), cv2.FONT_HERSHEY_SIMPLEX,
                              0.5, (255, 255, 0), 1)
                
                # Show ignored balls count if any
                if hasattr(self, 'ignored_balls') and self.ignored_balls:
                    ignored_count = len(self.ignored_balls)
                    cv2.putText(display_frame, f"Ignored balls in obstacle: {ignored_count}",
                              (10, y + 50), cv2.FONT_HERSHEY_SIMPLEX,
                              0.5, (128, 128, 128), 1)
                
                cv2.imshow("Path Planning", display_frame)
                
                # Handle key commands
                if key == ord('q'):
                    break
                elif key == ord('d'):
                    show_debug = not show_debug
                    logger.info(f"Debug view {'enabled' if show_debug else 'disabled'}")
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
                elif key == ord('r'):
                    # Recalibrate field and obstacle
                    logger.info("🔄 Starting recalibration...")
                    self.calibration_points = []
                    self.obstacle_points = []
                    self.homography_matrix = None
                    cv2.destroyWindow("Path Planning")
                    self.calibrate()
                    cv2.namedWindow("Path Planning")
                    logger.info("✅ Recalibration complete")
                elif key == 13:  # Enter key
                    if path:
                        # Execute the planned path
                        logger.info("Executing path with {} points".format(len(path)))
                        
                        try:
                            # First, ensure robot has safe clearance before starting
                            logger.info("🔍 Checking robot clearance before path execution...")
                            self.update_robot_position()
                            clearance_check = self.check_clearance_and_escape()
                            if not clearance_check:
                                logger.warning("⚠️ Could not achieve safe clearance - proceeding anyway")
                            
                            time.sleep(0.5)  # Brief pause after any clearance movement
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
                                    
                                    # Move closer to goal for delivery (additional 3cm forward)
                                    logger.info("Moving closer to goal for delivery")
                                    if not self.move(3):
                                        raise Exception("Failed to move closer to goal")
                                    
                                    # After reaching goal position, deliver balls
                                    logger.info("Starting ball delivery sequence")
                                    delivery_success = False
                                    try:
                                        delivery_success = self.deliver_balls()
                                        if delivery_success:
                                            logger.info("✅ Ball delivery completed successfully")
                                        else:
                                            logger.warning("⚠️ Ball delivery failed, but continuing with backup")
                                    except Exception as delivery_error:
                                        logger.error(f"⚠️ Ball delivery error: {delivery_error}, but continuing with backup")
                                    
                                    # Always back up 20cm after delivery attempt to clear the goal area
                                    logger.info("Backing up 20cm after delivery")
                                    backup_success = self.move(-20)
                                    if not backup_success:
                                        logger.error("Failed to back up after delivery - this could cause issues")
                                        # Don't raise exception here to avoid stopping the whole sequence
                                    
                                    # Update position after backup and check clearance
                                    if backup_success:
                                        self.update_robot_position()
                                        # Check if robot needs additional clearance from walls/obstacles
                                        clearance_success = self.check_clearance_and_escape()
                                        if not clearance_success:
                                            logger.warning("Failed to achieve required clearance from walls/obstacles")
                                    
                                    if delivery_success:
                                        logger.info("✅ Delivery sequence complete, ready for next collection")
                                    else:
                                        logger.info("⚠️ Delivery had issues but backup completed, ready for next collection")
                                elif point['type'] == 'tank_approach':
                                    logger.info(f"Tank turn approach to target")
                                    if not self.move_to_target_live(target_pos, "approach"):
                                        raise Exception("Tank approach movement failed")
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
                                else:  # any other approach point (legacy support)
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
                            
                            # Clear current path after successful execution
                            path = []
                            logger.info("✅ Path execution completed successfully")
                            
                            # Immediately look for remaining balls for continuous operation
                            logger.info("🔍 Scanning for remaining balls...")
                            time.sleep(1)  # Brief pause to let robot settle
                            
                            # Update robot position and look for more balls
                            ret, scan_frame = self.get_fresh_frame()
                            if ret and self.update_robot_position(scan_frame):
                                remaining_balls = self.detect_balls()
                                
                                if remaining_balls:
                                    logger.info(f"🎯 Found {len(remaining_balls)} remaining balls - planning new collection path")
                                    # Force immediate replanning by resetting timing
                                    last_plan_time = 0
                                    last_plan_positions = []
                                    # Continue main loop to plan and execute new path
                                else:
                                    logger.info("🏁 No more balls detected - collection complete!")
                                    cv2.waitKey(2000)  # Pause to show completion message
                            else:
                                logger.warning("⚠️ Could not scan for remaining balls - check robot markers")
                                cv2.waitKey(2000)
                            
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