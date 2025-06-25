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

# Purple ranges for the direction marker (much more selective than red!)
PURPLE_RANGES = [
    # Main purple range (violet/purple hues)
    (np.array([120, 60, 60]), np.array([150, 255, 255])),
    # Magenta-purple range (for different lighting)
    (np.array([145, 50, 50]), np.array([170, 255, 255])),
]

# Marker dimensions
GREEN_MARKER_WIDTH_CM = 20  # Width of green base sheet
PURPLE_MARKER_WIDTH_CM = 5  # Width of purple direction marker

# Wall configuration
WALL_SAFETY_MARGIN = 1  # cm, minimum distance to keep from walls

# Goal configuration
SMALL_GOAL_SIDE = "left"  # "left" or "right" - where the small goal (A) is placed
GOAL_OFFSET_CM = 6   # cm, distance to stop before goal edge (closer to goal)

# Robot configuration
ROBOT_START_X = 20  # cm from left edge
ROBOT_START_Y = 20  # cm from bottom edge
ROBOT_WIDTH = 26.5   # cm
ROBOT_LENGTH = 20  # cm
ROBOT_START_HEADING = 0  # degrees (0 = facing east)

# Physical constraints
FIELD_WIDTH_CM = 180
FIELD_HEIGHT_CM = 120
COLLECTION_DISTANCE_CM = 20  # Distance to move forward when collecting
APPROACH_DISTANCE_CM = 30    # Distance to keep from ball for approach
MAX_BALLS_PER_TRIP = 5

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
        self.robot_pos = (ROBOT_START_X, ROBOT_START_Y)  # Starting position (green base marker)
        self.robot_heading = ROBOT_START_HEADING  # Starting heading
        self.robot_front_pos = (ROBOT_START_X, ROBOT_START_Y)  # Front position (purple heading marker)
        
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
        
        # Automatic operation control
        self.automatic_mode = False  # Becomes True after first delivery

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
        
        # Find purple direction marker within 40 pixels of green marker
        purple_center = None
        if green_center:  # Only search for purple if green marker is found
            # Create a mask for the search area (40 pixel radius around green marker)
            search_mask = np.zeros_like(hsv[:,:,0])
            cv2.circle(search_mask, green_center, 40, 255, -1)
            
            # Detect purple direction marker using multiple color ranges
            purple_mask = np.zeros_like(hsv[:,:,0])
            
            # Try each purple range and combine the results
            for purple_lower, purple_upper in PURPLE_RANGES:
                range_mask = cv2.inRange(hsv, purple_lower, purple_upper)
                purple_mask = cv2.bitwise_or(purple_mask, range_mask)
            
            # Apply the search area mask to limit detection to within 40 pixels of green
            purple_mask = cv2.bitwise_and(purple_mask, search_mask)
            
            # Apply morphological operations to clean up purple mask
            kernel_small = np.ones((2,2), np.uint8)
            purple_mask = cv2.morphologyEx(purple_mask, cv2.MORPH_OPEN, kernel_small)
            purple_mask = cv2.morphologyEx(purple_mask, cv2.MORPH_CLOSE, kernel)
            
            # Apply Gaussian blur to reduce noise
            purple_mask = cv2.GaussianBlur(purple_mask, (3, 3), 0)
            
            purple_contours, _ = cv2.findContours(purple_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if purple_contours:
                # Filter and score purple contours
                scored_contours = []
                
                for contour in purple_contours:
                    area = cv2.contourArea(contour)
                    if area > 30:  # Lower minimum area for purple marker
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
                    
                    # Instead of using centroid, find the tip point (furthest point from green marker)
                    max_distance = 0
                    tip_point = None
                    
                    # Check each point in the contour to find the one furthest from green marker
                    for point in best_contour:
                        px, py = point[0]
                        distance = math.hypot(px - green_center[0], py - green_center[1])
                        if distance > max_distance:
                            max_distance = distance
                            tip_point = (px, py)
                    
                    if tip_point:
                        purple_center = tip_point
        
        return green_center, purple_center

    def update_robot_position(self, frame=None):
        """Update robot position and heading based on visual markers"""
        # Capture fresh frame if none provided
        if frame is None:
            ret, frame = self.get_fresh_frame()
            if not ret:
                return False
        
        green_center, purple_center = self.detect_markers(frame)
        
        if green_center and purple_center and self.homography_matrix is not None:
            # Convert green center (robot base) to cm coordinates
            green_px = np.array([[[float(green_center[0]), float(green_center[1])]]], dtype="float32")
            green_cm = cv2.perspectiveTransform(green_px, np.linalg.inv(self.homography_matrix))[0][0]
            
            # Convert purple center (direction marker) to cm coordinates
            purple_px = np.array([[[float(purple_center[0]), float(purple_center[1])]]], dtype="float32")
            purple_cm = cv2.perspectiveTransform(purple_px, np.linalg.inv(self.homography_matrix))[0][0]
            
            # Update robot position (green marker center)
            self.robot_pos = (green_cm[0], green_cm[1])
            
            # Update robot front position (purple marker center - where collector is)
            self.robot_front_pos = (purple_cm[0], purple_cm[1])
            
            # Calculate heading from green to purple marker
            dx = purple_cm[0] - green_cm[0]
            dy = purple_cm[1] - green_cm[1]
            self.robot_heading = math.degrees(math.atan2(dy, dx))
            
            return True
        
        return False

    def draw_robot_markers(self, frame):
        """Draw detected robot markers on frame"""
        green_center, purple_center = self.detect_markers(frame)
        
        if green_center:
            # Draw green base marker
            cv2.circle(frame, green_center, 10, (0, 255, 0), -1)
            cv2.circle(frame, green_center, 12, (255, 255, 255), 2)
            # Draw search area for purple marker (40 pixel radius)
            cv2.circle(frame, green_center, 40, (0, 255, 255), 1)  # Yellow circle to show search area
        
        if purple_center:
            # Draw purple direction marker
            cv2.circle(frame, purple_center, 8, (128, 0, 128), -1)  # Purple color in BGR
            cv2.circle(frame, purple_center, 10, (255, 255, 255), 2)
        
        if green_center and purple_center:
            # Draw line from base to direction marker
            cv2.line(frame, green_center, purple_center, (255, 255, 255), 2)
        
        return frame

    def show_detection_debug(self, frame):
        """Show debug view of marker detection masks"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # First detect green to get search area
        green_center, _ = self.detect_markers(frame)
        
        # Create combined purple mask using all ranges
        purple_mask_combined = np.zeros_like(hsv[:,:,0])
        individual_masks = []
        
        for i, (purple_lower, purple_upper) in enumerate(PURPLE_RANGES):
            range_mask = cv2.inRange(hsv, purple_lower, purple_upper)
            individual_masks.append(range_mask)
            purple_mask_combined = cv2.bitwise_or(purple_mask_combined, range_mask)
        
        # Apply search area restriction if green marker is found
        if green_center:
            search_mask = np.zeros_like(hsv[:,:,0])
            cv2.circle(search_mask, green_center, 40, 255, -1)
            purple_mask_combined = cv2.bitwise_and(purple_mask_combined, search_mask)
        
        # Apply the same filtering as in detect_markers
        kernel_small = np.ones((2,2), np.uint8)
        kernel = np.ones((3,3), np.uint8)
        purple_mask_filtered = cv2.morphologyEx(purple_mask_combined, cv2.MORPH_OPEN, kernel_small)
        purple_mask_filtered = cv2.morphologyEx(purple_mask_filtered, cv2.MORPH_CLOSE, kernel)
        purple_mask_filtered = cv2.GaussianBlur(purple_mask_filtered, (3, 3), 0)
        
        # Create a visualization combining original frame and masks
        debug_frame = frame.copy()
        
        # Show purple detections as colored overlay
        purple_colored = cv2.applyColorMap(purple_mask_filtered, cv2.COLORMAP_PLASMA)
        debug_frame = cv2.addWeighted(debug_frame, 0.7, purple_colored, 0.3, 0)
        
        # Draw search area if green marker found
        if green_center:
            cv2.circle(debug_frame, green_center, 40, (0, 255, 255), 2)
            cv2.putText(debug_frame, "Purple search area", 
                       (green_center[0] + 10, green_center[1] - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Add text showing how many ranges detected something
        active_ranges = sum(1 for mask in individual_masks if cv2.countNonZero(mask) > 0)
        cv2.putText(debug_frame, f"Active purple ranges: {active_ranges}/{len(PURPLE_RANGES)}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show total purple pixels detected
        total_purple_pixels = cv2.countNonZero(purple_mask_filtered)
        cv2.putText(debug_frame, f"Purple pixels: {total_purple_pixels}", 
                   (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return debug_frame

    def setup_walls(self):
        """Set up wall segments based on calibration points, excluding goal areas"""
        if len(self.calibration_points) != 4:
            return

        # Create wall segments with small safety margin
        margin = WALL_SAFETY_MARGIN
        
        # Get goal Y ranges for both goals with extra clearance
        goal_a_y_vals = [y for (x, y) in self.goal_ranges['A']]
        goal_b_y_vals = [y for (x, y) in self.goal_ranges['B']]
        
        # Add extra clearance around goals to prevent wall detection issues
        goal_clearance = 8  # cm extra clearance around goal openings
        
        goal_a_y_min = min(goal_a_y_vals) - goal_clearance
        goal_a_y_max = max(goal_a_y_vals) + goal_clearance
        goal_b_y_min = min(goal_b_y_vals) - goal_clearance
        goal_b_y_max = max(goal_b_y_vals) + goal_clearance

        self.walls = [
            # Bottom wall (full width)
            (margin, margin, FIELD_WIDTH_CM - margin, margin),
            
            # Top wall (full width)
            (margin, FIELD_HEIGHT_CM - margin, FIELD_WIDTH_CM - margin, FIELD_HEIGHT_CM - margin),
        ]
        
        # Left wall segments (excluding goals with extra clearance)
        if self.small_goal_side == "left":
            # Goal A on left, so exclude its Y range with extra clearance
            if goal_a_y_min > margin:
                self.walls.append((margin, margin, margin, max(margin, goal_a_y_min)))
            if goal_a_y_max < FIELD_HEIGHT_CM - margin:
                self.walls.append((margin, min(FIELD_HEIGHT_CM - margin, goal_a_y_max), margin, FIELD_HEIGHT_CM - margin))
        else:
            # Goal B on left, so exclude its Y range with extra clearance
            if goal_b_y_min > margin:
                self.walls.append((margin, margin, margin, max(margin, goal_b_y_min)))
            if goal_b_y_max < FIELD_HEIGHT_CM - margin:
                self.walls.append((margin, min(FIELD_HEIGHT_CM - margin, goal_b_y_max), margin, FIELD_HEIGHT_CM - margin))
        
        # Right wall segments (excluding goals with extra clearance)
        if self.small_goal_side == "right":
            # Goal A on right, so exclude its Y range with extra clearance
            if goal_a_y_min > margin:
                self.walls.append((FIELD_WIDTH_CM - margin, margin, FIELD_WIDTH_CM - margin, max(margin, goal_a_y_min)))
            if goal_a_y_max < FIELD_HEIGHT_CM - margin:
                self.walls.append((FIELD_WIDTH_CM - margin, min(FIELD_HEIGHT_CM - margin, goal_a_y_max), FIELD_WIDTH_CM - margin, FIELD_HEIGHT_CM - margin))
        else:
            # Goal B on right, so exclude its Y range with extra clearance
            if goal_b_y_min > margin:
                self.walls.append((FIELD_WIDTH_CM - margin, margin, FIELD_WIDTH_CM - margin, max(margin, goal_b_y_min)))
            if goal_b_y_max < FIELD_HEIGHT_CM - margin:
                self.walls.append((FIELD_WIDTH_CM - margin, min(FIELD_HEIGHT_CM - margin, goal_b_y_max), FIELD_WIDTH_CM - margin, FIELD_HEIGHT_CM - margin))

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
            
            # Blend the overlay with the original frame
            alpha = 0.3
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        return frame

    def check_wall_collision(self, start_pos, end_pos, walls, safety_margin):
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

    def check_obstacle_collision(self, start_pos, end_pos):
        """Check if a direct path would intersect with the obstacle"""
        if (IGNORED_AREA["x_max"] <= IGNORED_AREA["x_min"] or 
            IGNORED_AREA["y_max"] <= IGNORED_AREA["y_min"]):
            return False  # No obstacle defined
        
        # Add bigger safety margin around obstacle for collision detection
        margin = 15  # Much larger margin for obstacle avoidance (was WALL_SAFETY_MARGIN = 1cm)
        x_min = IGNORED_AREA["x_min"] - margin
        x_max = IGNORED_AREA["x_max"] + margin
        y_min = IGNORED_AREA["y_min"] - margin
        y_max = IGNORED_AREA["y_max"] + margin
        
        x1, y1 = start_pos
        x2, y2 = end_pos
        
        # Method 1: Check if either endpoint is inside the expanded obstacle
        if (x_min <= x1 <= x_max and y_min <= y1 <= y_max) or \
           (x_min <= x2 <= x_max and y_min <= y2 <= y_max):
            return True
        
        # Method 2: Check if the line crosses the obstacle using parametric line equation
        # Sample points along the line and check if any fall within obstacle
        num_samples = 20  # Check 20 points along the line
        for i in range(num_samples + 1):
            t = i / num_samples
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            
            if x_min <= x <= x_max and y_min <= y <= y_max:
                return True
        
        # Method 3: Additional check using line-rectangle intersection for robustness
        # Check intersection with each edge of the obstacle rectangle
        obstacle_edges = [
            ((x_min, y_min), (x_max, y_min)),  # top
            ((x_max, y_min), (x_max, y_max)),  # right
            ((x_max, y_max), (x_min, y_max)),  # bottom
            ((x_min, y_max), (x_min, y_min))   # left
        ]
        
        for edge_start, edge_end in obstacle_edges:
            x3, y3 = edge_start
            x4, y4 = edge_end
            
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(denom) < 1e-10:  # Lines are parallel (using small epsilon for floating point)
                continue
                
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
            u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
            
            if 0 <= t <= 1 and 0 <= u <= 1:
                return True
        
        return False

    def plan_path_around_obstacle(self, start_pos, target_pos):
        """Plan a path around the obstacle if direct path is blocked"""
        # First check if direct path is clear
        if not self.check_obstacle_collision(start_pos, target_pos):
            return [target_pos]  # Direct path is fine
        
        logger.info(f"Direct path blocked by obstacle, planning adaptive detour...")
        
        # Get obstacle center and bounds
        obs_center_x = (IGNORED_AREA["x_min"] + IGNORED_AREA["x_max"]) / 2
        obs_center_y = (IGNORED_AREA["y_min"] + IGNORED_AREA["y_max"]) / 2
        
        # Determine which side of obstacle to go around based on start and target positions
        start_to_obs_x = obs_center_x - start_pos[0]
        start_to_obs_y = obs_center_y - start_pos[1]
        target_to_obs_x = obs_center_x - target_pos[0]
        target_to_obs_y = obs_center_y - target_pos[1]
        
        # Choose the side that makes most sense geometrically
        # If start and target are on same side of obstacle, go around the closer edge
        safety_margin = 25  # Large safety margin
        
        # Calculate expanded obstacle bounds
        x_min = IGNORED_AREA["x_min"] - safety_margin
        x_max = IGNORED_AREA["x_max"] + safety_margin
        y_min = IGNORED_AREA["y_min"] - safety_margin
        y_max = IGNORED_AREA["y_max"] + safety_margin
        
        # Ensure waypoints are within field bounds with margin
        field_margin = 10
        x_min = max(x_min, field_margin)
        x_max = min(x_max, FIELD_WIDTH_CM - field_margin)
        y_min = max(y_min, field_margin)
        y_max = min(y_max, FIELD_HEIGHT_CM - field_margin)
        
        # Determine best route: left, right, top, or bottom around obstacle
        routes = []
        
        # Route 1: Go around left side (via bottom-left, top-left)
        if x_min > field_margin:
            left_bottom = (x_min, max(y_min, start_pos[1] - 10))
            left_top = (x_min, min(y_max, target_pos[1] + 10))
            routes.append(("left", [left_bottom, left_top, target_pos]))
        
        # Route 2: Go around right side (via bottom-right, top-right)  
        if x_max < FIELD_WIDTH_CM - field_margin:
            right_bottom = (x_max, max(y_min, start_pos[1] - 10))
            right_top = (x_max, min(y_max, target_pos[1] + 10))
            routes.append(("right", [right_bottom, right_top, target_pos]))
        
        # Route 3: Go around bottom (via bottom-left, bottom-right)
        if y_min > field_margin:
            bottom_left = (max(x_min, start_pos[0] - 10), y_min)
            bottom_right = (min(x_max, target_pos[0] + 10), y_min)
            routes.append(("bottom", [bottom_left, bottom_right, target_pos]))
        
        # Route 4: Go around top (via top-left, top-right)
        if y_max < FIELD_HEIGHT_CM - field_margin:
            top_left = (max(x_min, start_pos[0] - 10), y_max)
            top_right = (min(x_max, target_pos[0] + 10), y_max)
            routes.append(("top", [top_left, top_right, target_pos]))
        
        # Find the shortest valid route
        best_route = None
        best_distance = float('inf')
        
        for route_name, waypoints in routes:
            total_distance = 0
            valid_route = True
            
            # Check if all segments are clear
            current_pos = start_pos
            for waypoint in waypoints:
                # Check collision for this segment
                if (self.check_obstacle_collision(current_pos, waypoint) or
                    self.check_wall_collision(current_pos, waypoint, self.walls, WALL_SAFETY_MARGIN)):
                    valid_route = False
                    break
                
                # Add distance
                total_distance += math.hypot(waypoint[0] - current_pos[0], waypoint[1] - current_pos[1])
                current_pos = waypoint
            
            if valid_route and total_distance < best_distance:
                best_distance = total_distance
                best_route = (route_name, waypoints)
        
        if best_route:
            route_name, waypoints = best_route
            logger.info(f"Using {route_name} route around obstacle with {len(waypoints)} waypoints")
            return waypoints
        else:
            logger.warning("Could not find safe route around obstacle, trying direct path")
            return [target_pos]

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
            
            # Set up field walls first
            self.setup_walls()
            
            # ↓↓↓  NEW CODE  ↓↓↓ -----------------------------------------------
            # Add obstacle walls AFTER field walls are set up
            # Inflate by the same safety margin you use for the field walls
            m = WALL_SAFETY_MARGIN
            self.walls.extend([
                (x_min - m, y_min - m, x_max + m, y_min - m),  # top
                (x_max + m, y_min - m, x_max + m, y_max + m),  # right
                (x_max + m, y_max + m, x_min - m, y_max + m),  # bottom
                (x_min - m, y_max + m, x_min - m, y_min - m)   # left
            ])
            
            logger.info(f"✅ Added 4 obstacle walls with {m}cm safety margin")
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
                
                # Skip drawing direction arrows since we simplified to direct movement
                
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

    def check_wall_proximity(self) -> bool:
        """Check if robot front (collector) is too close to any wall and back up if needed"""
        if not self.walls or not self.robot_front_pos:
            return True
        
        wall_danger_distance = 5  # cm - reduced from 8cm to be less aggressive
        
        for wall in self.walls:
            wall_start = (wall[0], wall[1])
            wall_end = (wall[2], wall[3])
            distance = point_to_line_distance(self.robot_front_pos, wall_start, wall_end)
            
            if distance < wall_danger_distance:
                logger.warning(f"Robot front (collector) very close to wall ({distance:.1f}cm)! Small backup for safety")
                
                # Back up 20cm away from the wall
                if not self.move(-20):
                    logger.warning("Failed to back up from wall - continuing anyway")
                    return True  # Don't fail the whole movement, just continue
                
                # Update position after backing up
                if not self.update_robot_position():
                    logger.warning("Lost robot tracking after wall backup")
                    return True  # Don't fail, just continue
                
                logger.info(f"Backed away from wall, new front position: ({self.robot_front_pos[0]:.1f}, {self.robot_front_pos[1]:.1f})")
                return True
        
        return True

    def move_to_target_simple(self, target_pos: Tuple[float, float], target_type: str = "approach", is_backing: bool = False) -> bool:
        """Simple movement with backwards capability and position checking"""
        # Update current position
        if not self.update_robot_position():
            logger.warning("Lost robot tracking")
            return False
        
        # Check for wall proximity and back up if needed
        if not self.check_wall_proximity():
            logger.error("Wall avoidance failed")
            return False
        
        # Calculate distance and angle to target
        dx = target_pos[0] - self.robot_pos[0]
        dy = target_pos[1] - self.robot_pos[1]
        distance_to_target = math.hypot(dx, dy)
        target_angle = math.degrees(math.atan2(dy, dx))
        
        # Check if we're already close enough - more forgiving for goals
        distance_tolerance = 15 if target_type == "goal" else 5  # Much more forgiving for goal delivery
        if distance_to_target <= distance_tolerance:
            logger.info(f"Already at target {target_pos}")
            return True
        
        # Calculate angle differences for forward and backward movement
        forward_angle_diff = (target_angle - self.robot_heading + 180) % 360 - 180
        backward_angle_diff = (target_angle - self.robot_heading) % 360 - 180
        
        # Decide whether to go forward or backward (choose the option with less turning)
        should_back_up = abs(backward_angle_diff) < abs(forward_angle_diff) and abs(backward_angle_diff) < 90
        
        # CRITICAL: Ball collection AND goal delivery MUST be forward motion only!
        # Also prevent backing up during goal approach to avoid random backing
        if target_type == "collect" or target_type == "goal" or target_type == "approach":
            should_back_up = False
            if target_type == "collect":
                logger.info("Ball collection: forcing forward movement (collector only works when moving forward)")
            elif target_type == "goal":
                logger.info("Goal delivery: forcing forward movement (collector must face goal to deliver balls)")
            elif target_type == "approach":
                logger.info("Goal approach: forcing forward movement (no backing up during goal runs)")
        
        # SAFETY: Check if backing up would be safe for the entire path to target
        if should_back_up and self.walls:
            # Check the full backwards path to target, not just a short backup
            backup_angle_rad = math.radians(self.robot_heading + 180)  # Opposite to current heading
            
            # Check multiple points along the backwards path
            backup_safe = True
            min_safe_distance = 12  # cm - minimum safe distance from walls
            
            # Check points at 10cm intervals along the backwards path
            for check_distance in [10, 20, min(30, distance_to_target)]:
                if check_distance > distance_to_target:
                    break
                    
                test_x = self.robot_pos[0] + check_distance * math.cos(backup_angle_rad)
                test_y = self.robot_pos[1] + check_distance * math.sin(backup_angle_rad)
                
                # Check distance to all walls from this test position
                min_wall_distance = float('inf')
                for wall in self.walls:
                    wall_start = (wall[0], wall[1])
                    wall_end = (wall[2], wall[3])
                    wall_distance = point_to_line_distance((test_x, test_y), wall_start, wall_end)
                    min_wall_distance = min(min_wall_distance, wall_distance)
                
                # If any point along the path is too close to walls, backing up is unsafe
                if min_wall_distance < min_safe_distance:
                    backup_safe = False
                    logger.info(f"Safety override: backing up {check_distance}cm would put robot {min_wall_distance:.1f}cm from wall (need {min_safe_distance}cm)")
                    break
            
            # Also check if robot is already close to walls (corner detection)
            current_min_wall_distance = float('inf')
            for wall in self.walls:
                wall_start = (wall[0], wall[1])
                wall_end = (wall[2], wall[3])
                current_distance = point_to_line_distance(self.robot_pos, wall_start, wall_end)
                current_min_wall_distance = min(current_min_wall_distance, current_distance)
            
            # If robot is already very close to walls (in corner), always go forward
            if current_min_wall_distance < 20:
                backup_safe = False
                logger.info(f"Safety override: robot in corner/near wall ({current_min_wall_distance:.1f}cm), forcing forward movement")
            
            if not backup_safe:
                should_back_up = False
        
        if should_back_up:
            # Back up to target
            angle_diff = backward_angle_diff
            logger.info(f"Backing up to target (requires {abs(angle_diff):.1f}° turn vs {abs(forward_angle_diff):.1f}° forward)")
        else:
            # Go forward to target  
            angle_diff = forward_angle_diff
            logger.info(f"Going forward to target (requires {abs(angle_diff):.1f}° turn)")
        
        # Turn to face the right direction
        if abs(angle_diff) > 2:  # Turn for any significant angle difference (reduced from 10 to 2 degrees)
            logger.info(f"Turning {angle_diff:.1f} degrees")
            if not self.turn(angle_diff):
                return False
            # Update position after turn
            self.update_robot_position()
        
        # Move to target with position checking for longer distances
        # For ball collection, use stepped approach for better control
        collection_threshold = 25 if target_type == "collect" else 15
        if distance_to_target > collection_threshold:
            return self._move_with_position_check(target_pos, target_type, should_back_up, distance_to_target)
        else:
            # Short movement - just go directly
            if target_type == "collect":
                logger.info(f"Collecting ball at {distance_to_target:.1f}cm")
                return self.collect(distance_to_target)
            else:
                move_distance = -distance_to_target if should_back_up else distance_to_target
                action = "Backing up" if should_back_up else "Moving forward"
                logger.info(f"{action} {abs(move_distance):.1f}cm")
                return self.move(move_distance)
    
    def _move_with_position_check(self, target_pos: Tuple[float, float], target_type: str, is_backing: bool, total_distance: float) -> bool:
        """Move to target in steps with position checking and direction recalculation"""
        step_size = 10  # Move in 10cm increments for longer distances
        steps_taken = 0
        
        # Adaptive max steps based on distance (with generous buffer for course corrections)
        estimated_steps = int(total_distance / step_size) + 5  # +5 for course corrections
        max_steps = max(50, estimated_steps)  # At least 15 steps, more if needed
        
        logger.info(f"Starting stepped movement: {total_distance:.1f}cm distance, max {max_steps} steps allowed")
        
        # Progress tracking to detect if robot is stuck
        last_distance = total_distance
        no_progress_count = 0
        
        while steps_taken < max_steps:
            # Update current position
            if not self.update_robot_position():
                logger.warning("Lost robot tracking during movement")
                return False
            
            # Check for wall proximity before each step
            if not self.check_wall_proximity():
                logger.warning("Wall avoidance interrupted movement")
                # Recalculate position after wall backup
                if not self.update_robot_position():
                    return False
            
            # Calculate remaining distance and direction
            dx = target_pos[0] - self.robot_pos[0]
            dy = target_pos[1] - self.robot_pos[1]
            remaining_distance = math.hypot(dx, dy)
            target_angle = math.degrees(math.atan2(dy, dx))
            
            # Check if we've reached the target
            distance_tolerance = 15 if target_type == "goal" else 12  # Much more forgiving for goal delivery
            if remaining_distance <= distance_tolerance:
                logger.info(f"Reached target with position checking")
                return True
            
            # Final approach for ball collection (MUST be forward motion)
            # Start collection earlier to avoid pushing balls
            if target_type == "collect" and remaining_distance <= 22:  # Increased from 15cm to 22cm
                # NO MORE TURNING for final approach - just collect straight ahead
                logger.info(f"Final collection approach: {remaining_distance:.1f}cm (starting collection early)")
                return self.collect(remaining_distance)
            
            # Recalculate direction based on current position
            angle_diff = (target_angle - self.robot_heading + 180) % 360 - 180
            
            # For backing moves, face away from target
            if is_backing:
                angle_diff = (angle_diff + 180) % 360 - 180
            
            # Adjust heading for better precision - allow frequent small corrections
            # Robot can turn as much as needed to face target directly
            angle_threshold = 3 if target_type == "goal" else 2  # Very precise angle tolerance for better accuracy
            if abs(angle_diff) > angle_threshold:
                # Turn directly to face target - no artificial limitations
                logger.info(f"Adjusting direction: turning {angle_diff:.1f} degrees to face target")
                if not self.turn(angle_diff):
                    return False
                self.update_robot_position()
            
            # Calculate next step distance
            step_distance = min(step_size, remaining_distance)
            
            # Move one step
            move_distance = -step_distance if is_backing else step_distance
            action = "Backing" if is_backing else "Forward"
            logger.info(f"{action} step: {step_distance:.1f}cm (remaining: {remaining_distance:.1f}cm)")
            
            if not self.move(move_distance):
                logger.error("Movement step failed")
                return False
            
            steps_taken += 1
            
            # Update live tracking visualization after each step
            if not self.update_live_tracking(target_pos, remaining_distance, steps_taken, max_steps):
                logger.warning("Movement interrupted by user")
                return False
            
            # Brief pause for position update
            time.sleep(0.3)  # Slightly longer pause for better position tracking
        
        # Final position check
        if not self.update_robot_position():
            logger.warning("Lost robot tracking at end of movement")
            return False
            
        final_dx = target_pos[0] - self.robot_pos[0]
        final_dy = target_pos[1] - self.robot_pos[1]
        final_distance = math.hypot(final_dx, final_dy)
        
        logger.warning(f"Max steps reached, still {final_distance:.1f}cm from target")
        return False

    def deliver_balls(self) -> bool:
        """Run collector in reverse to deliver balls"""
        try:
            logger.info(f"Sending COLLECT_REVERSE command for {self.delivery_time} seconds")
            result = self.send_command("COLLECT_REVERSE", duration=self.delivery_time)
            
            # Enable automatic mode after first delivery attempt (regardless of success)
            if not self.automatic_mode:
                self.automatic_mode = True
                logger.info("🤖 AUTOMATIC MODE ENABLED - Will auto-execute future paths")
            
            if result:
                logger.info("COLLECT_REVERSE command executed successfully")
            else:
                logger.warning("COLLECT_REVERSE command returned failure, but continuing in automatic mode")
            return result
        except Exception as e:
            logger.error("Exception during ball delivery: {}".format(e))
            # Still enable automatic mode even if delivery failed
            if not self.automatic_mode:
                self.automatic_mode = True
                logger.info("🤖 AUTOMATIC MODE ENABLED after delivery attempt (despite error)")
            return False

    def calculate_goal_approach_path(self, current_pos, current_heading):
        """Calculate path to goal via center staging area with obstacle avoidance"""
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
            goal_x = -2  # Position collector 2cm into goal opening for better delivery
            # Staging area much closer to goal since we can tank turn
            staging_x = 30  # Only 30cm from left edge (much closer than 25% = 45cm)
        else:  # Right side goal
            goal_x = FIELD_WIDTH_CM + 2  # Position collector 2cm into goal opening for better delivery
            # Staging area much closer to goal since we can tank turn
            staging_x = FIELD_WIDTH_CM - 30  # Only 30cm from right edge (much closer than 75% = 135cm)
        
        # Staging area Y coordinate - aligned with goal for straight final approach
        staging_y = goal_y
        
        # Calculate distance to staging area
        distance_to_staging = math.hypot(staging_x - current_pos[0], staging_y - current_pos[1])
        
        path = []
        logger.info(f"Goal planning: current_pos=({current_pos[0]:.1f}, {current_pos[1]:.1f}), staging=({staging_x:.1f}, {staging_y:.1f}), distance={distance_to_staging:.1f}cm")
        
        # Plan path to staging area with obstacle avoidance
        if distance_to_staging > 20:  # If more than 20cm from staging area
            logger.info(f"Planning route via center staging area at ({staging_x:.1f}, {staging_y:.1f})")
            
            # Get waypoints to staging area around obstacle
            staging_waypoints = self.plan_path_around_obstacle(current_pos, (staging_x, staging_y))
            
            # Add all waypoints as approach points
            for waypoint in staging_waypoints:
                path.append({
                    'type': 'approach',
                    'pos': waypoint
                })
            
            # Plan final path from staging to goal with obstacle avoidance
            goal_waypoints = self.plan_path_around_obstacle((staging_x, staging_y), (goal_x, goal_y))
            
            # Add goal waypoints (skip first one since it's the staging area)
            for waypoint in goal_waypoints[1:]:  # Skip first waypoint (staging area)
                if waypoint == (goal_x, goal_y):
                    path.append({
                        'type': 'goal',
                        'pos': waypoint
                    })
                else:
                    path.append({
                        'type': 'approach',
                        'pos': waypoint
                    })
        else:
            logger.info("Already near staging area, going directly to goal with obstacle avoidance")
            
            # Plan direct path to goal with obstacle avoidance
            goal_waypoints = self.plan_path_around_obstacle(current_pos, (goal_x, goal_y))
            
            # Add all waypoints
            for waypoint in goal_waypoints:
                if waypoint == (goal_x, goal_y):
                    path.append({
                        'type': 'goal',
                        'pos': waypoint
                    })
                else:
                    path.append({
                        'type': 'approach',
                        'pos': waypoint
                    })
        
        # SAFETY: Ensure we always have a goal point in the path
        has_goal = any(p['type'] == 'goal' for p in path)
        if not has_goal:
            logger.warning("No goal point found in path! Adding direct goal approach...")
            path.append({
                'type': 'goal',
                'pos': (goal_x, goal_y)
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
                
                # Replan if:
                # 1. Robot markers are detected AND not skipping processing AND
                # 2. Either:
                #    a) It's been at least X seconds since last plan OR
                #    b) Ball positions have changed significantly OR
                #    c) Robot has moved significantly OR
                #    d) No current path exists
                should_replan = False
                if self.robot_pos and not skip_heavy_processing:
                    # Reduce wait time if no path is currently planned (for continuous operation)
                    min_plan_interval = 1 if not path else 3  # 1 second if no path, 3 seconds if path exists
                    
                    if current_time - last_plan_time >= min_plan_interval:
                        should_replan = True
                    elif balls and last_plan_positions:
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
                    
                    # Plan path through all balls in the batch
                    path = []
                    current_pos = self.robot_pos
                    current_heading = self.robot_heading

                    if balls:
                        logger.info(f"Planning to collect {len(balls)} balls from position {self.robot_pos}")
                        
                        # Enable automatic mode when balls are detected for the first time
                        if not self.automatic_mode:
                            self.automatic_mode = True
                            logger.info("🤖 AUTOMATIC MODE ENABLED - Robot will now operate autonomously")
                        
                        # Sort balls by distance from current robot position
                        balls.sort(key=lambda b: math.hypot(
                            b[0] - self.robot_pos[0],
                            b[1] - self.robot_pos[1]
                        ))

                        # Take up to MAX_BALLS_PER_TRIP closest balls
                        current_batch = balls[:MAX_BALLS_PER_TRIP]
                        remaining_balls = current_batch.copy()

                        # Add balls to path with obstacle avoidance
                        while remaining_balls:
                            closest_ball = min(remaining_balls, 
                                             key=lambda b: math.hypot(
                                                 b[0] - current_pos[0],
                                                 b[1] - current_pos[1]
                                             ))
                            
                            ball_pos = (closest_ball[0], closest_ball[1])
                            
                            # Plan path around obstacle if needed
                            waypoints = self.plan_path_around_obstacle(current_pos, ball_pos)
                            
                            # Add waypoint approach points if detour is needed
                            for i, waypoint in enumerate(waypoints[:-1]):  # All except the last (ball) position
                                path.append({
                                    'type': 'approach',
                                    'pos': waypoint
                                })
                            
                            # Add the ball collection point
                            path.append({
                                'type': 'collect',
                                'pos': ball_pos,
                                'ball_type': closest_ball[2]
                            })
                            
                            current_pos = ball_pos
                            current_heading = self.robot_heading
                            remaining_balls.remove(closest_ball)
                    else:
                        logger.info(f"No balls detected - planning delivery run from position {self.robot_pos}")

                    # ALWAYS add goal approach path for delivery (whether we collected balls or not)
                    goal_path = self.calculate_goal_approach_path(current_pos, current_heading)
                    path.extend(goal_path)
                    
                    balls_in_path = len([p for p in path if p['type'] == 'collect'])
                    logger.info(f"Added goal delivery path with {len(goal_path)} waypoints (balls in path: {balls_in_path})")
                    
                    # If automatic mode is enabled, execute path immediately
                    if self.automatic_mode and path:
                        logger.info("🤖 Auto-executing path in automatic mode...")
                        # Add a small delay to show the path briefly
                        time.sleep(0.5)
                        # Trigger execution by simulating Enter key press
                        key = 13  # Enter key code

                # Prepare the display frame
                display_frame = frame.copy()
                
                # Always show path preview when path exists
                if path:
                    display_frame = self.draw_path(display_frame, path)
                    # Add path ready indicator
                    if self.automatic_mode:
                        cv2.putText(display_frame, "AUTO MODE: Executing path automatically...", 
                                  (10, display_frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.7, (0, 255, 0), 2)
                    else:
                        cv2.putText(display_frame, "PATH READY - Press ENTER to execute", 
                                  (10, display_frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.7, (0, 255, 0), 2)
                
                # Get key input (unless auto-execution is happening)
                if not (self.automatic_mode and path):
                    key = cv2.waitKey(1) & 0xFF
                else:
                    # In automatic mode with path, don't override the auto-execution
                    if 'key' not in locals():
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
                
                # Show planned path count and operation status
                if path:
                    ball_count = len([p for p in path if p['type'] == 'collect'])
                    cv2.putText(display_frame, f"Path planned: {ball_count} balls",
                              (10, y + 10), cv2.FONT_HERSHEY_SIMPLEX,
                              0.5, (0, 255, 0), 1)
                    if self.automatic_mode:
                        cv2.putText(display_frame, "AUTOMATIC MODE: Fully autonomous operation",
                                  (10, y + 30), cv2.FONT_HERSHEY_SIMPLEX,
                                  0.4, (0, 255, 0), 1)
                    else:
                        cv2.putText(display_frame, "MANUAL MODE: Press ENTER to execute (auto when balls detected)",
                                  (10, y + 30), cv2.FONT_HERSHEY_SIMPLEX,
                                  0.4, (0, 255, 255), 1)
                else:
                    cv2.putText(display_frame, "Scanning for balls...",
                              (10, y + 10), cv2.FONT_HERSHEY_SIMPLEX,
                              0.5, (255, 255, 0), 1)
                
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
                            for point in path:
                                target_pos = point['pos']
                                
                                if point['type'] == 'collect':
                                    logger.info(f"Collecting {point['ball_type']} ball")
                                    if not self.move_to_target_simple(target_pos, "collect"):
                                        raise Exception("Ball collection failed")
                                elif point['type'] == 'approach':
                                    logger.info(f"Moving to staging area in center")
                                    if not self.move_to_target_simple(target_pos, "approach"):
                                        raise Exception("Approach movement failed")
                                elif point['type'] == 'goal':
                                    logger.info(f"Moving to goal")
                                    if not self.move_to_target_simple(target_pos, "goal"):
                                        raise Exception("Goal movement failed")
                                    
                                    # Final alignment before delivery - face goal opening directly
                                    logger.info("Final alignment for goal delivery")
                                    goal_side = self.goal_ranges[self.selected_goal][0][0]  # X coordinate of goal
                                    if goal_side == 0:  # Left goal
                                        target_angle = 180  # Face left (west)
                                    else:  # Right goal  
                                        target_angle = 0    # Face right (east)
                                    
                                    # Calculate angle difference and align if needed
                                    angle_diff = (target_angle - self.robot_heading + 180) % 360 - 180
                                    if abs(angle_diff) > 5:  # Only adjust if more than 5 degrees off
                                        logger.info(f"Aligning with goal: turning {angle_diff:.1f} degrees")
                                        if not self.turn(angle_diff):
                                            logger.warning("Final alignment failed, but continuing")
                                        else:
                                            self.update_robot_position()
                                    
                                    # Move collector into goal for proper delivery (additional 8cm forward)
                                    logger.info("Moving collector into goal for proper delivery")
                                    if not self.move(8):
                                        raise Exception("Failed to move collector into goal")
                                    
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
                                    if not self.move(-20):
                                        logger.error("Failed to back up after delivery - this could cause issues")
                                        # Don't raise exception here to avoid stopping the whole sequence
                                    
                                    if delivery_success:
                                        logger.info("✅ Delivery sequence complete, ready for next collection")
                                    else:
                                        logger.info("⚠️ Delivery had issues but backup completed, ready for next collection")
                                
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
                                    logger.info(f"🎯 Found {len(remaining_balls)} remaining balls - automatically continuing collection")
                                    
                                    # Enable automatic mode when balls are found for the first time
                                    if not self.automatic_mode:
                                        self.automatic_mode = True
                                        logger.info("🤖 AUTOMATIC MODE ENABLED - Robot will now operate autonomously")
                                    
                                    # Force immediate replanning by resetting timing
                                    last_plan_time = 0
                                    last_plan_positions = []
                                    # Continue main loop to plan and execute new path automatically
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

    def update_live_tracking(self, target_pos, remaining_distance, current_step, max_steps):
        """Update live visualization during movement with current progress"""
        try:
            # Get fresh frame and update robot position
            ret, frame = self.get_fresh_frame()
            if not ret:
                return True
                
            # Update robot position from markers
            if not self.update_robot_position(frame):
                return True
            
            # Draw all the standard overlays
            frame = self.draw_walls(frame)
            frame = self.draw_robot_markers(frame)
            frame = self.draw_status(frame)
            
            # Draw target position
            if self.homography_matrix is not None:
                target_cm = np.array([[[target_pos[0], target_pos[1]]]], dtype="float32")
                target_px = cv2.perspectiveTransform(target_cm, self.homography_matrix)[0][0].astype(int)
                
                # Draw target with pulsing effect based on step count
                pulse_size = 8 + int(3 * math.sin(current_step * 0.5))
                cv2.circle(frame, tuple(target_px), pulse_size, (0, 255, 255), 2)  # Yellow target
                cv2.putText(frame, "TARGET", (target_px[0] + 15, target_px[1] - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Draw line from heading marker (collector position) to target
                green_center, purple_center = self.detect_markers(frame)
                if purple_center:
                    # Use purple heading marker as starting point (collector position)
                    cv2.line(frame, purple_center, tuple(target_px), (0, 255, 255), 2)
                    # Draw a small arrow at purple center pointing to target
                    cv2.circle(frame, purple_center, 3, (0, 255, 255), -1)
                else:
                    # Fallback to robot base position if purple marker not detected
                    robot_cm = np.array([[[self.robot_pos[0], self.robot_pos[1]]]], dtype="float32")
                    robot_px = cv2.perspectiveTransform(robot_cm, self.homography_matrix)[0][0].astype(int)
                    cv2.line(frame, tuple(robot_px), tuple(target_px), (0, 255, 255), 1)
            
            # Progress information overlay
            progress_pct = int((current_step / max_steps) * 100) if max_steps > 0 else 0
            status_lines = [
                f"LIVE MOVEMENT TRACKING",
                f"Target: ({target_pos[0]:.1f}, {target_pos[1]:.1f}) cm",
                f"Robot: ({self.robot_pos[0]:.1f}, {self.robot_pos[1]:.1f}) cm", 
                f"Remaining: {remaining_distance:.1f} cm",
                f"Step: {current_step}/{max_steps} ({progress_pct}%)",
                f"Press ESC to stop movement"
            ]
            
            # Draw semi-transparent background for status
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, frame.shape[0] - 150), (400, frame.shape[0] - 10), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Draw status text
            y_start = frame.shape[0] - 135
            for i, line in enumerate(status_lines):
                color = (0, 255, 0) if i == 0 else (255, 255, 255)  # Green for title, white for info
                cv2.putText(frame, line, (15, y_start + i * 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Progress bar
            bar_x, bar_y = 15, frame.shape[0] - 25
            bar_width, bar_height = 200, 10
            
            # Background bar
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
            
            # Progress fill
            fill_width = int((progress_pct / 100) * bar_width)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), (0, 255, 0), -1)
            
            # Progress percentage text
            cv2.putText(frame, f"{progress_pct}%", (bar_x + bar_width + 10, bar_y + 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Update display
            cv2.imshow("Path Planning", frame)
            
            # Check for user input to stop movement
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                logger.warning("Movement stopped by user (ESC pressed)")
                return False
                
        except Exception as e:
            logger.warning(f"Live tracking update failed: {e}")
            
        return True



if __name__ == "__main__":
    collector = BallCollector()
    collector.run() 