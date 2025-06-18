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
ROBOFLOW_API_KEY = "qJTLU5ku2vpBGQUwjBx2"
RF_WORKSPACE = "cdio-nczdp"
RF_PROJECT = "cdio-golfbot2025" 
RF_VERSION = 13

# Color detection ranges (HSV)
GREEN_LOWER = np.array([35, 50, 50], dtype=np.uint8)
GREEN_UPPER = np.array([85, 255, 255], dtype=np.uint8)
PINK_LOWER = np.array([145, 50, 50], dtype=np.uint8)
PINK_UPPER = np.array([175, 255, 255], dtype=np.uint8)
# Orange ball HSV range (typical range for orange)
ORANGE_LOWER = np.array([5, 150, 150], dtype=np.uint8)
ORANGE_UPPER = np.array([15, 255, 255], dtype=np.uint8)

# Marker dimensions
GREEN_MARKER_WIDTH_CM = 20  # Width of green base sheet
PINK_MARKER_WIDTH_CM = 5    # Width of pink direction marker

# Wall configuration
WALL_SAFETY_MARGIN = 1  # cm, minimum distance to keep from walls
GOAL_WIDTH = 30  # cm, width of each goal area

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

# Goal configuration
LEFT_GOAL_CENTER = (0, FIELD_HEIGHT_CM / 2)  # Left goal center coordinates
RIGHT_GOAL_CENTER = (FIELD_WIDTH_CM, FIELD_HEIGHT_CM / 2)  # Right goal center coordinates

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
        
        # Goal configuration
        self.goals = [
            {
                'position': LEFT_GOAL_CENTER,
                'width': GOAL_WIDTH,
                'approach_distance': 20
            },
            {
                'position': RIGHT_GOAL_CENTER,
                'width': GOAL_WIDTH,
                'approach_distance': 20
            }
        ]
        self.current_goal = None  # Will be set to nearest goal during path planning

        # Add ball collection counter
        self.collected_balls = 0
        self.delivery_time = 2.0  # Seconds to run collector in reverse
        
        # Field HSV statistics
        self.field_hsv_mean = None
        self.field_value_threshold = None

    def calculate_field_hsv_stats(self, frame):
        """Calculate average HSV values for the field area"""
        if self.homography_matrix is None:
            return
            
        # Create a mask for the field area
        field_corners_cm = np.array([
            [[0, 0]],
            [[FIELD_WIDTH_CM, 0]],
            [[FIELD_WIDTH_CM, FIELD_HEIGHT_CM]],
            [[0, FIELD_HEIGHT_CM]]
        ], dtype="float32")
        field_corners_px = cv2.perspectiveTransform(field_corners_cm, self.homography_matrix)
        field_corners_px = field_corners_px.astype(int)
        
        # Create field mask
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [field_corners_px], 255)
        
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Calculate mean HSV values for the field area
        self.field_hsv_mean = cv2.mean(hsv, mask=mask)[:3]
        # Set value threshold above the mean field value
        self.field_value_threshold = min(255, self.field_hsv_mean[2] * 1.3)  # 30% above mean
        
        logger.info(f"Field HSV mean: {self.field_hsv_mean}")
        logger.info(f"Value threshold: {self.field_value_threshold}")

    def detect_markers(self, frame):
        """Detect green base and pink direction markers"""
        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Detect green base marker
        green_mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
        green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Detect pink direction marker
        pink_mask = cv2.inRange(hsv, PINK_LOWER, PINK_UPPER)
        pink_contours, _ = cv2.findContours(pink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find largest green contour (base marker)
        green_center = None
        if green_contours:
            largest_green = max(green_contours, key=cv2.contourArea)
            if cv2.contourArea(largest_green) > 100:  # Minimum area threshold
                M = cv2.moments(largest_green)
                if M["m00"] != 0:
                    green_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        
        # Find largest pink contour (direction marker)
        pink_center = None
        if pink_contours:
            largest_pink = max(pink_contours, key=cv2.contourArea)
            if cv2.contourArea(largest_pink) > 50:  # Minimum area threshold
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
        """Move forward/backward by distance_cm"""
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

    def is_within_field(self, x_cm, y_cm):
        """Check if a point is within the valid field boundaries"""
        # Add a small margin (1cm) to avoid edge cases
        margin = 1
        return (margin <= x_cm <= FIELD_WIDTH_CM - margin and 
                margin <= y_cm <= FIELD_HEIGHT_CM - margin)

    def detect_balls(self) -> List[Tuple[float, float, str]]:
        """Detect white and orange balls using HSV thresholding"""
        ret, frame = self.cap.read()
        if not ret or self.homography_matrix is None:
            return []

        # Update field stats periodically
        if self.field_hsv_mean is None:
            self.calculate_field_hsv_stats(frame)
            
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        balls = []
        
        # Detect white balls (high value, low saturation)
        if self.field_value_threshold is not None:
            white_lower = np.array([0, 0, self.field_value_threshold], dtype=np.uint8)
            white_upper = np.array([180, 30, 255], dtype=np.uint8)
            white_mask = cv2.inRange(hsv, white_lower, white_upper)
            
            # Clean up mask
            kernel = np.ones((5,5), np.uint8)
            white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
            white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
            
            # Find white ball contours
            white_contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, 
                                               cv2.CHAIN_APPROX_SIMPLE)
            
            # Process white balls
            for contour in white_contours:
                area = cv2.contourArea(contour)
                if 100 < area < 2000:  # Adjust these thresholds based on your setup
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        x_px = int(M["m10"] / M["m00"])
                        y_px = int(M["m01"] / M["m00"])
                        
                        # Convert to cm using homography
                        pt_px = np.array([[[x_px, y_px]]], dtype="float32")
                        pt_cm = cv2.perspectiveTransform(pt_px, 
                                                       np.linalg.inv(self.homography_matrix))[0][0]
                        x_cm, y_cm = pt_cm
                        
                        # Check if ball is within valid area
                        if (self.is_within_field(x_cm, y_cm) and
                            not (IGNORED_AREA["x_min"] <= x_cm <= IGNORED_AREA["x_max"] and
                                 IGNORED_AREA["y_min"] <= y_cm <= IGNORED_AREA["y_max"])):
                            balls.append((x_cm, y_cm, "white"))
        
        # Detect orange balls
        orange_mask = cv2.inRange(hsv, ORANGE_LOWER, ORANGE_UPPER)
        
        # Clean up mask
        kernel = np.ones((5,5), np.uint8)
        orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_OPEN, kernel)
        orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find orange ball contours
        orange_contours, _ = cv2.findContours(orange_mask, cv2.RETR_EXTERNAL, 
                                            cv2.CHAIN_APPROX_SIMPLE)
        
        # Process orange balls
        for contour in orange_contours:
            area = cv2.contourArea(contour)
            if 100 < area < 2000:  # Adjust these thresholds based on your setup
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    x_px = int(M["m10"] / M["m00"])
                    y_px = int(M["m01"] / M["m00"])
                    
                    # Convert to cm using homography
                    pt_px = np.array([[[x_px, y_px]]], dtype="float32")
                    pt_cm = cv2.perspectiveTransform(pt_px, 
                                                   np.linalg.inv(self.homography_matrix))[0][0]
                    x_cm, y_cm = pt_cm
                    
                    # Check if ball is within valid area
                    if (self.is_within_field(x_cm, y_cm) and
                        not (IGNORED_AREA["x_min"] <= x_cm <= IGNORED_AREA["x_max"] and
                             IGNORED_AREA["y_min"] <= y_cm <= IGNORED_AREA["y_max"])):
                        balls.append((x_cm, y_cm, "orange"))
        
        return balls

    def draw_debug_view(self, frame):
        """Draw debug visualization of ball detection"""
        if self.homography_matrix is None:
            return frame
            
        debug_frame = frame.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Draw white ball detection
        if self.field_value_threshold is not None:
            white_lower = np.array([0, 0, self.field_value_threshold], dtype=np.uint8)
            white_upper = np.array([180, 30, 255], dtype=np.uint8)
            white_mask = cv2.inRange(hsv, white_lower, white_upper)
            
            white_debug = cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR)
            white_debug = cv2.addWeighted(frame, 0.7, white_debug, 0.3, 0)
            
            # Draw white ball contours
            white_contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, 
                                               cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(white_debug, white_contours, -1, (0, 255, 0), 2)
            
            # Show white detection window
            cv2.imshow("White Ball Detection", white_debug)
        
        # Draw orange ball detection
        orange_mask = cv2.inRange(hsv, ORANGE_LOWER, ORANGE_UPPER)
        orange_debug = cv2.cvtColor(orange_mask, cv2.COLOR_GRAY2BGR)
        orange_debug = cv2.addWeighted(frame, 0.7, orange_debug, 0.3, 0)
        
        # Draw orange ball contours
        orange_contours, _ = cv2.findContours(orange_mask, cv2.RETR_EXTERNAL, 
                                            cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(orange_debug, orange_contours, -1, (0, 255, 0), 2)
        
        # Show orange detection window
        cv2.imshow("Orange Ball Detection", orange_debug)
        
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
        """Draw the planned path on the frame"""
        # First draw the field
        frame = self.draw_field(frame)
        
        if path:
            overlay = frame.copy()
            
            # Colors for different point types
            colors = {
                'approach': (0, 255, 255),  # Yellow
                'collect': (0, 255, 0),     # Green
                'goal': (255, 0, 0)         # Red
            }
            
            # Draw lines between points
            for i in range(len(path) - 1):
                # Convert cm to pixels for both points
                start_cm = np.array([[[path[i]['pos'][0], path[i]['pos'][1]]]], dtype="float32")
                end_cm = np.array([[[path[i+1]['pos'][0], path[i+1]['pos'][1]]]], dtype="float32")
                
                start_px = cv2.perspectiveTransform(start_cm, self.homography_matrix)[0][0].astype(int)
                end_px = cv2.perspectiveTransform(end_cm, self.homography_matrix)[0][0].astype(int)
                
                # Draw line with arrow
                cv2.arrowedLine(overlay, tuple(start_px), tuple(end_px), (150, 150, 150), 2)
            
            # Draw points
            for point in path:
                pt_cm = np.array([[[point['pos'][0], point['pos'][1]]]], dtype="float32")
                pt_px = cv2.perspectiveTransform(pt_cm, self.homography_matrix)[0][0].astype(int)
                
                # Draw point
                color = colors[point['type']]
                cv2.circle(overlay, tuple(pt_px), 5, color, -1)
                cv2.circle(overlay, tuple(pt_px), 7, (255, 255, 255), 1)
                
                # Draw direction for approach and goal points
                if point['type'] in ['approach', 'goal']:
                    angle_rad = math.radians(point['angle'])
                    end_x = pt_px[0] + int(20 * math.cos(angle_rad))
                    end_y = pt_px[1] + int(20 * math.sin(angle_rad))
                    cv2.arrowedLine(overlay, tuple(pt_px), (end_x, end_y), color, 2)
                
                # Add labels
                if point['type'] == 'collect':
                    cv2.putText(overlay, point.get('ball_type', ''), 
                              (pt_px[0] + 10, pt_px[1] + 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Blend overlay with original frame
            alpha = 0.6
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            
            # Draw robot position and heading
            robot_cm = np.array([[[self.robot_pos[0], self.robot_pos[1]]]], dtype="float32")
            robot_px = cv2.perspectiveTransform(robot_cm, self.homography_matrix)[0][0].astype(int)
            
            # Robot circle
            cv2.circle(frame, tuple(robot_px), 8, (255, 0, 0), -1)
            cv2.circle(frame, tuple(robot_px), 10, (255, 255, 255), 1)
            
            # Robot heading
            angle_rad = math.radians(self.robot_heading)
            end_x = robot_px[0] + int(25 * math.cos(angle_rad))
            end_y = robot_px[1] + int(25 * math.sin(angle_rad))
            cv2.arrowedLine(frame, tuple(robot_px), (end_x, end_y), (255, 0, 0), 2)
            
            # Add current goal indicator if one is selected
            if self.current_goal:
                goal_x, goal_y = self.current_goal['position']
                goal_cm = np.array([[[goal_x, goal_y]]], dtype="float32")
                goal_px = cv2.perspectiveTransform(goal_cm, self.homography_matrix)[0][0].astype(int)
                cv2.circle(frame, tuple(goal_px), 12, (0, 255, 0), 2)
        
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

    def deliver_balls(self) -> bool:
        """Run collector in reverse to deliver balls"""
        try:
            return self.send_command("COLLECT_REVERSE", duration=self.delivery_time)
        except Exception as e:
            logger.error("Failed to deliver balls: {}".format(e))
            return False

    def calculate_goal_approach_path(self, current_pos, current_heading):
        """Calculate a smooth path to the nearest accessible goal"""
        # Find nearest accessible goal
        self.current_goal = self.find_nearest_goal(current_pos)
        if not self.current_goal:
            logger.warning("No accessible goals found!")
            return []

        goal_x, goal_y = self.current_goal['position']
        approach_dist = self.current_goal['approach_distance']
        
        # Calculate direct distance and angle to goal
        dx = goal_x - current_pos[0]
        dy = goal_y - current_pos[1]
        direct_distance = math.hypot(dx, dy)
        angle_to_goal = math.degrees(math.atan2(dy, dx))
        
        # If we're already well-aligned with the goal (within 15 degrees), just go straight
        angle_diff = (angle_to_goal - current_heading + 180) % 360 - 180
        if abs(angle_diff) < 15 and not check_wall_collision(current_pos, 
                                                           (goal_x, goal_y), 
                                                           self.walls, 
                                                           WALL_SAFETY_MARGIN):
            return [{
                'type': 'goal',
                'pos': (goal_x - approach_dist if goal_x == FIELD_WIDTH_CM else goal_x + approach_dist, 
                       goal_y),
                'angle': 0 if goal_x == FIELD_WIDTH_CM else 180  # Face into goal
            }]
        
        # Otherwise, calculate a curved approach path
        # First point: swing out to create better approach angle
        swing_distance = min(50, direct_distance * 0.4)
        
        # If we're to the left of the goal, swing right, and vice versa
        swing_direction = 1 if current_pos[1] < goal_y else -1
        swing_angle = current_heading + (45 * swing_direction)
        
        # Calculate swing point
        swing_x = current_pos[0] + swing_distance * math.cos(math.radians(swing_angle))
        swing_y = current_pos[1] + swing_distance * math.sin(math.radians(swing_angle))
        
        # Second point: intermediate approach point
        approach_x = goal_x - (approach_dist * 1.5) if goal_x == FIELD_WIDTH_CM else goal_x + (approach_dist * 1.5)
        approach_y = goal_y
        
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
                'angle': 0 if goal_x == FIELD_WIDTH_CM else 180
            })
        
        final_x = goal_x - approach_dist if goal_x == FIELD_WIDTH_CM else goal_x + approach_dist
        if not check_wall_collision((approach_x, approach_y), (final_x, goal_y), self.walls, WALL_SAFETY_MARGIN):
            path.append({
                'type': 'goal',
                'pos': (final_x, goal_y),
                'angle': 0 if goal_x == FIELD_WIDTH_CM else 180
            })
        
        return path

    def draw_field(self, frame):
        """Draw field elements including walls, goals, and markers"""
        if self.homography_matrix is None:
            return frame

        # Create overlay for semi-transparent elements
        overlay = frame.copy()
        
        # Draw field outline with fill
        field_corners_cm = np.array([
            [[0, 0]],
            [[FIELD_WIDTH_CM, 0]],
            [[FIELD_WIDTH_CM, FIELD_HEIGHT_CM]],
            [[0, FIELD_HEIGHT_CM]]
        ], dtype="float32")
        field_corners_px = cv2.perspectiveTransform(field_corners_cm, self.homography_matrix)
        field_corners_px = field_corners_px.astype(int)
        
        # Fill outside area with dark overlay
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [field_corners_px], 255)
        outside_mask = cv2.bitwise_not(mask)
        frame[outside_mask > 0] = frame[outside_mask > 0] // 3  # Darken outside area
        
        # Draw field border
        cv2.polylines(overlay, [field_corners_px], True, (255, 255, 255), 2)

        # Draw goals
        for goal in self.goals:
            # Calculate goal area corners
            goal_x, goal_y = goal['position']
            half_width = goal['width'] / 2
            goal_corners_cm = np.array([
                [[goal_x - 10, goal_y - half_width]],  # Add depth to goal visualization
                [[goal_x + 10, goal_y - half_width]],
                [[goal_x + 10, goal_y + half_width]],
                [[goal_x - 10, goal_y + half_width]]
            ], dtype="float32")
            goal_corners_px = cv2.perspectiveTransform(goal_corners_cm, self.homography_matrix)
            goal_corners_px = goal_corners_px.astype(int)
            
            # Draw goal area
            cv2.fillPoly(overlay, [goal_corners_px], (0, 255, 0, 128))
            cv2.polylines(overlay, [goal_corners_px], True, (0, 255, 0), 2)
            
            # Add goal label
            goal_center_cm = np.array([[[goal_x, goal_y]]], dtype="float32")
            goal_center_px = cv2.perspectiveTransform(goal_center_cm, self.homography_matrix)[0][0].astype(int)
            cv2.putText(overlay, "GOAL", 
                       (goal_center_px[0] - 20, goal_center_px[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw walls with safety margins
        if self.walls:
            for wall in self.walls:
                # Convert wall endpoints from cm to pixels
                start_cm = np.array([[[wall[0], wall[1]]]], dtype="float32")
                end_cm = np.array([[[wall[2], wall[3]]]], dtype="float32")
                
                start_px = cv2.perspectiveTransform(start_cm, self.homography_matrix)[0][0].astype(int)
                end_px = cv2.perspectiveTransform(end_cm, self.homography_matrix)[0][0].astype(int)
                
                # Draw the wall line
                cv2.line(overlay, tuple(start_px), tuple(end_px), (0, 0, 255), 2)
                
                # Draw safety margin area
                margin_pts = []
                angle = math.atan2(end_px[1] - start_px[1], end_px[0] - start_px[0])
                margin_px = int(WALL_SAFETY_MARGIN * 10)  # Convert cm to approximate pixels
                
                dx = int(margin_px * math.sin(angle))
                dy = int(margin_px * math.cos(angle))
                
                margin_pts = np.array([
                    [start_px[0] - dx, start_px[1] + dy],
                    [end_px[0] - dx, end_px[1] + dy],
                    [end_px[0] + dx, end_px[1] - dy],
                    [start_px[0] + dx, start_px[1] - dy]
                ], np.int32)
                
                cv2.fillPoly(overlay, [margin_pts], (0, 0, 255, 64))

        # Blend overlay with original frame
        alpha = 0.4
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        return frame

    def find_nearest_goal(self, position):
        """Find the nearest accessible goal from the given position"""
        nearest_goal = None
        min_distance = float('inf')
        
        for goal in self.goals:
            goal_pos = goal['position']
            distance = math.hypot(goal_pos[0] - position[0], 
                                goal_pos[1] - position[1])
            
            # Check if path to goal is blocked
            if not check_wall_collision(position, goal_pos, self.walls, WALL_SAFETY_MARGIN):
                if distance < min_distance:
                    min_distance = distance
                    nearest_goal = goal
        
        return nearest_goal

    def collect_balls(self):
        """Main ball collection loop"""
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
                    frame = self.draw_field(frame)  # Still show field even without robot
                else:
                    # Detect balls
                    balls = self.detect_balls()
                    current_time = time.time()
                    
                    # Only replan if:
                    # 1. We have balls AND robot markers are detected AND
                    # 2. Either:
                    #    a) It's been at least 3 seconds since last plan OR
                    #    b) Ball positions have changed significantly
                    should_replan = False
                    if balls:
                        if current_time - last_plan_time >= 3:
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
                        
                        # Sort balls by distance from current robot position
                        balls.sort(key=lambda b: math.hypot(
                            b[0] - self.robot_pos[0],
                            b[1] - self.robot_pos[1]
                        ))

                        # Take only the closest ball
                        current_batch = balls[:1]
                        path = []
                        
                        if current_batch:
                            closest_ball = current_batch[0]
                            ball_pos = (closest_ball[0], closest_ball[1])
                            approach_pos, target_angle = self.get_approach_vector(ball_pos)
                            
                            # Check for wall collisions
                            if not check_wall_collision(self.robot_pos, approach_pos, self.walls, WALL_SAFETY_MARGIN) and \
                               not check_wall_collision(approach_pos, ball_pos, self.walls, WALL_SAFETY_MARGIN):
                                
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
                                
                                # If this will be our third ball, add path to goal
                                if self.collected_balls >= 2:  # Already have 2, this will be the third
                                    goal_path = self.calculate_goal_approach_path(ball_pos, target_angle)
                                    path.extend(goal_path)

                        # Draw path on frame
                        frame_with_path = self.draw_path(frame, path)
                        
                        # Add collection counter and key command help
                        help_text = [
                            f"Balls Collected: {self.collected_balls}/3",
                            "Commands:",
                            "SPACE - Execute path",
                            "Q - Quit"
                        ]
                        y = 150
                        for text in help_text:
                            cv2.putText(frame_with_path, text,
                                      (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                                      0.5, (255, 255, 255), 1)
                            y += 20
                        
                        cv2.imshow("Path Planning", frame_with_path)
                        
                        # Handle key commands
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            break
                        elif key == ord(' '):
                            # Execute the planned path
                            logger.info("Executing path with {} points".format(len(path)))
                            
                            try:
                                for point in path:
                                    # Wait for visual marker detection
                                    retries = 0
                                    while not self.update_robot_position(frame) and retries < 10:
                                        ret, frame = self.cap.read()
                                        retries += 1
                                        time.sleep(0.1)
                                    
                                    if retries >= 10:
                                        raise Exception("Lost robot marker tracking")

                                    # Calculate turn angle from current heading
                                    target_pos = point['pos']
                                    dx = target_pos[0] - self.robot_pos[0]
                                    dy = target_pos[1] - self.robot_pos[1]
                                    target_angle = math.degrees(math.atan2(dy, dx))
                                    
                                    # Turn to face target
                                    angle_diff = (target_angle - self.robot_heading + 180) % 360 - 180
                                    if abs(angle_diff) > 5:
                                        logger.info(f"Turning {angle_diff:.1f} degrees")
                                        if not self.turn(angle_diff):
                                            raise Exception("Turn command failed")
                                    
                                    # Move to target position
                                    distance = math.hypot(dx, dy)
                                    
                                    if point['type'] == 'collect':
                                        logger.info(f"Collecting {point['ball_type']} ball")
                                        if not self.collect(COLLECTION_DISTANCE_CM):
                                            raise Exception("Collect command failed")
                                        self.collected_balls += 1  # Increment counter after successful collection
                                    elif point['type'] == 'goal':
                                        logger.info(f"Moving {distance:.1f} cm to goal")
                                        if not self.move(distance):
                                            raise Exception("Move command failed")
                                        # After reaching goal position, deliver balls
                                        if not self.deliver_balls():
                                            raise Exception("Ball delivery failed")
                                        self.collected_balls = 0  # Reset counter after delivery
                                    else:  # approach point
                                        logger.info(f"Moving {distance:.1f} cm")
                                        if not self.move(distance):
                                            raise Exception("Move command failed")
                                    
                                    # Update visualization
                                    ret, frame = self.cap.read()
                                    if ret:
                                        self.update_robot_position(frame)
                                        frame = self.draw_status(frame)
                                        frame = self.draw_robot_markers(frame)
                                        frame_with_path = self.draw_path(frame, path)
                                        cv2.imshow("Path Planning", frame_with_path)
                                        cv2.waitKey(1)
                                
                                # Pause briefly after completing the path
                                cv2.waitKey(1000)
                                
                            except Exception as e:
                                logger.error("Path execution failed: {}".format(e))
                                self.stop()
                                cv2.waitKey(2000)
                        
                    # Show frame even when not replanning
                    cv2.imshow("Path Planning", frame)
                    cv2.waitKey(1)
                    
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