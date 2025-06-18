#!/usr/bin/env python3
"""
Continuous Ball Collection Client

This script implements a continuous ball collection strategy:
1. Detect balls in the camera view
2. For each group of up to 3 closest balls:
   a. Calculate optimal approach angles
   b. Move to each ball, collect it
   c. Return to goal when 3 balls collected
3. Repeat until no balls remain
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
        
        # Goal dimensions and positions
        self.goal_y_center = FIELD_HEIGHT_CM / 2  # Center Y coordinate of goal
        self.goal_approach_distance = 20  # Distance to stop in front of goal
        self.delivery_time = 2.0  # Seconds to run collector in reverse

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
            # First draw the starting box
            frame = self.draw_starting_box(frame)
            
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

    def detect_balls(self) -> List[Tuple[float, float, str]]:
        """Detect balls in current camera view, return list of (x_cm, y_cm, label)"""
        ret, frame = self.cap.read()
        if not ret:
            return []

        # Resize for Roboflow model
        small = cv2.resize(frame, (416, 416))
        
        # Get predictions
        predictions = self.model.predict(small, confidence=30, overlap=20).json()
        
        balls = []
        scale_x = frame.shape[1] / 416
        scale_y = frame.shape[0] / 416
        
        for pred in predictions.get('predictions', []):
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
                    balls.append((x_cm, y_cm, pred['class']))
        
        return balls

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
        if not self.homography_matrix is None:
            overlay = frame.copy()
            
            # Colors for different point types
            colors = {
                'approach': (0, 255, 255),  # Yellow
                'collect': (0, 255, 0),     # Green
                'goal': (0, 0, 255)         # Red
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
        
        while True:
            try:
                # Capture frame for visualization
                ret, frame = self.cap.read()
                if not ret:
                    continue

                # Draw walls and safety margins
                frame = self.draw_walls(frame)

                # Add status overlay
                frame = self.draw_status(frame)

                # Detect balls
                balls = self.detect_balls()
                current_time = time.time()
                
                # Only replan if:
                # 1. We have balls AND
                # 2. Either:
                #    a) It's been at least 3 seconds since last plan OR
                #    b) Ball positions have changed significantly
                should_replan = False
                if balls:
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
                    logger.info("Planning to collect {} balls".format(len(balls)))

                    # Sort balls by distance from robot
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
                        "SPACE - Execute path",
                        "Q - Quit",
                        "R - Reset robot position"
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
                    elif key == ord('r'):
                        self.robot_pos = (ROBOT_START_X, ROBOT_START_Y)
                        self.robot_heading = ROBOT_START_HEADING
                        logger.info("Reset robot position to start")
                    elif key == ord(' '):
                        # Execute the planned path
                        logger.info("Executing path with {} points".format(len(path)))
                        
                        try:
                            for point in path:
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
                                    self.robot_heading = target_angle
                                
                                # Move to target position
                                distance = math.hypot(dx, dy)
                                
                                if point['type'] == 'collect':
                                    logger.info(f"Collecting {point['ball_type']} ball")
                                    if not self.collect(COLLECTION_DISTANCE_CM):
                                        raise Exception("Collect command failed")
                                elif point['type'] == 'goal':
                                    logger.info(f"Moving {distance:.1f} cm to goal")
                                    if not self.move(distance):
                                        raise Exception("Move command failed")
                                    # After reaching goal position, deliver balls
                                    if not self.deliver_balls():
                                        raise Exception("Ball delivery failed")
                                else:  # approach point
                                    logger.info(f"Moving {distance:.1f} cm")
                                    if not self.move(distance):
                                        raise Exception("Move command failed")
                                
                                # Update robot position
                                self.robot_pos = target_pos
                                
                                # Update visualization
                                ret, frame = self.cap.read()
                                if ret:
                                    frame = self.draw_status(frame)
                                    frame_with_path = self.draw_path(frame, path)
                                    cv2.imshow("Path Planning", frame_with_path)
                                    cv2.waitKey(1)
                            
                            # Pause briefly after completing the path
                            cv2.waitKey(1000)
                            
                        except Exception as e:
                            logger.error("Path execution failed: {}".format(e))
                            self.stop()
                            cv2.waitKey(2000)
                    
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