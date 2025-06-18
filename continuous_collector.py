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
import threading
from queue import Queue, Empty
from dataclasses import dataclass

# Configuration
EV3_IP = "172.20.10.6"
EV3_PORT = 12345
ROBOFLOW_API_KEY = "qJTLU5ku2vpBGQUwjBx2"
RF_WORKSPACE = "cdio-nczdp"
RF_PROJECT = "cdio-golfbot2025" 
RF_VERSION = 13

# Color detection ranges (HSV)
GREEN_LOWER = np.array([35, 50, 50])
GREEN_UPPER = np.array([85, 255, 255])
PINK_LOWER = np.array([145, 50, 50])
PINK_UPPER = np.array([175, 255, 255])

# Marker dimensions
GREEN_MARKER_WIDTH_CM = 20  # Width of green base sheet
PINK_MARKER_WIDTH_CM = 5    # Width of pink direction marker

# Wall configuration
WALL_SAFETY_MARGIN = 1  # cm, minimum distance to keep from walls
GOAL_WIDTH_SMALL = 8  # cm (80mm)
GOAL_WIDTH_LARGE = 20  # cm (200mm)
GOAL_WIDTH = GOAL_WIDTH_SMALL  # Default to small goal

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

@dataclass
class FrameData:
    frame: np.ndarray
    timestamp: float
    robot_pos: Optional[Tuple[float, float]] = None
    robot_heading: Optional[float] = None

class CameraThread(threading.Thread):
    def __init__(self, camera_index=1):
        super().__init__()
        self.daemon = True
        self.frame_queue = Queue(maxsize=1)  # Only keep latest frame
        self.stop_flag = threading.Event()
        self.camera_index = camera_index

    def run(self):
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            raise RuntimeError("Could not open camera")
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Back to higher resolution
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        try:
            while not self.stop_flag.is_set():
                ret, frame = cap.read()
                if ret:
                    # Clear queue and put new frame
                    while self.frame_queue.qsize() > 0:
                        try:
                            self.frame_queue.get_nowait()
                        except Empty:
                            break
                    
                    frame_data = FrameData(
                        frame=frame,
                        timestamp=time.time()
                    )
                    self.frame_queue.put(frame_data)
                time.sleep(0.01)  # Small sleep to prevent busy waiting
        finally:
            cap.release()

    def get_frame(self) -> Optional[FrameData]:
        try:
            return self.frame_queue.get_nowait()
        except Empty:
            return None

    def stop(self):
        self.stop_flag.set()

class ProcessingThread(threading.Thread):
    def __init__(self, ball_collector):
        super().__init__()
        self.daemon = True
        self.ball_collector = ball_collector
        self.processed_frame_queue = Queue(maxsize=1)
        self.stop_flag = threading.Event()

    def run(self):
        while not self.stop_flag.is_set():
            frame_data = self.ball_collector.camera_thread.get_frame()
            if frame_data is not None:
                # Update robot position
                if self.ball_collector.homography_matrix is not None:
                    green_center, _, pink_endpoints = self.ball_collector.detect_markers(frame_data.frame)
                    if green_center and pink_endpoints:
                        # Convert green center to cm coordinates
                        green_px = np.array([[[float(green_center[0]), float(green_center[1])]]], dtype="float32")
                        green_cm = cv2.perspectiveTransform(green_px, np.linalg.inv(self.ball_collector.homography_matrix))[0][0]
                        
                        # Find front point
                        front_point = max(pink_endpoints, 
                                        key=lambda p: math.hypot(p[0] - green_center[0],
                                                               p[1] - green_center[1]))
                        
                        # Convert front point to cm coordinates
                        front_px = np.array([[[float(front_point[0]), float(front_point[1])]]], dtype="float32")
                        front_cm = cv2.perspectiveTransform(front_px, np.linalg.inv(self.ball_collector.homography_matrix))[0][0]
                        
                        # Calculate heading
                        dx = front_cm[0] - green_cm[0]
                        dy = front_cm[1] - green_cm[1]
                        heading = math.degrees(math.atan2(dy, dx))
                        
                        # Update frame data with position info
                        frame_data.robot_pos = (float(green_cm[0]), float(green_cm[1]))
                        frame_data.robot_heading = heading
                
                # Put processed frame in queue
                while self.processed_frame_queue.qsize() > 0:
                    try:
                        self.processed_frame_queue.get_nowait()
                    except Empty:
                        break
                self.processed_frame_queue.put(frame_data)
            time.sleep(0.01)

    def get_processed_frame(self) -> Optional[FrameData]:
        try:
            return self.processed_frame_queue.get_nowait()
        except Empty:
            return None

    def stop(self):
        self.stop_flag.set()

class BallCollector:
    def __init__(self):
        # Initialize camera thread
        self.camera_thread = CameraThread()
        self.camera_thread.start()
        
        # Initialize processing thread
        self.processing_thread = ProcessingThread(self)
        self.processing_thread.start()
        
        # Initialize Roboflow
        self.rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        self.project = self.rf.workspace(RF_WORKSPACE).project(RF_PROJECT)
        self.model = self.project.version(RF_VERSION).model
        
        # Robot state
        self.robot_pos = (ROBOT_START_X, ROBOT_START_Y)
        self.robot_heading = ROBOT_START_HEADING
        
        # Calibration points for homography
        self.calibration_points = []
        self.homography_matrix = None
        
        # Wall configuration
        self.walls = []
        
        # Goal dimensions and positions
        self.goal_y_center = FIELD_HEIGHT_CM / 2
        self.goal_approach_distance = 20
        self.delivery_time = 10.0
        self.current_goal_width = GOAL_WIDTH_SMALL
        self.is_small_goal = True

        # Color ranges for marker detection
        self.green_lower = GREEN_LOWER
        self.green_upper = GREEN_UPPER
        self.pink_lower = PINK_LOWER
        self.pink_upper = PINK_UPPER

        # Debug window for marker detection
        self.debug_window = True
        cv2.namedWindow("Marker Debug")

    def detect_markers(self, frame):
        """Detect green base and pink direction markers"""
        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Detect green base marker
        green_mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
        kernel = np.ones((5,5), np.uint8)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
        
        green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Detect pink direction marker
        pink_mask = cv2.inRange(hsv, self.pink_lower, self.pink_upper)
        pink_mask = cv2.morphologyEx(pink_mask, cv2.MORPH_OPEN, kernel)
        pink_mask = cv2.morphologyEx(pink_mask, cv2.MORPH_CLOSE, kernel)
        
        pink_contours, _ = cv2.findContours(pink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find green marker center and orientation
        green_center = None
        green_rect = None
        if green_contours:
            largest_green = max(green_contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_green)
            if area > 1000:  # Minimum area threshold
                green_rect = cv2.minAreaRect(largest_green)
                green_center = (int(green_rect[0][0]), int(green_rect[0][1]))
        
        # Find pink marker endpoints
        pink_endpoints = None
        if pink_contours:
            largest_pink = max(pink_contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_pink)
            if area > 100:  # Minimum area threshold
                pink_points = largest_pink.squeeze()
                if len(pink_points.shape) >= 2:
                    # Find the two points furthest apart
                    max_dist = 0
                    for i in range(len(pink_points)):
                        for j in range(i + 1, len(pink_points)):
                            dist = np.linalg.norm(pink_points[i] - pink_points[j])
                            if dist > max_dist:
                                max_dist = dist
                                pink_endpoints = (
                                    tuple(pink_points[i]),
                                    tuple(pink_points[j])
                                )

        return green_center, green_rect, pink_endpoints

    def update_robot_position(self):
        """Update robot position from latest processed frame"""
        frame_data = self.processing_thread.get_processed_frame()
        if frame_data is not None:
            if frame_data.robot_pos is not None and frame_data.robot_heading is not None:
                self.robot_pos = frame_data.robot_pos
                self.robot_heading = frame_data.robot_heading
                return True
        return False

    def calibrate(self):
        """Perform camera calibration by clicking 4 corners"""
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(self.calibration_points) < 4:
                    self.calibration_points.append((x, y))
                    logger.info(f"Corner {len(self.calibration_points)} set: ({x}, {y})")

                if len(self.calibration_points) == 4:
                    # Build homography matrix
                    dst_pts = np.array([
                        [0, 0],
                        [FIELD_WIDTH_CM, 0],
                        [FIELD_WIDTH_CM, FIELD_HEIGHT_CM],
                        [0, FIELD_HEIGHT_CM]
                    ], dtype="float32")
                    src_pts = np.array(self.calibration_points, dtype="float32")
                    self.homography_matrix = cv2.getPerspectiveTransform(dst_pts, src_pts)
                    
                    # Set up walls
                    self.setup_walls()
                    logger.info("✅ Calibration and wall setup complete")

        cv2.namedWindow("Calibration")
        cv2.setMouseCallback("Calibration", mouse_callback)
        
        while len(self.calibration_points) < 4:
            frame_data = self.camera_thread.get_frame()
            if frame_data is None:
                continue
                
            frame = frame_data.frame
            # Draw existing points
            for i, pt in enumerate(self.calibration_points):
                cv2.circle(frame, pt, 5, (0, 255, 0), -1)
                cv2.putText(frame, str(i+1), (pt[0]+10, pt[1]+10),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("Calibration", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyWindow("Calibration")

    def setup_walls(self):
        """Set up wall segments based on calibration points, excluding goal areas"""
        if len(self.calibration_points) != 4:
            return

        # Calculate goal positions (in cm)
        goal_y_min = (FIELD_HEIGHT_CM / 2) - (self.current_goal_width / 2)
        goal_y_max = (FIELD_HEIGHT_CM / 2) + (self.current_goal_width / 2)

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

    def detect_balls(self) -> List[Tuple[float, float, str]]:
        """Detect balls in current camera view, return list of (x_cm, y_cm, label)"""
        frame_data = self.processing_thread.get_processed_frame()
        if frame_data is None:
            return []

        frame = frame_data.frame
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

    def get_approach_vector(self, target_pos: Tuple[float, float], try_reverse: bool = True) -> Tuple[Tuple[float, float], float, bool]:
        """Calculate approach position and angle for a target position, with optional reverse movement"""
        # Get vector from robot to target
        dx = target_pos[0] - self.robot_pos[0]
        dy = target_pos[1] - self.robot_pos[1]
        distance = math.hypot(dx, dy)
        
        # Try forward approach first
        forward_angle = math.degrees(math.atan2(dy, dx))
        forward_diff = (forward_angle - self.robot_heading + 180) % 360 - 180
        
        # If reverse is allowed, also try reverse approach
        if try_reverse:
            reverse_angle = (forward_angle + 180) % 360
            reverse_diff = (reverse_angle - self.robot_heading + 180) % 360 - 180
            
            # Use reverse if it requires less turning
            if abs(reverse_diff) < abs(forward_diff):
                ratio = (distance + APPROACH_DISTANCE_CM) / distance if distance > 0 else 1
                approach_x = self.robot_pos[0] + dx * ratio
                approach_y = self.robot_pos[1] + dy * ratio
                return (approach_x, approach_y), reverse_angle, True
        
        # Default to forward approach
        ratio = (distance - APPROACH_DISTANCE_CM) / distance if distance > APPROACH_DISTANCE_CM else 0
        approach_x = self.robot_pos[0] + dx * ratio
        approach_y = self.robot_pos[1] + dy * ratio
        return (approach_x, approach_y), forward_angle, False

    def draw_path(self, frame, path, preview_mode=False):
        """Draw the planned path on the frame"""
        if not self.homography_matrix is None:
            overlay = frame.copy()
            
            # Colors for different point types
            colors = {
                'approach': (0, 255, 255),  # Yellow
                'collect': (0, 255, 0),     # Green
                'goal': (0, 0, 255),        # Red
                'reverse': (255, 0, 255)    # Purple for reverse movements
            }
            
            # Draw lines between points
            for i in range(len(path) - 1):
                # Convert cm to pixels for both points
                start_cm = np.array([[[path[i]['pos'][0], path[i]['pos'][1]]]], dtype="float32")
                end_cm = np.array([[[path[i+1]['pos'][0], path[i+1]['pos'][1]]]], dtype="float32")
                
                start_px = cv2.perspectiveTransform(start_cm, self.homography_matrix)[0][0].astype(int)
                end_px = cv2.perspectiveTransform(end_cm, self.homography_matrix)[0][0].astype(int)
                
                # Draw line (dashed if preview mode)
                if preview_mode:
                    # Draw dashed line
                    dash_length = 10
                    dx = end_px[0] - start_px[0]
                    dy = end_px[1] - start_px[1]
                    dist = math.hypot(dx, dy)
                    if dist > 0:
                        num_dashes = int(dist / (2 * dash_length))
                        for j in range(num_dashes):
                            x1 = int(start_px[0] + dx * (2*j) / (2*num_dashes))
                            y1 = int(start_px[1] + dy * (2*j) / (2*num_dashes))
                            x2 = int(start_px[0] + dx * (2*j+1) / (2*num_dashes))
                            y2 = int(start_px[1] + dy * (2*j+1) / (2*num_dashes))
                            cv2.line(overlay, (x1, y1), (x2, y2), (150, 150, 150), 2)
                else:
                    cv2.line(overlay, tuple(start_px), tuple(end_px), (150, 150, 150), 2)
            
            # Draw points
            for point in path:
                pt_cm = np.array([[[point['pos'][0], point['pos'][1]]]], dtype="float32")
                pt_px = cv2.perspectiveTransform(pt_cm, self.homography_matrix)[0][0].astype(int)
                
                # Draw point
                point_type = 'reverse' if point.get('is_reverse', False) else point['type']
                color = colors[point_type]
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
                elif point.get('is_reverse', False):
                    cv2.putText(overlay, "R", 
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

            if preview_mode:
                # Add preview mode text
                cv2.putText(frame, "PREVIEW MODE - Press ENTER to execute, ESC to cancel", 
                          (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.7, (0, 255, 255), 2)
        
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
            cv2.rectangle(frame, (5, 5), (200, 130), (0, 0, 0), -1)
            cv2.rectangle(frame, (5, 5), (200, 130), (255, 255, 255), 1)
            
            # Add FPS counter
            cv2.putText(frame, f"FPS: {self.fps}",
                       (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Battery status
            battery_pct = status.get("battery_percentage", 0)
            battery_color = (0, 255, 0) if battery_pct > 20 else (0, 0, 255)
            cv2.putText(frame, f"Battery: {battery_pct:.1f}%",
                       (x, y + line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, battery_color, 1)
            
            # Motor speeds
            left_speed = status.get("left_motor", {}).get("speed", 0)
            right_speed = status.get("right_motor", {}).get("speed", 0)
            collector_speed = status.get("collector_motor", {}).get("speed", 0)
            
            cv2.putText(frame, f"Left Motor: {left_speed}",
                       (x, y + 2*line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Right Motor: {right_speed}",
                       (x, y + 3*line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Collector: {collector_speed}",
                       (x, y + 4*line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add frame processing time
            frame_time = time.time() - self.last_frame_time
            cv2.putText(frame, f"Frame time: {frame_time*1000:.1f}ms",
                       (x, y + 5*line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
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
        if abs(angle_diff) < 15 and not self.check_wall_collision(current_pos, (goal_x, goal_y), self.walls, WALL_SAFETY_MARGIN):
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
        if not self.check_wall_collision(current_pos, (swing_x, swing_y), self.walls, WALL_SAFETY_MARGIN):
            path.append({
                'type': 'approach',
                'pos': (swing_x, swing_y),
                'angle': swing_angle
            })
        
        if not self.check_wall_collision((swing_x, swing_y), (approach_x, approach_y), self.walls, WALL_SAFETY_MARGIN):
            path.append({
                'type': 'approach',
                'pos': (approach_x, approach_y),
                'angle': 0  # Align with goal
            })
        
        if not self.check_wall_collision((approach_x, approach_y), (goal_x, goal_y), self.walls, WALL_SAFETY_MARGIN):
            path.append({
                'type': 'goal',
                'pos': (goal_x, goal_y),
                'angle': 0
            })
        
        return path

    def collect_balls(self):
        """Main ball collection loop"""
        last_plan_time = 0
        last_plan_positions = []
        
        while True:
            try:
                # Get latest processed frame
                frame_data = self.processing_thread.get_processed_frame()
                if frame_data is None:
                    continue

                # Update robot position
                if not self.update_robot_position():
                    logger.warning("Could not detect robot markers")
                    continue

                # Detect balls
                balls = self.detect_balls()
                current_time = time.time()
                
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
                        approach_pos, target_angle, is_reverse = self.get_approach_vector(ball_pos)
                        
                        # Check for wall collisions
                        if self.check_wall_collision(current_pos, approach_pos, self.walls, WALL_SAFETY_MARGIN):
                            logger.warning(f"Path to ball at {ball_pos} blocked by wall, skipping")
                            remaining_balls.remove(closest_ball)
                            continue
                        
                        if self.check_wall_collision(approach_pos, ball_pos, self.walls, WALL_SAFETY_MARGIN):
                            logger.warning(f"Approach to ball at {ball_pos} blocked by wall, skipping")
                            remaining_balls.remove(closest_ball)
                            continue
                        
                        path.append({
                            'type': 'approach',
                            'pos': approach_pos,
                            'angle': target_angle,
                            'is_reverse': is_reverse
                        })
                        path.append({
                            'type': 'collect',
                            'pos': ball_pos,
                            'ball_type': closest_ball[2],
                            'is_reverse': is_reverse
                        })
                        
                        current_pos = ball_pos
                        current_heading = target_angle
                        remaining_balls.remove(closest_ball)

                    # Add goal approach path if we have collected any balls
                    if path:
                        goal_path = self.calculate_goal_approach_path(current_pos, current_heading)
                        path.extend(goal_path)
                        
                        # Execute the path
                        try:
                            self._execute_path(path, frame_data.frame)
                        except Exception as e:
                            logger.error(f"Path execution failed: {e}")
                            self.stop()
                            time.sleep(2)

                time.sleep(0.1)  # Small sleep to prevent CPU overuse
                    
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                time.sleep(0.1)  # Delay on error to prevent rapid error loops

    def _execute_path(self, path, frame):
        """Execute a planned path with continuous position updates"""
        for point in path:
            # Initial position update
            if not self.update_robot_position():
                raise Exception("Lost robot marker tracking before movement")

            # Calculate initial movement parameters
            target_pos = point['pos']
            dx = target_pos[0] - self.robot_pos[0]
            dy = target_pos[1] - self.robot_pos[1]
            distance = math.hypot(dx, dy)
            target_angle = math.degrees(math.atan2(dy, dx))
            
            # Adjust angle for reverse movement
            if point.get('is_reverse', False):
                target_angle = (target_angle + 180) % 360
            
            # Turn to face target with continuous position updates
            angle_diff = (target_angle - self.robot_heading + 180) % 360 - 180
            if abs(angle_diff) > 5:
                logger.info(f"Turning {angle_diff:.1f} degrees")
                if not self.turn(angle_diff):
                    raise Exception("Turn command failed")
                
                # Multiple position updates after turn
                for _ in range(3):
                    if not self.update_robot_position():
                        raise Exception("Lost robot marker tracking during turn")
                    # Recalculate angle difference after position update
                    new_angle_diff = (target_angle - self.robot_heading + 180) % 360 - 180
                    if abs(new_angle_diff) > 5:  # If still off by more than 5 degrees
                        logger.info(f"Correcting turn by {new_angle_diff:.1f} degrees")
                        if not self.turn(new_angle_diff):
                            raise Exception("Turn correction failed")
                    time.sleep(0.1)  # Small delay between updates
            
            # Handle different point types with continuous position monitoring
            if point['type'] == 'collect':
                logger.info(f"Collecting {point['ball_type']} ball")
                # Start collection movement
                if not self.collect(COLLECTION_DISTANCE_CM):
                    raise Exception("Collect command failed")
                
                # Monitor position during collection
                collection_start_time = time.time()
                while time.time() - collection_start_time < 5:  # Max 5 seconds for collection
                    if not self.update_robot_position():
                        logger.warning("Position tracking lost during collection")
                    time.sleep(0.1)
            
            elif point['type'] == 'goal':
                logger.info(f"Moving {distance:.1f} cm to goal")
                # Start movement
                if not self.move(distance):
                    raise Exception("Move command failed")
                
                # Monitor position during movement to goal
                movement_start_time = time.time()
                last_correction_time = movement_start_time
                while time.time() - movement_start_time < 10:  # Max 10 seconds for movement
                    if not self.update_robot_position():
                        logger.warning("Position tracking lost during movement")
                        continue
                    
                    # Check if we need course correction (every 0.5 seconds)
                    current_time = time.time()
                    if current_time - last_correction_time >= 0.5:
                        # Recalculate distance and angle to goal
                        dx = target_pos[0] - self.robot_pos[0]
                        dy = target_pos[1] - self.robot_pos[1]
                        current_distance = math.hypot(dx, dy)
                        
                        # If we're close enough to goal, break
                        if current_distance < 5:  # Within 5cm of goal
                            break
                        
                        last_correction_time = current_time
                    
                    time.sleep(0.1)
                
                # Deliver balls at goal
                logger.info("Delivering balls")
                if not self.deliver_balls():
                    raise Exception("Ball delivery failed")
            
            else:  # approach points
                logger.info(f"Moving {distance:.1f} cm {'backwards' if point.get('is_reverse', False) else 'forwards'}")
                move_distance = -distance if point.get('is_reverse', False) else distance
                
                # Start movement
                if not self.move(move_distance):
                    raise Exception("Move command failed")
                
                # Monitor position during approach movement
                movement_start_time = time.time()
                last_correction_time = movement_start_time
                while time.time() - movement_start_time < 10:  # Max 10 seconds for movement
                    if not self.update_robot_position():
                        logger.warning("Position tracking lost during movement")
                        continue
                    
                    # Check if we need course correction (every 0.5 seconds)
                    current_time = time.time()
                    if current_time - last_correction_time >= 0.5:
                        # Recalculate distance to target
                        dx = target_pos[0] - self.robot_pos[0]
                        dy = target_pos[1] - self.robot_pos[1]
                        current_distance = math.hypot(dx, dy)
                        
                        # If we're close enough to target, break
                        if current_distance < 5:  # Within 5cm of target
                            break
                        
                        last_correction_time = current_time
                    
                    time.sleep(0.1)

    def run(self):
        """Main run loop"""
        try:
            # First calibrate camera perspective
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
            self.camera_thread.stop()
            self.processing_thread.stop()
            self.camera_thread.join()
            self.processing_thread.join()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    collector = BallCollector()
    collector.run() 