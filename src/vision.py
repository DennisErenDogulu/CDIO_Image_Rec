#!/usr/bin/env python3
"""
Computer Vision Module

Handles camera operations, marker detection, ball detection, and homography calculations.
"""

import cv2
import math
import numpy as np
import logging
from typing import List, Tuple, Optional
from roboflow import Roboflow

from .config import (
    ROBOFLOW_API_KEY, RF_WORKSPACE, RF_PROJECT, RF_VERSION,
    GREEN_LOWER, GREEN_UPPER, PURPLE_RANGES,
    CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS,
    FIELD_WIDTH_CM, FIELD_HEIGHT_CM, IGNORED_AREA
)

logger = logging.getLogger(__name__)


class VisionSystem:
    """Handles all computer vision tasks including camera, detection, and calibration"""
    
    def __init__(self):
        # Initialize Roboflow
        self.rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        self.project = self.rf.workspace(RF_WORKSPACE).project(RF_PROJECT)
        self.model = self.project.version(RF_VERSION).model
        
        # Initialize camera
        self.cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera")

        # Set camera properties for reduced lag
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        
        # Calibration
        self.calibration_points = []
        self.homography_matrix = None
        self.obstacle_points = []
    
    def flush_camera_buffer(self):
        """Flush camera buffer to get the most recent frame"""
        for _ in range(3):
            self.cap.grab()

    def get_fresh_frame(self):
        """Get the most recent frame with minimal lag"""
        self.flush_camera_buffer()
        ret, frame = self.cap.read()
        return ret, frame

    def detect_markers(self, frame):
        """Detect green base and purple direction markers"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Detect green base marker
        green_mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
        kernel = np.ones((3,3), np.uint8)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
        
        green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        green_center = None
        if green_contours:
            valid_green_contours = []
            for contour in green_contours:
                area = cv2.contourArea(contour)
                if area > 100:
                    rect = cv2.minAreaRect(contour)
                    width, height = rect[1]
                    if width > 0 and height > 0:
                        aspect_ratio = max(width, height) / min(width, height)
                        if aspect_ratio < 3:
                            valid_green_contours.append(contour)
            
            if valid_green_contours:
                largest_green = max(valid_green_contours, key=cv2.contourArea)
                M = cv2.moments(largest_green)
                if M["m00"] != 0:
                    green_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        
        # Find purple direction marker
        purple_center = None
        if green_center:
            search_mask = np.zeros_like(hsv[:,:,0])
            cv2.circle(search_mask, green_center, 40, 255, -1)
            
            purple_mask = np.zeros_like(hsv[:,:,0])
            for purple_lower, purple_upper in PURPLE_RANGES:
                range_mask = cv2.inRange(hsv, purple_lower, purple_upper)
                purple_mask = cv2.bitwise_or(purple_mask, range_mask)
            
            purple_mask = cv2.bitwise_and(purple_mask, search_mask)
            
            kernel_small = np.ones((2,2), np.uint8)
            purple_mask = cv2.morphologyEx(purple_mask, cv2.MORPH_OPEN, kernel_small)
            purple_mask = cv2.morphologyEx(purple_mask, cv2.MORPH_CLOSE, kernel)
            purple_mask = cv2.GaussianBlur(purple_mask, (3, 3), 0)
            
            purple_contours, _ = cv2.findContours(purple_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if purple_contours:
                scored_contours = []
                for contour in purple_contours:
                    area = cv2.contourArea(contour)
                    if area > 30:
                        perimeter = cv2.arcLength(contour, True)
                        if perimeter > 0:
                            circularity = 4 * np.pi * area / (perimeter * perimeter)
                            hull = cv2.convexHull(contour)
                            hull_area = cv2.contourArea(hull)
                            compactness = area / hull_area if hull_area > 0 else 0
                            score = area * 0.1 + circularity * 100 + compactness * 50
                            if 30 < area < 500:
                                score += 20
                            scored_contours.append((contour, score))
                
                if scored_contours:
                    scored_contours.sort(key=lambda x: x[1], reverse=True)
                    best_contour = scored_contours[0][0]
                    
                    max_distance = 0
                    tip_point = None
                    for point in best_contour:
                        px, py = point[0]
                        distance = math.hypot(px - green_center[0], py - green_center[1])
                        if distance > max_distance:
                            max_distance = distance
                            tip_point = (px, py)
                    
                    if tip_point:
                        purple_center = tip_point
        
        return green_center, purple_center

    def detect_balls(self) -> List[Tuple[float, float, str]]:
        """Detect balls in current camera view"""
        ret, frame = self.get_fresh_frame()
        if not ret:
            return []

        small = cv2.resize(frame, (416, 416))
        predictions = self.model.predict(small, confidence=30, overlap=20).json()
        
        balls = []
        scale_x = frame.shape[1] / 416
        scale_y = frame.shape[0] / 416
        
        target_ball_types = ['white_ball', 'orange_ball']
        
        for pred in predictions.get('predictions', []):
            ball_class = pred['class']
            if ball_class not in target_ball_types:
                continue
                
            x_px = int(pred['x'] * scale_x)
            y_px = int(pred['y'] * scale_y)
            
            if self.homography_matrix is not None:
                pt_px = np.array([[[x_px, y_px]]], dtype="float32")
                pt_cm = cv2.perspectiveTransform(pt_px, np.linalg.inv(self.homography_matrix))[0][0]
                x_cm, y_cm = pt_cm
                
                if not (IGNORED_AREA["x_min"] <= x_cm <= IGNORED_AREA["x_max"] and
                       IGNORED_AREA["y_min"] <= y_cm <= IGNORED_AREA["y_max"]):
                    balls.append((x_cm, y_cm, ball_class))
        
        return balls

    def calibrate(self):
        """Perform camera calibration"""
        calibration_phase = "corners"
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal calibration_phase
            
            if event == cv2.EVENT_LBUTTONDOWN:
                if calibration_phase == "corners" and len(self.calibration_points) < 4:
                    self.calibration_points.append((x, y))
                    logger.info(f"Corner {len(self.calibration_points)} set: ({x}, {y})")

                    if len(self.calibration_points) == 4:
                        dst_pts = np.array([
                            [0, 0],
                            [FIELD_WIDTH_CM, 0],
                            [FIELD_WIDTH_CM, FIELD_HEIGHT_CM],
                            [0, FIELD_HEIGHT_CM]
                        ], dtype="float32")
                        src_pts = np.array(self.calibration_points, dtype="float32")
                        self.homography_matrix = cv2.getPerspectiveTransform(dst_pts, src_pts)
                        calibration_phase = "obstacle"
                        logger.info("✅ Field corners calibrated. Now mark obstacle area:")
                
                elif calibration_phase == "obstacle" and len(self.obstacle_points) < 2:
                    self.obstacle_points.append((x, y))
                    if len(self.obstacle_points) == 2:
                        self.setup_obstacle_area()

        cv2.namedWindow("Calibration")
        cv2.setMouseCallback("Calibration", mouse_callback)
        
        while calibration_phase == "corners" or len(self.obstacle_points) < 2:
            ret, frame = self.get_fresh_frame()
            if not ret:
                continue
            
            if calibration_phase == "corners":
                instruction = f"Click field corners: 1=top-left, 2=top-right, 3=bottom-right, 4=bottom-left ({len(self.calibration_points)}/4)"
                cv2.putText(frame, instruction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                for i, pt in enumerate(self.calibration_points):
                    cv2.circle(frame, pt, 8, (0, 255, 0), -1)
                    cv2.putText(frame, str(i+1), (pt[0]+15, pt[1]+15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            elif calibration_phase == "obstacle":
                instruction = f"Mark obstacle area: click top-left, then bottom-right ({len(self.obstacle_points)}/2)"
                cv2.putText(frame, instruction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                for i, pt in enumerate(self.calibration_points):
                    cv2.circle(frame, pt, 6, (0, 255, 0), -1)
                
                for i, pt in enumerate(self.obstacle_points):
                    cv2.circle(frame, pt, 8, (0, 0, 255), -1)
                    cv2.putText(frame, f"OBS{i+1}", (pt[0]+15, pt[1]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            cv2.imshow("Calibration", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and calibration_phase == "obstacle" and len(self.obstacle_points) == 0:
                logger.info("Skipping obstacle marking")
                self.obstacle_points = [(0, 0), (1, 1)]
                self.setup_obstacle_area()
        
        cv2.destroyWindow("Calibration")
    
    def setup_obstacle_area(self):
        """Convert obstacle pixel coordinates to cm"""
        if len(self.obstacle_points) == 2 and self.homography_matrix is not None:
            pt1_px = np.array([[[float(self.obstacle_points[0][0]), float(self.obstacle_points[0][1])]]], dtype="float32")
            pt2_px = np.array([[[float(self.obstacle_points[1][0]), float(self.obstacle_points[1][1])]]], dtype="float32")
            
            pt1_cm = cv2.perspectiveTransform(pt1_px, np.linalg.inv(self.homography_matrix))[0][0]
            pt2_cm = cv2.perspectiveTransform(pt2_px, np.linalg.inv(self.homography_matrix))[0][0]
            
            x_min = min(pt1_cm[0], pt2_cm[0])
            x_max = max(pt1_cm[0], pt2_cm[0])
            y_min = min(pt1_cm[1], pt2_cm[1])
            y_max = max(pt1_cm[1], pt2_cm[1])
            
            IGNORED_AREA["x_min"] = x_min
            IGNORED_AREA["x_max"] = x_max
            IGNORED_AREA["y_min"] = y_min
            IGNORED_AREA["y_max"] = y_max
            
            logger.info(f"✅ Obstacle area set: ({x_min:.1f}, {y_min:.1f}) to ({x_max:.1f}, {y_max:.1f}) cm")

    def draw_robot_markers(self, frame):
        """Draw detected robot markers on frame"""
        green_center, purple_center = self.detect_markers(frame)
        
        if green_center:
            cv2.circle(frame, green_center, 10, (0, 255, 0), -1)
            cv2.circle(frame, green_center, 12, (255, 255, 255), 2)
            cv2.circle(frame, green_center, 40, (0, 255, 255), 1)
        
        if purple_center:
            cv2.circle(frame, purple_center, 8, (128, 0, 128), -1)
            cv2.circle(frame, purple_center, 10, (255, 255, 255), 2)
        
        if green_center and purple_center:
            cv2.line(frame, green_center, purple_center, (255, 255, 255), 2)
        
        return frame

    def show_detection_debug(self, frame):
        """Show debug view of marker detection"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        green_center, _ = self.detect_markers(frame)
        
        purple_mask_combined = np.zeros_like(hsv[:,:,0])
        for purple_lower, purple_upper in PURPLE_RANGES:
            range_mask = cv2.inRange(hsv, purple_lower, purple_upper)
            purple_mask_combined = cv2.bitwise_or(purple_mask_combined, range_mask)
        
        if green_center:
            search_mask = np.zeros_like(hsv[:,:,0])
            cv2.circle(search_mask, green_center, 40, 255, -1)
            purple_mask_combined = cv2.bitwise_and(purple_mask_combined, search_mask)
        
        kernel_small = np.ones((2,2), np.uint8)
        kernel = np.ones((3,3), np.uint8)
        purple_mask_filtered = cv2.morphologyEx(purple_mask_combined, cv2.MORPH_OPEN, kernel_small)
        purple_mask_filtered = cv2.morphologyEx(purple_mask_filtered, cv2.MORPH_CLOSE, kernel)
        purple_mask_filtered = cv2.GaussianBlur(purple_mask_filtered, (3, 3), 0)
        
        debug_frame = frame.copy()
        purple_colored = cv2.applyColorMap(purple_mask_filtered, cv2.COLORMAP_PLASMA)
        debug_frame = cv2.addWeighted(debug_frame, 0.7, purple_colored, 0.3, 0)
        
        return debug_frame

    def release(self):
        """Release camera resources"""
        if self.cap:
            self.cap.release() 