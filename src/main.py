#!/usr/bin/env python3
"""
Continuous Ball Collection Client - Main Application

This script implements a continuous ball collection strategy using visual markers.
Refactored into modular components for better maintainability.
"""

import cv2
import math
import time
import logging
import numpy as np
from typing import List, Tuple, Optional

from .config import (
    ROBOT_START_X, ROBOT_START_Y, ROBOT_START_HEADING,
    FRAME_SKIP_INTERVAL, MAX_BALLS_PER_TRIP, DELIVERY_TIME,
    WALL_SAFETY_MARGIN, FIELD_WIDTH_CM, FIELD_HEIGHT_CM, IGNORED_AREA
)
from .vision import VisionSystem
from .navigation import NavigationSystem
from .robot_comms import RobotComms
from .utils import point_to_line_distance, calculate_distance, calculate_angle

logger = logging.getLogger(__name__)


class BallCollector:
    """Main application class that orchestrates ball collection"""
    
    def __init__(self):
        # Initialize subsystems
        self.vision = VisionSystem()
        self.navigation = NavigationSystem()
        self.robot = RobotComms()
        
        # Robot state
        self.robot_pos = (ROBOT_START_X, ROBOT_START_Y)
        self.robot_heading = ROBOT_START_HEADING
        self.robot_front_pos = (ROBOT_START_X, ROBOT_START_Y)
        
        # Frame skipping for performance
        self.frame_skip_counter = 0
        self.frame_skip_interval = FRAME_SKIP_INTERVAL
        
        # Automatic operation control
        self.automatic_mode = False

    def update_robot_position(self, frame=None):
        """Update robot position and heading based on visual markers"""
        if frame is None:
            ret, frame = self.vision.get_fresh_frame()
            if not ret:
                return False
        
        green_center, purple_center = self.vision.detect_markers(frame)
        
        if green_center and purple_center and self.vision.homography_matrix is not None:
            # Convert markers to cm coordinates
            green_px = np.array([[[float(green_center[0]), float(green_center[1])]]], dtype="float32")
            green_cm = cv2.perspectiveTransform(green_px, np.linalg.inv(self.vision.homography_matrix))[0][0]
            
            purple_px = np.array([[[float(purple_center[0]), float(purple_center[1])]]], dtype="float32")
            purple_cm = cv2.perspectiveTransform(purple_px, np.linalg.inv(self.vision.homography_matrix))[0][0]
            
            # Update robot positions
            self.robot_pos = (green_cm[0], green_cm[1])
            self.robot_front_pos = (purple_cm[0], purple_cm[1])
            
            # Calculate heading
            dx = purple_cm[0] - green_cm[0]
            dy = purple_cm[1] - green_cm[1]
            self.robot_heading = math.degrees(math.atan2(dy, dx))
            
            return True
        
        return False

    def check_wall_proximity(self) -> bool:
        """Check if robot front (collector) is too close to any wall and back up if needed"""
        if not self.navigation.walls or not self.robot_front_pos:
            return True
        
        wall_danger_distance = 5  # cm
        
        for wall in self.navigation.walls:
            wall_start = (wall[0], wall[1])
            wall_end = (wall[2], wall[3])
            distance = point_to_line_distance(self.robot_front_pos, wall_start, wall_end)
            
            if distance < wall_danger_distance:
                logger.warning(f"Robot front (collector) very close to wall ({distance:.1f}cm)! Small backup for safety")
                
                # Back up 20cm away from the wall
                if not self.robot.move(-20):
                    logger.warning("Failed to back up from wall - continuing anyway")
                    return True
                
                # Update position after backing up
                if not self.update_robot_position():
                    logger.warning("Lost robot tracking after wall backup")
                    return True
                
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
        distance_to_target = calculate_distance(self.robot_pos, target_pos)
        target_angle = calculate_angle(self.robot_pos, target_pos)
        
        # Check if we're already close enough
        distance_tolerance = 15 if target_type == "goal" else 5
        if distance_to_target <= distance_tolerance:
            logger.info(f"Already at target {target_pos}")
            return True
        
        # Calculate angle differences for forward and backward movement
        forward_angle_diff = (target_angle - self.robot_heading + 180) % 360 - 180
        backward_angle_diff = (target_angle - self.robot_heading) % 360 - 180
        
        # Decide whether to go forward or backward
        should_back_up = abs(backward_angle_diff) < abs(forward_angle_diff) and abs(backward_angle_diff) < 90
        
        # CRITICAL: Ball collection AND goal delivery MUST be forward motion only!
        if target_type == "collect" or target_type == "goal" or target_type == "approach":
            should_back_up = False
        
        # SAFETY: Check if backing up would be safe
        if should_back_up and self.navigation.walls:
            backup_angle_rad = math.radians(self.robot_heading + 180)
            backup_safe = True
            min_safe_distance = 12  # cm
            
            # Check points along the backwards path
            for check_distance in [10, 20, min(30, distance_to_target)]:
                if check_distance > distance_to_target:
                    break
                    
                test_x = self.robot_pos[0] + check_distance * math.cos(backup_angle_rad)
                test_y = self.robot_pos[1] + check_distance * math.sin(backup_angle_rad)
                
                min_wall_distance = self.navigation.check_wall_proximity((test_x, test_y))
                
                if min_wall_distance < min_safe_distance:
                    backup_safe = False
                    logger.info(f"Safety override: backing up would be unsafe")
                    break
            
            # Check if robot is already close to walls
            current_min_wall_distance = self.navigation.check_wall_proximity(self.robot_pos)
            if current_min_wall_distance < 20:
                backup_safe = False
                logger.info(f"Safety override: robot in corner/near wall, forcing forward movement")
            
            if not backup_safe:
                should_back_up = False
        
        if should_back_up:
            angle_diff = backward_angle_diff
            logger.info(f"Backing up to target")
        else:
            angle_diff = forward_angle_diff
            logger.info(f"Going forward to target")
        
        # Turn to face the right direction
        if abs(angle_diff) > 2:
            logger.info(f"Turning {angle_diff:.1f} degrees")
            if not self.robot.turn(angle_diff):
                return False
            self.update_robot_position()
        
        # Move to target
        collection_threshold = 25 if target_type == "collect" else 15
        if distance_to_target > collection_threshold:
            return self._move_with_position_check(target_pos, target_type, should_back_up, distance_to_target)
        else:
            # Short movement - just go directly
            if target_type == "collect":
                logger.info(f"Collecting ball at {distance_to_target:.1f}cm")
                return self.robot.collect(distance_to_target)
            else:
                move_distance = -distance_to_target if should_back_up else distance_to_target
                action = "Backing up" if should_back_up else "Moving forward"
                logger.info(f"{action} {abs(move_distance):.1f}cm")
                return self.robot.move(move_distance)
    
    def _move_with_position_check(self, target_pos: Tuple[float, float], target_type: str, is_backing: bool, total_distance: float) -> bool:
        """Move to target in steps with position checking"""
        step_size = 10  # Move in 10cm increments
        steps_taken = 0
        
        estimated_steps = int(total_distance / step_size) + 5
        max_steps = max(50, estimated_steps)
        
        logger.info(f"Starting stepped movement: {total_distance:.1f}cm distance, max {max_steps} steps allowed")
        
        while steps_taken < max_steps:
            # Update current position
            if not self.update_robot_position():
                logger.warning("Lost robot tracking during movement")
                return False
            
            # Check for wall proximity before each step
            if not self.check_wall_proximity():
                logger.warning("Wall avoidance interrupted movement")
                if not self.update_robot_position():
                    return False
            
            # Calculate remaining distance and direction
            remaining_distance = calculate_distance(self.robot_pos, target_pos)
            target_angle = calculate_angle(self.robot_pos, target_pos)
            
            # Check if we've reached the target
            distance_tolerance = 15 if target_type == "goal" else 12
            if remaining_distance <= distance_tolerance:
                logger.info(f"Reached target with position checking")
                return True
            
            # Final approach for ball collection
            if target_type == "collect" and remaining_distance <= 22:
                logger.info(f"Final collection approach: {remaining_distance:.1f}cm")
                return self.robot.collect(remaining_distance)
            
            # Recalculate direction
            angle_diff = (target_angle - self.robot_heading + 180) % 360 - 180
            
            if is_backing:
                angle_diff = (angle_diff + 180) % 360 - 180
            
            # Adjust heading
            angle_threshold = 3 if target_type == "goal" else 2
            if abs(angle_diff) > angle_threshold:
                logger.info(f"Adjusting direction: turning {angle_diff:.1f} degrees")
                if not self.robot.turn(angle_diff):
                    return False
                self.update_robot_position()
            
            # Calculate next step distance
            step_distance = min(step_size, remaining_distance)
            
            # Move one step
            move_distance = -step_distance if is_backing else step_distance
            action = "Backing" if is_backing else "Forward"
            logger.info(f"{action} step: {step_distance:.1f}cm (remaining: {remaining_distance:.1f}cm)")
            
            if not self.robot.move(move_distance):
                logger.error("Movement step failed")
                return False
            
            steps_taken += 1
            
            # Update live tracking visualization
            if not self.update_live_tracking(target_pos, remaining_distance, steps_taken, max_steps):
                logger.warning("Movement interrupted by user")
                return False
            
            time.sleep(0.3)
        
        # Final position check
        if not self.update_robot_position():
            logger.warning("Lost robot tracking at end of movement")
            return False
            
        final_distance = calculate_distance(self.robot_pos, target_pos)
        logger.warning(f"Max steps reached, still {final_distance:.1f}cm from target")
        return False

    def update_live_tracking(self, target_pos, remaining_distance, current_step, max_steps):
        """Update live visualization during movement"""
        try:
            ret, frame = self.vision.get_fresh_frame()
            if not ret:
                return True
                
            if not self.update_robot_position(frame):
                return True
            
            # Draw overlays
            frame = self.draw_walls(frame)
            frame = self.vision.draw_robot_markers(frame)
            frame = self.draw_status(frame)
            
            # Draw target position
            if self.vision.homography_matrix is not None:
                target_cm = np.array([[[target_pos[0], target_pos[1]]]], dtype="float32")
                target_px = cv2.perspectiveTransform(target_cm, self.vision.homography_matrix)[0][0].astype(int)
                
                pulse_size = 8 + int(3 * math.sin(current_step * 0.5))
                cv2.circle(frame, tuple(target_px), pulse_size, (0, 255, 255), 2)
                cv2.putText(frame, "TARGET", (target_px[0] + 15, target_px[1] - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Progress information
            progress_pct = int((current_step / max_steps) * 100) if max_steps > 0 else 0
            status_lines = [
                f"LIVE MOVEMENT TRACKING",
                f"Target: ({target_pos[0]:.1f}, {target_pos[1]:.1f}) cm",
                f"Robot: ({self.robot_pos[0]:.1f}, {self.robot_pos[1]:.1f}) cm", 
                f"Remaining: {remaining_distance:.1f} cm",
                f"Step: {current_step}/{max_steps} ({progress_pct}%)",
                f"Press ESC to stop movement"
            ]
            
            # Draw status background
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, frame.shape[0] - 150), (400, frame.shape[0] - 10), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Draw status text
            y_start = frame.shape[0] - 135
            for i, line in enumerate(status_lines):
                color = (0, 255, 0) if i == 0 else (255, 255, 255)
                cv2.putText(frame, line, (15, y_start + i * 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            cv2.imshow("Path Planning", frame)
            
            # Check for user input to stop movement
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                logger.warning("Movement stopped by user (ESC pressed)")
                return False
                
        except Exception as e:
            logger.warning(f"Live tracking update failed: {e}")
            
        return True

    def draw_walls(self, frame):
        """Draw walls, safety margins, goals, and obstacles on the frame"""
        if self.vision.homography_matrix is None or not self.navigation.walls:
            return frame
           
        overlay = frame.copy()
        
        # Draw each wall segment
        for wall in self.navigation.walls:
            start_cm = np.array([[[wall[0], wall[1]]]], dtype="float32")
            end_cm = np.array([[[wall[2], wall[3]]]], dtype="float32")
            
            start_px = cv2.perspectiveTransform(start_cm, self.vision.homography_matrix)[0][0].astype(int)
            end_px = cv2.perspectiveTransform(end_cm, self.vision.homography_matrix)[0][0].astype(int)
            
            cv2.line(overlay, tuple(start_px), tuple(end_px), (0, 0, 255), 2)
        
        # Draw goal areas
        for goal_label, goal_points in self.navigation.goal_ranges.items():
            if goal_label == self.navigation.selected_goal:
                goal_color = (0, 255, 0)  # Green for selected goal
            else:
                goal_color = (0, 255, 255)  # Yellow for available goal
            
            for (x_cm, y_cm) in goal_points:
                pt_cm = np.array([[[x_cm, y_cm]]], dtype="float32")
                pt_px = cv2.perspectiveTransform(pt_cm, self.vision.homography_matrix)[0][0].astype(int)
                cv2.circle(overlay, tuple(pt_px), 4, goal_color, -1)
            
            # Draw goal label
            if goal_points:
                mid_y = sum(y for (x, y) in goal_points) / len(goal_points)
                label_x = goal_points[0][0]
                
                if label_x == 0:  # Left side
                    label_x += 10
                else:  # Right side
                    label_x -= 15
                
                label_cm = np.array([[[label_x, mid_y]]], dtype="float32")
                label_px = cv2.perspectiveTransform(label_cm, self.vision.homography_matrix)[0][0].astype(int)
                cv2.putText(overlay, f"Goal {goal_label}", tuple(label_px),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, goal_color, 2)
        
        # Draw obstacle area
        if (IGNORED_AREA["x_max"] > IGNORED_AREA["x_min"] and 
            IGNORED_AREA["y_max"] > IGNORED_AREA["y_min"]):
            
            obstacle_corners = [
                [IGNORED_AREA["x_min"], IGNORED_AREA["y_min"]],
                [IGNORED_AREA["x_max"], IGNORED_AREA["y_min"]],
                [IGNORED_AREA["x_max"], IGNORED_AREA["y_max"]],
                [IGNORED_AREA["x_min"], IGNORED_AREA["y_max"]]
            ]
            
            obstacle_px_points = []
            for corner in obstacle_corners:
                corner_cm = np.array([[[corner[0], corner[1]]]], dtype="float32")
                corner_px = cv2.perspectiveTransform(corner_cm, self.vision.homography_matrix)[0][0].astype(int)
                obstacle_px_points.append(corner_px)
            
            obstacle_pts = np.array(obstacle_px_points, np.int32)
            cv2.fillPoly(overlay, [obstacle_pts], (0, 165, 255))  # Orange
            cv2.polylines(overlay, [obstacle_pts], True, (0, 100, 200), 3)
            
            center_x = sum(pt[0] for pt in obstacle_px_points) // 4
            center_y = sum(pt[1] for pt in obstacle_px_points) // 4
            cv2.putText(overlay, "OBSTACLE", (center_x - 40, center_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Blend overlay with original frame
        alpha = 0.3
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        return frame

    def draw_status(self, frame):
        """Draw robot status information on frame"""
        status = self.robot.get_status()
        
        # Draw status box
        x, y = 10, 30
        line_height = 20
        
        cv2.rectangle(frame, (5, 5), (220, 150), (0, 0, 0), -1)
        cv2.rectangle(frame, (5, 5), (220, 150), (255, 255, 255), 1)
        
        # Goal status
        goal_type = "small" if self.navigation.selected_goal == 'A' else "large"
        cv2.putText(frame, f"Selected: Goal {self.navigation.selected_goal} ({goal_type})",
                   (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(frame, f"Small goal side: {self.navigation.small_goal_side}",
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

    def draw_path(self, frame, path):
        """Draw the planned path on the frame"""
        if not self.vision.homography_matrix:
            return frame
            
        overlay = frame.copy()
        
        # Colors for different point types
        colors = {
            'approach': (0, 255, 255),         # Yellow
            'tank_approach': (0, 255, 255),    # Yellow
            'collect': (0, 255, 0),            # Green
            'goal': (0, 0, 255),               # Red
            'backward_approach': (255, 128, 0), # Orange
            'backward_collect': (255, 165, 0),  # Orange red
        }
        
        # Draw lines between points
        for i in range(len(path) - 1):
            start_cm = np.array([[[path[i]['pos'][0], path[i]['pos'][1]]]], dtype="float32")
            end_cm = np.array([[[path[i+1]['pos'][0], path[i+1]['pos'][1]]]], dtype="float32")
            
            start_px = cv2.perspectiveTransform(start_cm, self.vision.homography_matrix)[0][0].astype(int)
            end_px = cv2.perspectiveTransform(end_cm, self.vision.homography_matrix)[0][0].astype(int)
            
            cv2.line(overlay, tuple(start_px), tuple(end_px), (150, 150, 150), 2)
        
        # Draw points
        for point in path:
            pt_cm = np.array([[[point['pos'][0], point['pos'][1]]]], dtype="float32")
            pt_px = cv2.perspectiveTransform(pt_cm, self.vision.homography_matrix)[0][0].astype(int)
            
            color = colors[point['type']]
            cv2.circle(overlay, tuple(pt_px), 5, color, -1)
            
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
        robot_px = cv2.perspectiveTransform(robot_cm, self.vision.homography_matrix)[0][0].astype(int)
        
        cv2.circle(frame, tuple(robot_px), 8, (255, 0, 0), -1)
        
        # Robot heading
        angle_rad = math.radians(self.robot_heading)
        end_x = robot_px[0] + int(25 * math.cos(angle_rad))
        end_y = robot_px[1] + int(25 * math.sin(angle_rad))
        cv2.arrowedLine(frame, tuple(robot_px), (end_x, end_y), (255, 0, 0), 2)
        
        return frame

    def collect_balls(self):
        """Main ball collection loop"""
        cv2.namedWindow("Path Planning")
        last_plan_time = 0
        last_plan_positions = []
        path = []
        show_debug = False
        
        while True:
            try:
                # Get fresh frame
                ret, frame = self.vision.get_fresh_frame()
                if not ret:
                    continue
                
                # Frame skipping for performance
                self.frame_skip_counter += 1
                skip_heavy_processing = (self.frame_skip_counter % self.frame_skip_interval) != 0

                # Update robot position
                if not self.update_robot_position(frame):
                    logger.warning("Could not detect robot markers")
                    cv2.putText(frame, "Robot markers not detected!", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.7, (0, 0, 255), 2)

                # Draw overlays
                frame = self.draw_walls(frame)
                frame = self.vision.draw_robot_markers(frame)
                frame = self.draw_status(frame)
                
                # Show debug view if enabled
                if show_debug:
                    frame = self.vision.show_detection_debug(frame)

                # Detect balls
                balls = []
                if not skip_heavy_processing:
                    balls = self.vision.detect_balls()
                current_time = time.time()
                
                # Replan path if needed
                should_replan = False
                if self.robot_pos and not skip_heavy_processing:
                    min_plan_interval = 1 if not path else 3
                    
                    if current_time - last_plan_time >= min_plan_interval:
                        should_replan = True
                    elif balls and last_plan_positions:
                        # Check if any ball has moved significantly
                        for ball in balls:
                            if not any(calculate_distance((ball[0], ball[1]), (old[0], old[1])) < 10 
                                     for old in last_plan_positions):
                                should_replan = True
                                break
                    else:
                        should_replan = True

                if should_replan:
                    last_plan_time = current_time
                    last_plan_positions = [(b[0], b[1]) for b in balls]
                    
                    # Plan path through all balls
                    path = []
                    current_pos = self.robot_pos
                    current_heading = self.robot_heading

                    if balls:
                        logger.info(f"Planning to collect {len(balls)} balls from position {self.robot_pos}")
                        
                        # Enable automatic mode when balls are detected
                        if not self.automatic_mode:
                            self.automatic_mode = True
                            logger.info("🤖 AUTOMATIC MODE ENABLED - Robot will now operate autonomously")
                        
                        # Sort balls by distance
                        balls.sort(key=lambda b: calculate_distance(self.robot_pos, (b[0], b[1])))

                        # Take up to MAX_BALLS_PER_TRIP closest balls
                        current_batch = balls[:MAX_BALLS_PER_TRIP]
                        remaining_balls = current_batch.copy()

                        # Add balls to path with obstacle avoidance
                        while remaining_balls:
                            closest_ball = min(remaining_balls, 
                                             key=lambda b: calculate_distance(current_pos, (b[0], b[1])))
                            
                            ball_pos = (closest_ball[0], closest_ball[1])
                            
                            # Plan path around obstacle if needed
                            waypoints = self.navigation.plan_path_around_obstacle(current_pos, ball_pos)
                            
                            # Add waypoint approach points
                            for i, waypoint in enumerate(waypoints[:-1]):
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

                    # ALWAYS add goal approach path for delivery
                    goal_path = self.navigation.calculate_goal_approach_path(current_pos, current_heading)
                    path.extend(goal_path)
                    
                    balls_in_path = len([p for p in path if p['type'] == 'collect'])
                    logger.info(f"Added goal delivery path with {len(goal_path)} waypoints (balls in path: {balls_in_path})")
                    
                    # If automatic mode is enabled, execute path immediately
                    if self.automatic_mode and path:
                        logger.info("🤖 Auto-executing path in automatic mode...")
                        time.sleep(0.5)
                        key = 13  # Enter key code

                # Prepare display frame
                display_frame = frame.copy()
                
                # Show path preview
                if path:
                    display_frame = self.draw_path(display_frame, path)
                    if self.automatic_mode:
                        cv2.putText(display_frame, "AUTO MODE: Executing path automatically...", 
                                  (10, display_frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.7, (0, 255, 0), 2)
                    else:
                        cv2.putText(display_frame, "PATH READY - Press ENTER to execute", 
                                  (10, display_frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.7, (0, 255, 0), 2)
                
                # Get key input
                if not (self.automatic_mode and path):
                    key = cv2.waitKey(1) & 0xFF
                else:
                    if 'key' not in locals():
                        key = cv2.waitKey(1) & 0xFF
                
                # Add command help
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
                
                # Show status
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
                        cv2.putText(display_frame, "MANUAL MODE: Press ENTER to execute",
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
                    self.navigation.select_goal('A')
                elif key == ord('2'):
                    self.navigation.select_goal('B')
                elif key == ord('3'):
                    new_side = "right" if self.navigation.small_goal_side == "left" else "left"
                    self.navigation.set_goal_side(new_side)
                elif key == ord('r'):
                    logger.info("🔄 Starting recalibration...")
                    cv2.destroyWindow("Path Planning")
                    self.vision.calibration_points = []
                    self.vision.obstacle_points = []
                    self.vision.homography_matrix = None
                    self.vision.calibrate()
                    self.navigation.setup_walls()
                    cv2.namedWindow("Path Planning")
                    logger.info("✅ Recalibration complete")
                elif key == 13:  # Enter key
                    if path:
                        self.execute_path(path)
                        path = []
                        # Look for remaining balls
                        time.sleep(1)
                        ret, scan_frame = self.vision.get_fresh_frame()
                        if ret and self.update_robot_position(scan_frame):
                            remaining_balls = self.vision.detect_balls()
                            if remaining_balls:
                                logger.info(f"🎯 Found {len(remaining_balls)} remaining balls")
                                if not self.automatic_mode:
                                    self.automatic_mode = True
                                    logger.info("🤖 AUTOMATIC MODE ENABLED")
                                last_plan_time = 0
                                last_plan_positions = []
                            else:
                                logger.info("🏁 No more balls detected - collection complete!")
                                cv2.waitKey(2000)
                        else:
                            logger.warning("⚠️ Could not scan for remaining balls")
                            cv2.waitKey(2000)
                    else:
                        logger.warning("No path planned to execute")
                    
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                cv2.waitKey(1000)

        cv2.destroyWindow("Path Planning")

    def execute_path(self, path):
        """Execute the planned path"""
        logger.info("Executing path with {} points".format(len(path)))
        
        try:
            for point in path:
                target_pos = point['pos']
                
                if point['type'] == 'collect':
                    logger.info(f"Collecting {point['ball_type']} ball")
                    if not self.move_to_target_simple(target_pos, "collect"):
                        raise Exception("Ball collection failed")
                elif point['type'] == 'approach':
                    logger.info(f"Moving to staging area")
                    if not self.move_to_target_simple(target_pos, "approach"):
                        raise Exception("Approach movement failed")
                elif point['type'] == 'goal':
                    logger.info(f"Moving to goal")
                    if not self.move_to_target_simple(target_pos, "goal"):
                        raise Exception("Goal movement failed")
                    
                    # Final alignment before delivery
                    logger.info("Final alignment for goal delivery")
                    goal_side = self.navigation.goal_ranges[self.navigation.selected_goal][0][0]
                    if goal_side == 0:  # Left goal
                        target_angle = 180  # Face left (west)
                    else:  # Right goal  
                        target_angle = 0    # Face right (east)
                    
                    # Calculate angle difference and align if needed
                    angle_diff = (target_angle - self.robot_heading + 180) % 360 - 180
                    if abs(angle_diff) > 5:
                        logger.info(f"Aligning with goal: turning {angle_diff:.1f} degrees")
                        if not self.robot.turn(angle_diff):
                            logger.warning("Final alignment failed, but continuing")
                        else:
                            self.update_robot_position()
                    
                    # Move collector into goal
                    logger.info("Moving collector into goal for proper delivery")
                    if not self.robot.move(8):
                        raise Exception("Failed to move collector into goal")
                    
                    # Deliver balls
                    logger.info("Starting ball delivery sequence")
                    delivery_success = False
                    try:
                        delivery_success = self.robot.deliver_balls(DELIVERY_TIME)
                        if delivery_success:
                            logger.info("✅ Ball delivery completed successfully")
                        else:
                            logger.warning("⚠️ Ball delivery failed, but continuing with backup")
                    except Exception as delivery_error:
                        logger.error(f"⚠️ Ball delivery error: {delivery_error}, but continuing with backup")
                    
                    # Always back up after delivery
                    logger.info("Backing up 20cm after delivery")
                    if not self.robot.move(-20):
                        logger.error("Failed to back up after delivery")
                    
                    if delivery_success:
                        logger.info("✅ Delivery sequence complete")
                    else:
                        logger.info("⚠️ Delivery had issues but backup completed")
                
                # Update visualization during execution
                ret, frame = self.vision.get_fresh_frame()
                if ret:
                    self.update_robot_position(frame)
                    frame = self.draw_status(frame)
                    frame = self.vision.draw_robot_markers(frame)
                    frame_with_path = self.draw_path(frame, path)
                    cv2.putText(frame_with_path, "EXECUTING PATH LIVE...", 
                              (10, frame_with_path.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.7, (0, 255, 0), 2)
                    cv2.imshow("Path Planning", frame_with_path)
                    cv2.waitKey(1)
            
            logger.info("✅ Path execution completed successfully")
            
        except Exception as e:
            logger.error("Path execution failed: {}".format(e))
            self.robot.stop()
            cv2.waitKey(2000)

    def run(self):
        """Main run loop"""
        try:
            # Calibrate the camera perspective
            logger.info("Starting camera calibration...")
            self.vision.calibrate()
            
            if self.vision.homography_matrix is None:
                logger.error("Calibration failed")
                return
            
            # Setup navigation with calibrated obstacle area
            self.navigation.setup_walls()
            
            # Start continuous collection
            logger.info("Starting ball collection...")
            self.collect_balls()
            
        except KeyboardInterrupt:
            logger.info("Stopping...")
        finally:
            self.robot.stop()
            self.vision.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    collector = BallCollector()
    collector.run() 