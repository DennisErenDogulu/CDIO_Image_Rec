#!/usr/bin/env python3
"""
Navigation Module

Handles path planning, obstacle avoidance, movement calculations, and goal management.
"""

import math
import logging
from typing import List, Tuple, Dict, Optional

from .config import (
    FIELD_WIDTH_CM, FIELD_HEIGHT_CM, WALL_SAFETY_MARGIN,
    SMALL_GOAL_SIDE, APPROACH_DISTANCE_CM, IGNORED_AREA
)
from .utils import point_to_line_distance, check_wall_collision, calculate_distance, calculate_angle

logger = logging.getLogger(__name__)


class NavigationSystem:
    """Handles all navigation tasks including path planning and obstacle avoidance"""
    
    def __init__(self, small_goal_side=SMALL_GOAL_SIDE):
        self.small_goal_side = small_goal_side.lower()
        self.selected_goal = 'A'  # 'A' (small goal) or 'B' (large goal)
        self.goal_ranges = self._build_goal_ranges()
        self.walls = []
        
    def _build_goal_ranges(self) -> dict:
        """Build goal ranges for both goals A and B based on small_goal_side setting"""
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
    
    def set_goal_side(self, side: str):
        """Change which side the small goal is on"""
        self.small_goal_side = side.lower()
        self.goal_ranges = self._build_goal_ranges()
        self.setup_walls()  # Rebuild walls to exclude new goal positions
        logger.info(f"Set small goal side to {self.small_goal_side}")
    
    def select_goal(self, goal: str):
        """Select which goal to target ('A' for small, 'B' for large)"""
        self.selected_goal = goal.upper()
        logger.info(f"Selected Goal {self.selected_goal}")

    def setup_walls(self):
        """Set up wall segments based on calibration points, excluding goal areas"""
        margin = WALL_SAFETY_MARGIN
        
        # Get goal Y ranges for both goals with extra clearance
        goal_a_y_vals = [y for (x, y) in self.goal_ranges['A']]
        goal_b_y_vals = [y for (x, y) in self.goal_ranges['B']]
        
        # Add extra clearance around goals
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
        
        # Add obstacle walls if obstacle is defined
        if (IGNORED_AREA["x_max"] > IGNORED_AREA["x_min"] and 
            IGNORED_AREA["y_max"] > IGNORED_AREA["y_min"]):
            
            m = WALL_SAFETY_MARGIN
            x_min = IGNORED_AREA["x_min"]
            x_max = IGNORED_AREA["x_max"]
            y_min = IGNORED_AREA["y_min"]
            y_max = IGNORED_AREA["y_max"]
            
            self.walls.extend([
                (x_min - m, y_min - m, x_max + m, y_min - m),  # top
                (x_max + m, y_min - m, x_max + m, y_max + m),  # right
                (x_max + m, y_max + m, x_min - m, y_max + m),  # bottom
                (x_min - m, y_max + m, x_min - m, y_min - m)   # left
            ])

    def check_obstacle_collision(self, start_pos: Tuple[float, float], end_pos: Tuple[float, float]) -> bool:
        """Check if a direct path would intersect with the obstacle"""
        if (IGNORED_AREA["x_max"] <= IGNORED_AREA["x_min"] or 
            IGNORED_AREA["y_max"] <= IGNORED_AREA["y_min"]):
            return False  # No obstacle defined
        
        # Add bigger safety margin around obstacle for collision detection
        margin = 15  # Much larger margin for obstacle avoidance
        x_min = IGNORED_AREA["x_min"] - margin
        x_max = IGNORED_AREA["x_max"] + margin
        y_min = IGNORED_AREA["y_min"] - margin
        y_max = IGNORED_AREA["y_max"] + margin
        
        x1, y1 = start_pos
        x2, y2 = end_pos
        
        # Check if either endpoint is inside the expanded obstacle
        if (x_min <= x1 <= x_max and y_min <= y1 <= y_max) or \
           (x_min <= x2 <= x_max and y_min <= y2 <= y_max):
            return True
        
        # Sample points along the line and check if any fall within obstacle
        num_samples = 20
        for i in range(num_samples + 1):
            t = i / num_samples
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            
            if x_min <= x <= x_max and y_min <= y <= y_max:
                return True
        
        return False

    def plan_path_around_obstacle(self, start_pos: Tuple[float, float], target_pos: Tuple[float, float]) -> List[Tuple[float, float]]:
        """Plan a path around the obstacle if direct path is blocked"""
        # First check if direct path is clear
        if not self.check_obstacle_collision(start_pos, target_pos):
            return [target_pos]  # Direct path is fine
        
        logger.info(f"Direct path blocked by obstacle, planning adaptive detour...")
        
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
        
        # Route 1: Go around left side
        if x_min > field_margin:
            left_bottom = (x_min, max(y_min, start_pos[1] - 10))
            left_top = (x_min, min(y_max, target_pos[1] + 10))
            routes.append(("left", [left_bottom, left_top, target_pos]))
        
        # Route 2: Go around right side
        if x_max < FIELD_WIDTH_CM - field_margin:
            right_bottom = (x_max, max(y_min, start_pos[1] - 10))
            right_top = (x_max, min(y_max, target_pos[1] + 10))
            routes.append(("right", [right_bottom, right_top, target_pos]))
        
        # Route 3: Go around bottom
        if y_min > field_margin:
            bottom_left = (max(x_min, start_pos[0] - 10), y_min)
            bottom_right = (min(x_max, target_pos[0] + 10), y_min)
            routes.append(("bottom", [bottom_left, bottom_right, target_pos]))
        
        # Route 4: Go around top
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
                    check_wall_collision(current_pos, waypoint, self.walls, WALL_SAFETY_MARGIN)):
                    valid_route = False
                    break
                
                # Add distance
                total_distance += calculate_distance(current_pos, waypoint)
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

    def calculate_movement_efficiency(self, robot_pos: Tuple[float, float], robot_heading: float, target_pos: Tuple[float, float]) -> dict:
        """Calculate efficiency scores for different movement approaches - optimized for tank turns"""
        dx = target_pos[0] - robot_pos[0]
        dy = target_pos[1] - robot_pos[1]
        distance = calculate_distance(robot_pos, target_pos)
        target_angle = calculate_angle(robot_pos, target_pos)
        angle_diff = (target_angle - robot_heading + 180) % 360 - 180
        
        # Calculate efficiency for different approaches
        approaches = {}
        
        # 1. Direct tank turn + forward approach
        turn_cost = abs(angle_diff) / 360.0  # Reduced penalty for turns
        approaches['tank_turn_forward'] = {
            'total_cost': turn_cost + distance * 0.01,
            'method': 'tank_turn_forward',
            'angle': target_angle,
            'backing': False
        }
        
        # 2. Backward movement (only if it's significantly better)
        backward_angle = target_angle + 180
        backward_angle_diff = (backward_angle - robot_heading + 180) % 360 - 180
        # Only consider backward if it requires much less turning
        if abs(backward_angle_diff) < abs(angle_diff) * 0.5:
            approaches['backward'] = {
                'total_cost': abs(backward_angle_diff) / 360.0 + distance * 0.02,  # Higher distance penalty for backing
                'method': 'backward',
                'angle': backward_angle,
                'backing': True
            }
        
        return approaches

    def get_smooth_approach_vector(self, robot_pos: Tuple[float, float], robot_heading: float, target_pos: Tuple[float, float]) -> Tuple[Tuple[float, float], float, str]:
        """Calculate optimal approach using tank turn capabilities"""
        approaches = self.calculate_movement_efficiency(robot_pos, robot_heading, target_pos)
        
        # Choose the most efficient approach
        best_approach = min(approaches.values(), key=lambda x: x['total_cost'])
        method = best_approach['method']
        
        dx = target_pos[0] - robot_pos[0]
        dy = target_pos[1] - robot_pos[1]
        distance = calculate_distance(robot_pos, target_pos)
        
        if method == 'backward':
            # Move backward toward target
            if distance > APPROACH_DISTANCE_CM:
                ratio = (distance - APPROACH_DISTANCE_CM) / distance
                approach_x = robot_pos[0] + dx * ratio
                approach_y = robot_pos[1] + dy * ratio
            else:
                approach_x, approach_y = robot_pos
            return (approach_x, approach_y), best_approach['angle'], 'backward'
        
        # For tank turn forward (the most common case now)
        if distance > APPROACH_DISTANCE_CM:
            ratio = (distance - APPROACH_DISTANCE_CM) / distance
            approach_x = robot_pos[0] + dx * ratio
            approach_y = robot_pos[1] + dy * ratio
        else:
            approach_x, approach_y = robot_pos
        
        return (approach_x, approach_y), best_approach['angle'], 'tank_turn_forward'

    def calculate_goal_approach_path(self, current_pos: Tuple[float, float], current_heading: float) -> List[Dict]:
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
            goal_x = -2  # Position collector 2cm into goal opening
            staging_x = 30  # Only 30cm from left edge
        else:  # Right side goal
            goal_x = FIELD_WIDTH_CM + 2  # Position collector 2cm into goal opening
            staging_x = FIELD_WIDTH_CM - 30  # Only 30cm from right edge
        
        # Staging area Y coordinate - aligned with goal
        staging_y = goal_y
        
        # Calculate distance to staging area
        distance_to_staging = calculate_distance(current_pos, (staging_x, staging_y))
        
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

    def check_wall_proximity(self, position: Tuple[float, float]) -> float:
        """Check distance to nearest wall"""
        if not self.walls or not position:
            return float('inf')
        
        min_distance = float('inf')
        for wall in self.walls:
            wall_start = (wall[0], wall[1])
            wall_end = (wall[2], wall[3])
            distance = point_to_line_distance(position, wall_start, wall_end)
            min_distance = min(min_distance, distance)
        
        return min_distance 