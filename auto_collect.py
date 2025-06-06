#!/usr/bin/env python3
"""
Automatic ball collection runner.

This script captures a frame, plans a path to up to three nearest balls
and sends the path to the EV3 robot. After each path is executed,
a new frame is captured and the process repeats.
"""

import time
import cv2
import logging

from cv2_RoboflowDectection import (
    RoboFlowGridTest,
    EV3_IP,
    EV3_PORT,
    EV3_HEADING_DEFAULT,
    heuristic,
    astar,
    IGNORED_AREA,
    GOAL_OFFSET_CM,
)
from robot_connection.client_automated import send_path

logger = logging.getLogger("AutoCollect")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-8s %(message)s")


class AutoCollector(RoboFlowGridTest):
    def calibrate(self):
        cv2.namedWindow("Calibration")
        cv2.setMouseCallback("Calibration", self.click_to_set_corners)
        while self.homography_matrix is None:
            ret, frame = self.cap.read()
            if not ret:
                logger.error("Failed to read frame for calibration")
                continue
            cv2.imshow("Calibration", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyWindow("Calibration")

    def detect_balls(self, frame):
        positions = []
        small = cv2.resize(frame, (416, 416))
        preds = self.model.predict(small, confidence=30, overlap=20).json()
        scale_x = frame.shape[1] / 416
        scale_y = frame.shape[0] / 416
        for p in preds.get('predictions', []):
            x_px = int(p['x'] * scale_x)
            y_px = int(p['y'] * scale_y)
            label = p['class']
            cm_coords = self.pixel_to_cm(x_px, y_px)
            if cm_coords is not None:
                cx, cy = cm_coords
                if not (IGNORED_AREA["x_min"] <= cx <= IGNORED_AREA["x_max"] and
                        IGNORED_AREA["y_min"] <= cy <= IGNORED_AREA["y_max"]):
                    positions.append((cx, cy, label))
        return positions

    def select_closest(self, balls, count):
        start_g = self.cm_to_grid_coords(*self.start_point_cm)
        sorted_balls = sorted(
            balls,
            key=lambda b: heuristic(start_g, self.cm_to_grid_coords(b[0], b[1]))
        )
        return sorted_balls[:count]

    def build_path(self, balls):
        route_cm = [self.start_point_cm]
        current = self.start_point_cm
        remaining = balls.copy()
        while remaining:
            next_ball = min(
                remaining,
                key=lambda b: heuristic(
                    self.cm_to_grid_coords(*current),
                    self.cm_to_grid_coords(b[0], b[1])
                )
            )
            route_cm.append((next_ball[0], next_ball[1]))
            current = (next_ball[0], next_ball[1])
            remaining.remove(next_ball)

        goal_points = self.goal_range[self.selected_goal]
        y_mid = (goal_points[0][1] + goal_points[-1][1]) / 2
        is_left = self.selected_goal == 'A' and self.small_goal_side == "left"
        if self.selected_goal == 'B':
            is_left = not is_left
        x_target = GOAL_OFFSET_CM if is_left else self.real_width_cm - GOAL_OFFSET_CM
        route_cm.append((x_target, y_mid))

        self.full_grid_path = []
        max_gx = self.real_width_cm // self.grid_spacing_cm
        max_gy = self.real_height_cm // self.grid_spacing_cm
        for i in range(len(route_cm) - 1):
            start_cell = self.cm_to_grid_coords(*route_cm[i])
            end_cell = self.cm_to_grid_coords(*route_cm[i + 1])
            segment = astar(start_cell, end_cell, self.obstacles, max_gx, max_gy)
            if self.full_grid_path and segment and segment[0] == self.full_grid_path[-1]:
                segment = segment[1:]
            self.full_grid_path.extend(segment)
        return self.full_grid_path

    def run_cycle(self, balls_per_cycle=3):
        ret, frame = self.cap.read()
        if not ret:
            logger.error("Could not grab frame")
            return False
        balls = self.detect_balls(frame)
        if not balls:
            logger.info("No balls detected")
            return False
        selected = self.select_closest(balls, balls_per_cycle)
        path = self.build_path(selected)
        if path:
            logger.info("Sending path with %d points", len(path))
            send_path(EV3_IP, EV3_PORT, path, EV3_HEADING_DEFAULT)
            time.sleep(1)
            return True
        return False


def main():
    collector = AutoCollector()
    collector.calibrate()
    total_balls = 11
    collected = 0
    while collected < total_balls:
        if collector.run_cycle():
            collected += 3
        else:
            time.sleep(1)
    collector.cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()