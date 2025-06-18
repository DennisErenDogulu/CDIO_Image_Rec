#!/usr/bin/env python3
"""
Continuous Ball Collection Client (vision + receding-horizon planner)

This script implements an online, receding-horizon ball collection strategy:
1. Calibrate overhead camera → compute homography
2. Build wall obstacles and A* planner
3. Repeatedly:
   a. Detect balls and robot pose via vision
   b. Replan a short path through the next batch of balls
   c. Execute that path by issuing TURN / MOVE / COLLECT commands
"""
import cv2
import math
import json
import socket
import logging
import numpy as np
import time
from typing import List, Tuple

import cv2.aruco as aruco

# ArUco marker detection parameters
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_4X4_50)
ARUCO_PARAMS = aruco.DetectorParameters_create()

from roboflow import Roboflow
from receding_horizon_planner import AStarPlanner

# —— Configuration —————————————————————————————————————————
EV3_IP, EV3_PORT = "172.20.10.7", 12345

ROBOFLOW_API_KEY = "qJTLU5ku2vpBGQUwjBx2"
RF_WORKSPACE, RF_PROJECT, RF_VERSION = "cdio-nczdp", "cdio-golfbot2025", 12

FIELD_W_CM, FIELD_H_CM = 180, 120    # Table dimensions in cm
WALL_MARGIN_CM    = 3               # Safety margin from walls
SAFETY_RADIUS_CM  = WALL_MARGIN_CM  # Obstacle radius for planner
GOAL_WIDTH_CM     = 30              # Width of goal opening in cm

MAX_BATCH         = 3               # Balls per trip
REPLAN_INTERVAL   = 3.0             # Seconds between replans

COLLECT_DIST_CM   = 20              # Distance for COLLECT command
APPROACH_DIST_CM  = 30              # Distance to stop before ball

# Ignored central area (e.g. center obstacle)
IGNORED_AREA = dict(x_min=50, x_max=100, y_min=50, y_max=100)

# —— Logging —————————————————————————————————————————————
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)


def point_to_line_distance(p: Tuple[float,float],
                           a: Tuple[float,float],
                           b: Tuple[float,float]) -> float:
    """Shortest distance from point p to line segment a→b"""
    x,y = p; x1,y1 = a; x2,y2 = b
    vx, vy = x2 - x1, y2 - y1
    wx, wy = x - x1, y - y1
    L2 = vx*vx + vy*vy
    if L2 == 0:
        return math.hypot(wx, wy)
    t = max(0, min(1, (wx*vx + wy*vy) / L2))
    proj_x = x1 + t*vx
    proj_y = y1 + t*vy
    return math.hypot(x - proj_x, y - proj_y)


def check_wall_collision(p1: Tuple[float,float],
                         p2: Tuple[float,float],
                         walls: List[Tuple[float,float,float,float]],
                         margin: float) -> bool:
    """
    Return True if the segment p1→p2 comes within `margin` of any wall,
    or intersects the wall segment.
    Walls are list of (x1,y1,x2,y2).
    """
    for x1,y1,x3,y3 in walls:
        a = (x1,y1); b = (x3,y3)
        # endpoint too close?
        if (point_to_line_distance(p1,a,b) < margin or
            point_to_line_distance(p2,a,b) < margin):
            return True
        # segment intersection test
        x2,y2 = p2; x4,y4 = b
        denom = (p1[0]-p2[0])*(y3-y4) - (p1[1]-p2[1])*(x3-x4)
        if denom == 0:
            continue
        t = ((p1[0]-x3)*(y3-y4) - (p1[1]-y3)*(x3-x4)) / denom
        u = -((p1[0]-p2[0])*(p1[1]-y3) - (p1[1]-p2[1])*(p1[0]-x3)) / denom
        if 0 <= t <= 1 and 0 <= u <= 1:
            return True
    return False


class BallCollector:
    def __init__(self):
        # Roboflow model
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        proj = rf.workspace(RF_WORKSPACE).project(RF_PROJECT)
        self.model = proj.version(RF_VERSION).model

        # Camera
        self.cap = cv2.VideoCapture(1)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # State
        self.robot_pos = (20.0, 20.0)
        self.robot_heading = 0.0  # degrees

        # Calibration
        self.cal_pts: List[Tuple[int,int]] = []
        self.H: np.ndarray = None
        self.H_inv: np.ndarray = None

        # Walls & planner
        self.walls: List[Tuple[float,float,float,float]] = []
        self.planner: AStarPlanner = None

        # For replanning logic
        self.last_replan = 0.0
        self.prev_balls: List[Tuple[float,float]] = []

    def setup_walls_and_planner(self):
        """Define table walls (excluding goal) and init A* planner."""
        W, H, M = FIELD_W_CM, FIELD_H_CM, WALL_MARGIN_CM
        gy0 = (H - GOAL_WIDTH_CM)/2
        gy1 = gy0 + GOAL_WIDTH_CM
        # bottom, top, left segments, right segments
        self.walls = [
            (M, M, W - M, M),
            (M, H - M, W - M, H - M),
            (M, M, M, gy0), (M, gy1, M, H - M),
            (W - M, M, W - M, gy0), (W - M, gy1, W - M, H - M),
        ]
        # build circular obstacles at each segment endpoint
        obs = []
        for (x1,y1,x2,y2) in self.walls:
            obs.append((x1, y1, SAFETY_RADIUS_CM))
            obs.append((x2, y2, SAFETY_RADIUS_CM))
        self.planner = AStarPlanner(
            x_range=(0, FIELD_W_CM),
            y_range=(0, FIELD_H_CM),
            resolution=2.0,
            obstacles=obs
        )

    def calibrate(self):
        """Click 4 image corners → compute H and H_inv, then walls & planner."""
        def on_mouse(evt, x, y, flags, param):
            if evt == cv2.EVENT_LBUTTONDOWN and len(self.cal_pts) < 4:
                self.cal_pts.append((x, y))
                logger.info(f"Calib pt {len(self.cal_pts)}: ({x},{y})")
                if len(self.cal_pts) == 4:
                    dst = np.array([
                        [0, 0],
                        [FIELD_W_CM, 0],
                        [FIELD_W_CM, FIELD_H_CM],
                        [0, FIELD_H_CM]
                    ], np.float32)
                    src = np.array(self.cal_pts, np.float32)
                    self.H = cv2.getPerspectiveTransform(dst, src)
                    self.H_inv = np.linalg.inv(self.H)
                    self.setup_walls_and_planner()
                    logger.info("Calibration complete.")

        cv2.namedWindow("Calibrate")
        cv2.setMouseCallback("Calibrate", on_mouse)
        while len(self.cal_pts) < 4:
            ret, frame = self.cap.read()
            if not ret: continue
            for idx, pt in enumerate(self.cal_pts):
                cv2.circle(frame, pt, 5, (0,255,0), -1)
                cv2.putText(frame, str(idx+1), (pt[0]+5, pt[1]-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.imshow("Calibrate", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyWindow("Calibrate")

    def detect_balls(self) -> List[Tuple[float,float,str]]:
        """Return list of (x_cm, y_cm, class) for each detected ball."""
        ret, frame = self.cap.read()
        if not ret: return []
        small = cv2.resize(frame, (416, 416))
        preds = self.model.predict(small, confidence=30).json().get("predictions", [])
        balls = []
        sx, sy = frame.shape[1]/416, frame.shape[0]/416
        for p in preds:
            x_px, y_px = int(p["x"]*sx), int(p["y"]*sy)
            pt = np.array([[[x_px,y_px]]], np.float32)
            x_cm, y_cm = cv2.perspectiveTransform(pt, self.H_inv)[0][0]
            if not (IGNORED_AREA["x_min"] <= x_cm <= IGNORED_AREA["x_max"]
                    and IGNORED_AREA["y_min"] <= y_cm <= IGNORED_AREA["y_max"]):
                balls.append((x_cm, y_cm, p["class"]))
        return balls

    def detect_robot(self):
        """Detect green & pink papers, update robot_pos & robot_heading."""
        ret, frame = self.cap.read()
        if not ret:
            return
        small = cv2.resize(frame, (416, 416))
        preds = self.model.predict(small, confidence=30).json().get("predictions", [])
    
        # Filter for your two marker classes
        greens = [p for p in preds if p["class"] == "green_paper"]
        pinks  = [p for p in preds if p["class"] == "pink_paper"]
        if not greens or not pinks:
            return
    
        # Pick the highest‐confidence box of each
        g = max(greens, key=lambda x: x["confidence"])
        p = max(pinks,  key=lambda x: x["confidence"])
    
        # Helper: pixel → world (cm)
        def pix2world(px, py):
            pt = np.array([[[px, py]]], np.float32)
            return cv2.perspectiveTransform(pt, self.H_inv)[0][0]
    
        # Scale Roboflow coords back to full frame
        fx, fy = frame.shape[1]/416, frame.shape[0]/416
        gx_px, gy_px = g["x"]*fx, g["y"]*fy
        px_px, py_px = p["x"]*fx, p["y"]*fy
        gx_cm, gy_cm = pix2world(gx_px, gy_px)
        px_cm, py_cm = pix2world(px_px, py_px)
    
        # Compute center & heading
        cx, cy = (gx_cm+px_cm)/2, (gy_cm+py_cm)/2
        heading = math.degrees(math.atan2(py_cm-gy_cm, px_cm-gx_cm))
    
        self.robot_pos = (cx, cy)
        self.robot_heading = heading
    def replan_if_needed(self, balls: List[Tuple[float,float,str]]):
        """Replan path every REPLAN_INTERVAL or when ball set changes."""
        now = time.time()
        moved = any(
            math.hypot(b[0]-b0[0], b[1]-b0[1]) > 10
            for b in balls for b0 in self.prev_balls
        ) if self.prev_balls else True

        if (now - self.last_replan > REPLAN_INTERVAL) and balls and moved:
            # choose nearest MAX_BATCH balls
            balls.sort(key=lambda b: math.hypot(b[0]-self.robot_pos[0], b[1]-self.robot_pos[1]))
            goals = [(b[0], b[1]) for b in balls[:MAX_BATCH]]
            self.path = self.planner.plan(self.robot_pos, goals)
            self.last_replan = now
            self.prev_balls = [(b[0], b[1]) for b in balls]
            logger.info(f"Replanned {len(self.path)} waypoints")

    def execute_path(self):
        """Follow each waypoint: turn, move, and collect when appropriate."""
        for wp in self.path:
            dx = wp[0] - self.robot_pos[0]
            dy = wp[1] - self.robot_pos[1]
            target_ang = math.degrees(math.atan2(dy, dx))
            turn_amt = ((target_ang - self.robot_heading + 180) % 360) - 180
            if abs(turn_amt) > 5:
                self.send_command("TURN", angle=turn_amt)
                self.robot_heading = target_ang
            dist = math.hypot(dx, dy)
            # if within COLLECTION_DIST, use COLLECT
            if dist < COLLECT_DIST_CM * 1.2:
                self.send_command("COLLECT", distance=COLLECT_DIST_CM)
            else:
                self.send_command("MOVE", distance=dist)
            self.robot_pos = (wp[0], wp[1])
            time.sleep(0.2)

    def collect_balls(self):
        """Main loop: detect, replan, execute until no balls remain."""
        while True:
            balls = self.detect_balls()
            if not balls:
                logger.info("All balls collected.")
                break
            self.detect_robot()
            self.replan_if_needed(balls)
            if hasattr(self, "path") and self.path:
                self.execute_path()

    def send_command(self, command: str, **params) -> bool:
        """Send a JSON command over TCP and expect 'OK'."""
        try:
            with socket.create_connection((EV3_IP, EV3_PORT), timeout=10) as s:
                msg = json.dumps({"command": command, **params}) + "\n"
                s.sendall(msg.encode())
                resp = s.recv(1024).decode().strip()
                return resp == "OK"
        except Exception as e:
            logger.error(f"Command {command} failed: {e}")
            return False

    def run(self):
        logger.info(">> Starting calibration")
        self.calibrate()
        if self.H is None:
            logger.error("Calibration aborted.")
            return
        logger.info(">> Starting collection")
        self.collect_balls()
        self.cap.release()
        logger.info(">> Done.")

if __name__ == "__main__":
    BallCollector().run()