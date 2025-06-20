# detection.py
#
# Run YOLO on each frame, detect balls/robot/obstacles,
# convert to world coordinates, and pass along to navigation.

import random
import cv2
import numpy as np
from ultralytics import YOLO
from queue import Empty
import time

from robot_client import config, navigation, robot_comm
from robot_client.navigation import planner
from robot_client import config as client_config

# === Global State ===
yolo_model = YOLO("weights_v4.pt")  # Adjust path if needed
ball_positions_cm = []
obstacles = set()
class_colors = {}

def pixel_to_cm(px, py):
    if client_config.inv_homography_matrix is None:
        return None
    pt = np.array([[[px, py]]], dtype="float32")
    real_pt = cv2.perspectiveTransform(pt, client_config.inv_homography_matrix)[0][0]
    return real_pt[0], real_pt[1]

def cm_to_grid_coords(x_cm, y_cm):
    return int(x_cm // client_config.GRID_SPACING_CM), int(y_cm // client_config.GRID_SPACING_CM)


def process_frames(frame_queue, output_queue, stop_event):
    global ball_positions_cm, obstacles

    # Track time for periodic pose sends
    last_pose_send = time.time()

    while not stop_event.is_set():
        # Grab either (frame, timestamp) or raw frame
        try:
            item = frame_queue.get(timeout=0.02)
        except Empty:
            continue

        if isinstance(item, tuple) and len(item) == 2:
            frame, capture_ts = item
        else:
            frame = item
            capture_ts = time.time()

        original = frame.copy()

        # — ArUco detection & update shared state —
        aruco_corners, aruco_ids = navigation.detect_aruco(original)
        if aruco_ids is not None:
            for idx, marker_id in enumerate(aruco_ids.flatten()):
                if marker_id != config.ARUCO_MARKER_ID:
                    continue
                pts = aruco_corners[idx][0]
                cx, cy = float(pts[:,0].mean()), float(pts[:,1].mean())
                real = navigation.pixel_to_cm(int(round(cx)), int(round(cy)))
                if real:
                    x_cm, y_cm = real
                    heading_deg = navigation.compute_aruco_heading(pts)
                    planner.robot_position_cm = (x_cm, y_cm)
                    client_config.ROBOT_HEADING = float(heading_deg)
                break

        # — YOLO inference —
        results = yolo_model(original, verbose=False)
        detections = results[0].boxes.data.tolist()

        ball_positions_cm.clear()
        new_obstacles = set()

        for x1, y1, x2, y2, conf, cls_id in detections:
            lbl = results[0].names[int(cls_id)].lower()
            cx_pix, cy_pix = int((x1 + x2) / 2), int((y1 + y2) / 2)
            is_robot = 'robot' in lbl

            if not is_robot:
                color = class_colors.setdefault(lbl, (
                    random.randint(0,255),
                    random.randint(0,255),
                    random.randint(0,255)
                ))
                cv2.rectangle(original, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(original, lbl, (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            real = navigation.pixel_to_cm(cx_pix, cy_pix)
            if real:
                gx, gy = navigation.cm_to_grid_coords(real[0], real[1])
                if 'ball' in lbl and not (
                    config.IGNORED_AREA['x_min'] <= real[0] <= config.IGNORED_AREA['x_max'] and
                    config.IGNORED_AREA['y_min'] <= real[1] <= config.IGNORED_AREA['y_max']
                ):
                    ball_positions_cm.append((real[0], real[1], lbl, cx_pix, cy_pix))
                elif is_robot and planner.robot_position_cm is None:
                    planner.robot_position_cm = real
                else:
                    new_obstacles.add((gx, gy))

        obstacles |= navigation.get_expanded_obstacles(new_obstacles)

        # — Draw grid and route —
        frame_grid  = navigation.draw_metric_grid(original)
        frame_route = navigation.draw_full_route(frame_grid, ball_positions_cm)

        # — Optional red‐cross obstacle detection (unchanged) —
        if client_config.homography_matrix is not None:
            hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
            mask1 = cv2.inRange(hsv, np.array([0,120,70]),  np.array([10,255,255]))
            mask2 = cv2.inRange(hsv, np.array([170,120,70]),np.array([180,255,255]))
            red_mask = cv2.bitwise_or(mask1, mask2)
            kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN,  kernel)
            contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

            best_cnt, best_area = None, 0
            px_per_x = original.shape[1] / client_config.REAL_WIDTH_CM
            px_per_y = original.shape[0] / client_config.REAL_HEIGHT_CM

            for cnt in contours:
                area_px = cv2.contourArea(cnt)
                if area_px < client_config.MIN_RED_AREA_PX:
                    continue
                x_r, y_r, w_r, h_r = cv2.boundingRect(cnt)
                area_cm = (w_r / px_per_x) * (h_r / px_per_y)
                if area_cm > client_config.MAX_RED_AREA_CM2:
                    continue
                if area_px > best_area:
                    best_cnt, best_area = cnt, area_px

            new_cross_obs = set()
            if best_cnt is not None:
                bx, by, bw_cnt, bh_cnt = cv2.boundingRect(best_cnt)
                for sx in range(bx, bx + bw_cnt, 10):
                    for sy in range(by, by + bh_cnt, 10):
                        if cv2.pointPolygonTest(best_cnt, (sx, sy), False) >= 0:
                            real = pixel_to_cm(sx, sy)
                            if not real:
                                continue
                            gx, gy = cm_to_grid_coords(real[0], real[1])
                            if (0 <= gx < client_config.REAL_WIDTH_CM // client_config.GRID_SPACING_CM and
                                0 <= gy < client_config.REAL_HEIGHT_CM // client_config.GRID_SPACING_CM):
                                new_cross_obs.add((gx, gy))
            obstacles |= new_cross_obs

        # — Push to display queue —
        try:
            output_queue.put((frame_route, capture_ts), timeout=0.02)
        except:
            pass

        # — Periodic pose send every 0.5s, but only if connected —
        now = time.time()
        if (
            planner.robot_position_cm is not None and
            robot_comm.robot_sock is not None and
            now - last_pose_send >= 0.5
        ):
            x_cm, y_cm = planner.robot_position_cm
            try:
                robot_comm.send_pose(x_cm, y_cm, client_config.ROBOT_HEADING, timestamp=capture_ts)
            except Exception as e:
                print(f"🔴 send_pose exception: {e}")
            last_pose_send = now

    print("process_frames exiting")
