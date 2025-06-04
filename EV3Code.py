#!/usr/bin/env python3
"""
EV3dev TCP Server Script incorporating Pybricks-style drive_and_collect

Listens on port 12345 for a JSON‐encoded payload containing:
    {
      "path": [[gx1, gy1], [gx2, gy2], …],
      "heading": "N"  # one of "N", "E", "S", "W"
    }
For each consecutive pair of grid cells, this script:
    1. Turns the robot to face the next cell
    2. Drives forward to that cell
    3. Runs the collector motor for 0.8 s
    4. Backs up 30 cm
After finishing the entire path, it responds with "DONE\n" and closes the connection.
"""

import socket
import threading
import json
import math
import time

from ev3dev2.motor import LargeMotor, MediumMotor, OUTPUT_B, OUTPUT_C, OUTPUT_A, SpeedPercent
from ev3dev2.robotics import DriveBase
from ev3dev2.display import Display

# ───────── Hardware Setup ─────────
# Ports: B = left wheel, C = right wheel, A = collector
left_motor      = LargeMotor(OUTPUT_B)
right_motor     = LargeMotor(OUTPUT_C)
collector_motor = MediumMotor(OUTPUT_A)

# Create a DriveBase so we can call .turn() and .straight()
WHEEL_DIAM_CM = 5.55   # 55.5 mm diameter
AXLE_TRACK_CM = 10.4   # 104 mm axle track
robot_drive = DriveBase(left_motor, right_motor, wheel_diameter=WHEEL_DIAM_CM, axle_track=AXLE_TRACK_CM)

# EV3 screen (for on-screen feedback)
display = Display()

# Grid spacing in cm (must match your PC’s grid)
GRID_SPACING_CM = 2

# Global heading (degrees from +X, CCW positive)
robot_heading_deg = 0.0

# ───────── Helper Functions ─────────

def grid_to_cm(gx, gy):
    """
    Convert grid coordinates (gx, gy) to the real-world center (x_cm, y_cm).
    Each cell is GRID_SPACING_CM × GRID_SPACING_CM, so center is offset by half a cell.
    """
    x_cm = gx * GRID_SPACING_CM + GRID_SPACING_CM / 2.0
    y_cm = gy * GRID_SPACING_CM + GRID_SPACING_CM / 2.0
    return x_cm, y_cm

def compute_turn_angle(dx, dy, current_heading):
    """
    Given a vector (dx, dy) in cm, and the robot’s current heading (degrees),
    return the smallest signed angular change (in degrees) needed so the robot
    faces the vector. 0° = +X axis, positive is CCW.
    """
    theta_rad = math.atan2(dy, dx)
    theta_deg = math.degrees(theta_rad)
    delta = (theta_deg - current_heading + 180.0) % 360.0 - 180.0
    return delta

def drive_and_collect(turn_deg, dist_cm):
    """
    Turn the robot by turn_deg degrees, drive forward dist_cm cm,
    run the collector for 0.8 s, then back up 30 cm.
    Displays status on the EV3 screen.
    """
    # Display on EV3 screen
    display.clear()
    display.text_box(f"TURN {turn_deg:.1f}°", font='courB24', x=10, y=10)
    display.text_box(f"DIST {dist_cm:.1f}cm", font='courB24', x=10, y=50)
    display.update()

    # Turn in place
    robot_drive.turn(turn_deg)

    # Drive forward dist_cm (DriveBase expects cm directly)
    robot_drive.straight(dist_cm)

    # Run collector for 0.8 s (800 ms)
    collector_motor.on_for_seconds(speed=500, seconds=0.8)

    # Back up 30 cm
    robot_drive.straight(-30)

    # Clear screen when done
    display.clear()
    display.update()

# ───────── Client Handler ─────────

def handle_client(conn, addr):
    global robot_heading_deg
    print(f"[TCP Server] Client connected: {addr}")

    # Read exactly one JSON object (terminated by newline)
    data_buffer = ""
    while True:
        chunk = conn.recv(1024)
        if not chunk:
            print("[TCP Server] Client disconnected before sending data.")
            conn.close()
            return
        data_buffer += chunk.decode("utf-8")
        if "\n" in data_buffer:
            json_str, _ = data_buffer.split("\n", 1)
            break

    # Parse JSON
    try:
        request = json.loads(json_str)
        path    = request.get("path", [])
        heading = request.get("heading", "E")
    except Exception as e:
        print(f"[TCP Server] JSON parse error: {e}")
        conn.sendall(b"ERROR\n")
        conn.close()
        return

    # Map heading letter to degrees
    heading_map = {
        "E": 0.0,
        "N": 90.0,
        "W": 180.0,
        "S": -90.0,
    }
    robot_heading_deg = heading_map.get(heading.upper(), 0.0)
    print(f"[TCP Server] Received path: {path}, initial heading: {robot_heading_deg}°")

    # Iterate through consecutive grid cells
    for i in range(len(path) - 1):
        gx1, gy1 = path[i]
        gx2, gy2 = path[i + 1]
        x1_cm, y1_cm = grid_to_cm(gx1, gy1)
        x2_cm, y2_cm = grid_to_cm(gx2, gy2)

        dx = x2_cm - x1_cm
        dy = y2_cm - y1_cm

        # Compute turn angle and perform turn
        delta_deg = compute_turn_angle(dx, dy, robot_heading_deg)
        drive_and_collect(delta_deg, 0)  # turn first (distance=0)
        robot_heading_deg = (robot_heading_deg + delta_deg) % 360.0

        # Drive straight the required distance
        distance_cm = math.hypot(dx, dy)
        drive_and_collect(0, distance_cm)  # drive forward, then collect & back

    # Entire path completed
    conn.sendall(b"DONE\n")
    print("[TCP Server] Path execution complete; sent DONE.")
    conn.close()
    print(f"[TCP Server] Client {addr} disconnected.")

# ───────── Main Server Loop ─────────

def main():
    HOST = ""        # Listen on all interfaces
    PORT = 12345     # Must match the PC’s send_path port

    print(f"[TCP Server] Starting on port {PORT} …")
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.bind((HOST, PORT))
    server_sock.listen(1)
    print(f"[TCP Server] Listening for incoming connections on port {PORT}…")

    try:
        while True:
            conn, addr = server_sock.accept()
            client_thread = threading.Thread(
                target=handle_client,
                args=(conn, addr),
                daemon=True
            )
            client_thread.start()
    except KeyboardInterrupt:
        print("\n[TCP Server] Shutting down…")
    finally:
        server_sock.close()

if __name__ == "__main__":
    main()
