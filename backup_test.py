#!/usr/bin/env python3
"""
backup_self_test.py
--------------------------------
Quick integration-test for the EV3 “back-up” logic.

Sequence:
  1. Query STATUS  ➜ grab initial wheel-encoder positions
  2. MOVE +d_cm    ➜ robot should drive forward (negative motor RPM)
  3. MOVE -d_cm    ➜ robot should drive backward (positive motor RPM)
  4. Query STATUS  ➜ grab final encoder positions
  5. Compare net travel; PASS if |net|  < tolerance_mm

Run it from your laptop/PC while the EV3 server is running.

Safety: clear a 1 m straight path before starting.

Author: you 😉
"""

import json
import math
import socket
import time

EV3_IP   = "172.20.10.6"
EV3_PORT = 12345
TEST_DISTANCE_CM = 30         # how far to drive out & back
TOLERANCE_MM     = 50         # max permissible net drift
WHEEL_DIAM_MM    = 69         # keep in sync with ev3 script!

# ───────────────── helpers ──────────────────────────────────────────
def send_cmd(sock, command, **params):
    """Send a single JSON command terminated by \n and return raw reply."""
    msg = {"command": command, **params}
    sock.sendall((json.dumps(msg) + "\n").encode())
    return sock.recv(4096).decode().strip()


def encoders_to_mm(delta_deg):
    """Convert degrees-of-rotation to linear distance (per wheel)."""
    return delta_deg * (math.pi * WHEEL_DIAM_MM) / 360.0


def get_encoders(status_json):
    data = json.loads(status_json)
    left  = data["left_motor"]["position"]
    right = data["right_motor"]["position"]
    return left, right


# ───────────────── test logic ───────────────────────────────────────
def main():
    print(f"Connecting to EV3 at {EV3_IP}:{EV3_PORT} …")
    with socket.create_connection((EV3_IP, EV3_PORT), timeout=10) as s:
        # 1️⃣ baseline encoders
        left0, right0 = get_encoders(send_cmd(s, "STATUS"))
        print(f"Start encoders: L={left0}°, R={right0}°")

        # 2️⃣ forward +d
        assert send_cmd(s, "MOVE", distance=+TEST_DISTANCE_CM) == "OK"
        time.sleep(0.2)                          # tiny pause

        # 3️⃣ backward -d   (positive RPM on EV3 because distance<0)
        assert send_cmd(s, "MOVE", distance=-TEST_DISTANCE_CM) == "OK"

        # 4️⃣ final encoders
        left1, right1 = get_encoders(send_cmd(s, "STATUS"))
        print(f" End  encoders: L={left1}°, R={right1}°")

    # 5️⃣ analyse
    dL_mm = encoders_to_mm(left1 - left0)
    dR_mm = encoders_to_mm(right1 - right0)
    drift_mm = abs((dL_mm + dR_mm) / 2)          # average both wheels

    print(f"\nNet wheel travel ≈ {drift_mm:.1f} mm "
          f"(tolerance {TOLERANCE_MM} mm)")
    if drift_mm <= TOLERANCE_MM:
        print("✅  PASS — sign convention looks good")
    else:
        print("❌  FAIL — robot did not return to start; "
              "check MOVE-sign logic on both sides")


if __name__ == "__main__":
    main()
