#!/usr/bin/env python3
"""
EV3 Movement & Backup Calibration Tool
======================================

Menu
  1. Forward-distance calibration     (measure 100 cm accuracy)
  2. Backup self-test                 (forward +d, back –d, check drift)
  3. Change EV3 IP address
  4. Exit
"""

import socket
import json
import time
import math

# ───────── Connection settings ─────────
EV3_IP   = "172.20.10.6"     # Change in menu if needed
EV3_PORT = 12345

# ───────── Robot geometry (must match tcp_server.py) ─────────
WHEEL_DIAM_MM = 69           # Wheel diameter
TEST_DISTANCE_CM = 30        # Distance for backup test
DRIFT_TOL_MM    = 50         # Acceptable net drift

# ───────── Helper functions ─────────
def _send_raw(command: str, **params):
    """
    Low-level helper: send one JSON command terminated by '\n'
    and return raw text reply.
    """
    msg = {"command": command, **params}
    with socket.create_connection((EV3_IP, EV3_PORT), timeout=5) as s:
        s.sendall((json.dumps(msg) + "\n").encode())
        return s.recv(4096).decode().strip()

def send_command(command: str, **params) -> bool:
    """Send command, expect 'OK'/'ERROR'."""
    try:
        return _send_raw(command, **params) == "OK"
    except Exception as e:
        print(f"Error sending command: {e}")
        return False

def query_status():
    """Return parsed STATUS dict or None."""
    try:
        return json.loads(_send_raw("STATUS"))
    except Exception as e:
        print(f"STATUS error: {e}")
        return None

def enc_to_mm(delta_deg):
    """Degrees to linear mm (per wheel)."""
    return delta_deg * (math.pi * WHEEL_DIAM_MM) / 360.0

# ───────── 1) Forward-distance calibration ─────────
def forward_calibration():
    print("\nForward-distance Calibration (100 cm)")
    print("-------------------------------------")
    print("• Place the robot on a line.")
    print("• Mark the CENTER of the drive wheels.")
    input("Press Enter to start …")

    print("▶ Moving forward 100 cm")
    ok = send_command("MOVE", distance=100)   # +100 → forward
    if not ok:
        print("✖ MOVE command failed.")
        return

    print("\nMovement done!")
    actual = float(input("Enter the actual distance travelled (cm): "))
    factor = 10 * (100 / actual)              # existing 10 × adjustment
    print(f"\nSuggested new conversion factor: {factor:.1f}")
    if abs(actual - 100) > 5:
        print("\nUpdate in tcp_server.py:")
        print(f"    mdiff.on_for_distance(..., distance_cm * {factor:.1f})")

# ───────── 2) Backup self-test ─────────
def backup_self_test():
    print("\nBackup Self-Test")
    print("----------------")
    print(f"Robot will drive +{TEST_DISTANCE_CM} cm forward, then −{TEST_DISTANCE_CM} cm back.")
    print("Make sure ~1 m of clear space.")
    input("Press Enter to start …")

    # 1️⃣ baseline encoders
    s0 = query_status()
    if not s0:
        print("✖ Could not read STATUS.")
        return
    l0, r0 = s0["left_motor"]["position"], s0["right_motor"]["position"]

    # 2️⃣ forward
    if not send_command("MOVE", distance=+TEST_DISTANCE_CM):
        print("✖ Forward MOVE failed."); return
    time.sleep(0.2)

    # 3️⃣ backward
    if not send_command("MOVE", distance=-TEST_DISTANCE_CM):
        print("✖ Reverse MOVE failed."); return

    # 4️⃣ final encoders
    s1 = query_status()
    if not s1:
        print("✖ Could not read STATUS (final)."); return
    l1, r1 = s1["left_motor"]["position"], s1["right_motor"]["position"]

    # 5️⃣ analyse drift
    drift_mm = abs((enc_to_mm(l1 - l0) + enc_to_mm(r1 - r0)) / 2)
    print(f"\nNet wheel travel ≈ {drift_mm:.1f} mm  (tolerance {DRIFT_TOL_MM} mm)")
    if drift_mm <= DRIFT_TOL_MM:
        print("✅ PASS — backup sign convention is correct")
    else:
        print("❌ FAIL — robot did not return to start")
        print("   Check sign logic in MOVE on both PC & EV3 sides.")

# ───────── Misc ─────────
def change_ip():
    global EV3_IP
    EV3_IP = input("New EV3 IP: ")

# ───────── Main menu ─────────
def main():
    while True:
        print("\nEV3 Calibration Tool")
        print("====================")
        print(f"Current EV3: {EV3_IP}:{EV3_PORT}")
        print("1. Forward-distance calibration")
        print("2. Backup self-test")
        print("3. Change EV3 IP address")
        print("4. Exit")
        choice = input("Choice (1-4): ").strip()

        if choice == "1":
            forward_calibration()
        elif choice == "2":
            backup_self_test()
        elif choice == "3":
            change_ip()
        elif choice == "4":
            break
        else:
            print("Invalid choice!")

if __name__ == "__main__":
    main()
