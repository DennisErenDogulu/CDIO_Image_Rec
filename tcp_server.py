#!/usr/bin/env python3
"""
tcp_server.py  –  EV3 brick side
────────────────────────────────
• Listens for one-line JSON messages terminated by '\n'.
• Two payload shapes:
    1)  {"command":"MOVE", "distance":100} etc.   ← legacy single command
    2)  {"path":[{"x":10,"y":10},{"x":10,"y":5,"rev":true},…],
         "ball_cells":[[10,5]], "heading":"E"}    ← new full path

In case (2) the server drives the full waypoint list and then replies "DONE\n".
In all other cases it executes the legacy command and replies "OK\n" or
returns a STATUS json.
"""

import socket, threading, json, math, time
from ev3dev2.motor import (
    OUTPUT_B, OUTPUT_C, OUTPUT_A, MediumMotor,
    MoveDifferential, SpeedRPM
)
from ev3dev2.wheel import Wheel
from ev3dev2.display import Display
from ev3dev2.power import PowerSupply

# ───────── Robot geometry ─────────
class MyWheel(Wheel):
    def __init__(self):
        super().__init__(diameter_mm=69, width_mm=35)   # 69 mm wheels

AXLE_TRACK_MM = 40                                      # wheel-to-wheel
mdiff  = MoveDifferential(OUTPUT_B, OUTPUT_C, MyWheel, AXLE_TRACK_MM)
collector = MediumMotor(OUTPUT_A)

display = Display()
power   = PowerSupply()

# ───────── Movement parameters ─────────
NORMAL_SPEED_RPM = -40          # negative → forward on this robot
SLOW_SPEED_RPM   = -25
TURN_SPEED_RPM   = -40
COLLECTOR_SPEED  = 25

ACCELERATION_MS  = 200
REVERSE_ACCEL_FACTOR = 1.5

CM_TO_MM   = 11.1               # 1 cm ≅ 11.1 mm (calibration)
TURN_FACTOR = (math.pi * AXLE_TRACK_MM) / 360.0 * 1.25   # with compensation

# ───────── Safe wrappers ─────────

def safe_motor_control(func):
    def wrapper(*args, **kw):
        try:
            result = func(*args, **kw)
            mdiff.off(brake=True)
            return result
        except Exception as e:
            print("[Motor] {}: {}".format(func.__name__, e))
            mdiff.off(brake=True)
            collector.off(brake=True)
            return False
    return wrapper

# ───────── Primitive executors ─────────
@safe_motor_control
def execute_move(distance_cm: float):
    """Positive distance_cm = drive forward; negative = drive backward."""
    is_reverse = distance_cm < 0
    accel = ACCELERATION_MS * (REVERSE_ACCEL_FACTOR if is_reverse else 1)
    mdiff.ramp_up_sp = accel
    mdiff.ramp_down_sp = accel

    speed = NORMAL_SPEED_RPM if distance_cm > 0 else -NORMAL_SPEED_RPM
    mdiff.on_for_distance(
        SpeedRPM(speed),
        abs(distance_cm) * CM_TO_MM
    )
    time.sleep(0.1)
    return True

@safe_motor_control
def execute_turn(angle_deg: float):
    mdiff.ramp_up_sp = mdiff.ramp_down_sp = ACCELERATION_MS * 2
    speed = TURN_SPEED_RPM if angle_deg > 0 else -TURN_SPEED_RPM
    mdiff.turn_degrees(SpeedRPM(speed), abs(angle_deg))
    time.sleep(0.2)
    return True

@safe_motor_control
def execute_collect(distance_cm: float):
    for s in range(0, COLLECTOR_SPEED+1, 5):
        collector.on(s)
        time.sleep(0.05)
    mdiff.ramp_up_sp = mdiff.ramp_down_sp = int(ACCELERATION_MS*1.5)
    mdiff.on_for_distance(SpeedRPM(SLOW_SPEED_RPM), distance_cm*CM_TO_MM)
    for s in range(COLLECTOR_SPEED, -1, -5):
        collector.on(s)
        time.sleep(0.05)
    collector.off(brake=True)
    return True

@safe_motor_control
def execute_collect_reverse(duration: float):
    for s in range(0, -COLLECTOR_SPEED-1, -5):
        collector.on(s)
        time.sleep(0.05)
    time.sleep(duration)
    for s in range(-COLLECTOR_SPEED, 1, 5):
        collector.on(s)
        time.sleep(0.05)
    collector.off(brake=True)
    return True

def execute_stop():
    mdiff.off(brake=True)
    collector.off(brake=True)
    return True

# ───────── STATUS helper ─────────

def get_status():
    try:
        return json.dumps({
            "battery_voltage": power.measured_voltage,
            "battery_current": power.measured_current,
            "battery_percentage": power.measured_voltage / power.max_voltage * 100,
            "left_motor": {
                "position": mdiff.left_motor.position,
                "speed":    mdiff.left_motor.speed
            },
            "right_motor": {
                "position": mdiff.right_motor.position,
                "speed":    mdiff.right_motor.speed
            },
            "collector_motor": {
                "position": collector.position,
                "speed":    collector.speed
            }
        })
    except Exception as e:
        return json.dumps({"error": str(e)})

# ───────── ★ NEW : waypoint executor ─────────

def drive_path(path):
    """
    Drive the list of waypoints.
    Each element: {"x":<cm>, "y":<cm>, "rev":<bool>}
    """
    cur_x = cur_y = 0.0      # you could feed start pos from client if needed

    for wp in path:
        tgt_x, tgt_y = wp["x"], wp["y"]
        rev = wp.get("rev", False)

        dx, dy   = tgt_x - cur_x, tgt_y - cur_y
        dist_cm  = math.hypot(dx, dy)
        angle_deg = math.degrees(math.atan2(dy, dx))

        # Face the direction we will actually drive
        if rev:
            angle_deg = (angle_deg + 180) % 360
            dist_cm   = -dist_cm          # sign flip → backward move

        if not execute_turn(angle_deg):
            return False
        if not execute_move(dist_cm):
            return False

        cur_x, cur_y = tgt_x, tgt_y
    return True

# ───────── Connection handler ─────────

def handle_client(conn, addr):
    print("[TCP] Client {} connected".format(addr))
    try:
        conn.settimeout(5.0)
        data_buffer = ""
        while True:
            chunk = conn.recv(1024)
            if not chunk:
                print("[TCP] Client {} disconnected before data.".format(addr))
                return
            data_buffer += chunk.decode()
            if "\n" in data_buffer:
                json_str, _ = data_buffer.split("\n", 1)
                break

        # ---------- Parse ----------
        try:
            payload = json.loads(json_str)
        except json.JSONDecodeError:
            conn.sendall(b"ERROR\n")
            return

        # ---------- 1) full-path payload ----------
        if "path" in payload:
            success = drive_path(payload["path"])
            conn.sendall(b"DONE\n" if success else b"ERROR\n")
            return

        # ---------- 2) legacy command ----------
        cmd = payload.get("command", "").upper()
        ok = False

        if cmd == "MOVE":
            ok = execute_move(float(payload["distance"]))
        elif cmd == "TURN":
            ok = execute_turn(float(payload["angle"]))
        elif cmd == "COLLECT":
            ok = execute_collect(float(payload["distance"]))
        elif cmd == "COLLECT_REVERSE":
            ok = execute_collect_reverse(float(payload["duration"]))
        elif cmd == "STOP":
            ok = execute_stop()
        elif cmd == "STATUS":
            conn.sendall(get_status().encode() + b"\n")
            return
        else:
            print("[TCP] Unknown command {}".format(cmd))

        conn.sendall(b"OK\n" if ok else b"ERROR\n")

    except socket.timeout:
        print("[TCP] Timeout with {}".format(addr))
        conn.sendall(b"ERROR\n")
    except Exception as e:
        print("[TCP] {} error: {}".format(addr, e))
        conn.sendall(b"ERROR\n")
    finally:
        conn.close()

# ───────── Main server loop ─────────

def main():
    HOST, PORT = "", 12345
    print("[TCP] Server on port {}".format(PORT))
    print("[TCP] Battery: {:.1f} V ({:.0f} %)".format(
          power.measured_voltage,
          power.measured_voltage / power.max_voltage * 100))

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(1)

    try:
        while True:
            conn, addr = server.accept()
            threading.Thread(
                target=handle_client,
                args=(conn, addr),
                daemon=True
            ).start()
    except KeyboardInterrupt:
        print("\n[TCP] Shutting down …")
    finally:
        server.close()
        execute_stop()

if __name__ == "__main__":
    main()
