#!/usr/bin/env python3
import socket
import threading
import json
import math
import time
from ev3dev2.motor import OUTPUT_B, OUTPUT_C, OUTPUT_A, MediumMotor
from ev3dev2.motor import MoveDifferential, SpeedRPM
from ev3dev2.wheel import Wheel
from ev3dev2.display import Display
from ev3dev2.power import PowerSupply

# ───────── Hardware Setup ─────────
AXLE_TRACK_MM = 160    # Axle distance in mm
WHEEL_DIAM_MM = 80     # Wheel diameter in mm
WHEEL_WIDTH_MM = 16    # Wheel width in mm

class MyWheel(Wheel):
    def __init__(self):
        super().__init__(diameter_mm=WHEEL_DIAM_MM,
                        width_mm=WHEEL_WIDTH_MM)

# MoveDifferential expects a "wheel_class"
mdiff = MoveDifferential(
    OUTPUT_B,
    OUTPUT_C,
    MyWheel,
    AXLE_TRACK_MM
)

collector = MediumMotor(OUTPUT_A)   # Collector motor
display = Display()                 # EV3 display
power = PowerSupply()               # For battery monitoring

# Movement speeds
NORMAL_SPEED_RPM = 40
SLOW_SPEED_RPM = 20
COLLECTOR_SPEED = 50  # Percentage

# Acceleration control (ramp up/down)
ACCELERATION_TIME_MS = 200  # Time to reach full speed

def safe_motor_control(func):
    """Decorator for safe motor control with error handling"""
    def wrapper(*args, **kwargs):
        try:
            # Set acceleration time
            mdiff.ramp_up_sp = ACCELERATION_TIME_MS
            mdiff.ramp_down_sp = ACCELERATION_TIME_MS
            
            # Execute motor command
            result = func(*args, **kwargs)
            
            # Ensure motors are stopped
            mdiff.off(brake=True)
            return result
        except Exception as e:
            print(f"Motor control error in {func.__name__}: {e}")
            # Emergency stop
            mdiff.off(brake=True)
            collector.off(brake=True)
            return False
    return wrapper

@safe_motor_control
def execute_move(distance_cm):
    """Move forward/backward by distance_cm (negative = backward)"""
    mdiff.on_for_distance(SpeedRPM(NORMAL_SPEED_RPM), distance_cm * 10)  # cm to mm
    return True

@safe_motor_control
def execute_turn(angle_deg):
    """Turn by angle_deg (positive = CCW, negative = CW)"""
    mdiff.turn_degrees(SpeedRPM(NORMAL_SPEED_RPM), angle_deg)
    return True

@safe_motor_control
def execute_collect(distance_cm):
    """Move forward slowly while running collector"""
    try:
        # Start collector motor with gradual speed increase
        for speed in range(0, COLLECTOR_SPEED + 1, 5):
            collector.on(speed)
            time.sleep(0.05)
        
        # Move forward slowly
        mdiff.on_for_distance(SpeedRPM(SLOW_SPEED_RPM), distance_cm * 10)
        
        # Gradually stop collector
        for speed in range(COLLECTOR_SPEED, -1, -5):
            collector.on(speed)
            time.sleep(0.05)
        
        collector.off(brake=True)
        return True
    except Exception as e:
        print(f"Collection error: {e}")
        collector.off(brake=True)
        return False

def execute_stop():
    """Stop all motors"""
    mdiff.off(brake=True)
    collector.off(brake=True)
    return True

def get_status():
    """Get current robot status"""
    try:
        status = {
            "battery_voltage": power.measured_voltage,
            "battery_current": power.measured_current,
            "battery_percentage": power.measured_voltage / power.max_voltage * 100,
            "left_motor": {
                "position": mdiff.left_motor.position,
                "speed": mdiff.left_motor.speed
            },
            "right_motor": {
                "position": mdiff.right_motor.position,
                "speed": mdiff.right_motor.speed
            },
            "collector_motor": {
                "position": collector.position,
                "speed": collector.speed
            }
        }
        return json.dumps(status)
    except Exception as e:
        print(f"Status error: {e}")
        return json.dumps({"error": str(e)})

def handle_client(conn, addr):
    """
    Runs in a separate thread for each new connection.
    Receives exactly one \n-terminated JSON line and processes it
    as a movement command.
    """
    print(f"[TCP Server] Client connected: {addr}")
    
    try:
        # Set timeout to prevent hanging
        conn.settimeout(5.0)
        
        # Read one line (ends with '\n')
        data_buffer = ""
        while True:
            chunk = conn.recv(1024)
            if not chunk:
                print(f"[TCP Server] Client {addr} disconnected before data.")
                return
            
            data_buffer += chunk.decode("utf-8")
            if "\n" in data_buffer:
                json_str, _ = data_buffer.split("\n", 1)
                break

        # Parse and execute command
        request = json.loads(json_str)
        command = request.get("command", "").upper()
        
        result = False
        response = "ERROR\n"
        
        if command == "MOVE" and "distance" in request:
            result = execute_move(float(request["distance"]))
            
        elif command == "TURN" and "angle" in request:
            result = execute_turn(float(request["angle"]))
            
        elif command == "COLLECT" and "distance" in request:
            result = execute_collect(float(request["distance"]))
            
        elif command == "STOP":
            result = execute_stop()
            
        elif command == "STATUS":
            status = get_status()
            conn.sendall(status.encode() + b"\n")
            return
            
        else:
            print("[TCP Server] Invalid command format")
        
        # Send response based on command execution
        response = "OK\n" if result else "ERROR\n"
        conn.sendall(response.encode())

    except socket.timeout:
        print(f"[TCP Server] Connection to {addr} timed out")
        conn.sendall(b"ERROR\n")
    
    except json.JSONDecodeError:
        print("[TCP Server] Invalid JSON format")
        conn.sendall(b"ERROR\n")
    
    except Exception as e:
        print(f"[TCP Server] Error processing command: {e}")
        conn.sendall(b"ERROR\n")
    
    finally:
        conn.close()

def main():
    HOST = ""      # Listen on all interfaces
    PORT = 12345   # Must match PC client's port

    print(f"[TCP Server] Starting on port {PORT} …")
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((HOST, PORT))
    server_sock.listen(1)
    
    # Show initial battery status
    voltage = power.measured_voltage
    percentage = voltage / power.max_voltage * 100
    print(f"[TCP Server] Battery: {voltage:.1f}V ({percentage:.0f}%)")
    
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
        execute_stop()  # Ensure motors are stopped

if __name__ == "__main__":
    main()
