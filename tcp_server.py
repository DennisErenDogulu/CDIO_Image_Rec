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
AXLE_TRACK_MM = 40    # Axle distance in mm
WHEEL_DIAM_MM = 69     # Wheel diameter in mm
WHEEL_WIDTH_MM = 35    # Wheel width in mm

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
NORMAL_SPEED_RPM = -40  # Negative for forward movement
REVERSE_SPEED_RPM = 40  # Positive for backward movement
SLOW_SPEED_RPM = -25   # Negative for forward movement
SLOW_REVERSE_SPEED_RPM = 25  # Positive for backward movement
TURN_SPEED_RPM = -40   # For turning
COLLECTOR_SPEED = 25  # Percentage

# Acceleration control (ramp up/down)
ACCELERATION_TIME_MS = 200  # Time to reach full speed
REVERSE_ACCEL_FACTOR = 1.5  # Multiplier for reverse movement acceleration

# Robot physical dimensions (in mm)
WHEEL_DIAMETER_MM = 69  # 6.9 cm
WHEEL_WIDTH_MM = 35    # 3.5
AXLE_TRACK_MM = 450   # 45 cm

# Calculate turn factor
# Since motors are balanced, focusing on mechanical compensation
# Increasing compensation to account for ball caster friction
TURN_COMPENSATION = 1.25  # 25% compensation for mechanical factors
TURN_FACTOR = (math.pi * AXLE_TRACK_MM) / 360.0 * TURN_COMPENSATION

def safe_motor_control(func):
    """Decorator for safe motor control with error handling"""
    def wrapper(*args, **kwargs):
        try:
            # Execute motor command
            result = func(*args, **kwargs)
            
            # Ensure motors are stopped
            mdiff.off(brake=True)
            return result
        except Exception as e:
            print("Motor control error in {}: {}".format(func.__name__, e))
            # Emergency stop
            mdiff.off(brake=True)
            collector.off(brake=True)
            return False
    return wrapper

@safe_motor_control
def execute_move(distance_cm):
    """Move forward/backward by distance_cm (negative = backward)"""
    try:
        # Determine direction and set appropriate acceleration
        is_reverse = distance_cm < 0
        accel_time = ACCELERATION_TIME_MS * REVERSE_ACCEL_FACTOR if is_reverse else ACCELERATION_TIME_MS
        
        # Set acceleration parameters
        mdiff.ramp_up_sp = accel_time
        mdiff.ramp_down_sp = accel_time
        
        # Set speed based on direction
        # For forward (positive distance): use NORMAL_SPEED_RPM (negative)
        # For backward (negative distance): use REVERSE_SPEED_RPM (positive)
        speed = REVERSE_SPEED_RPM if is_reverse else NORMAL_SPEED_RPM
        
        # Convert distance to mm (always positive)
        distance_mm = abs(distance_cm) * 11.1  # adjusted cm to mm conversion
        
        # Complete stop before changing direction
        mdiff.off(brake=True)
        if is_reverse:
            time.sleep(0.2)  # Extra pause for reverse
        
        # Execute movement
        mdiff.on_for_distance(SpeedRPM(speed), distance_mm)
        
        # Brief pause after movement
        time.sleep(0.1)
        
        return True
    except Exception as e:
        print("Move error: {}".format(e))
        mdiff.off(brake=True)
        return False

@safe_motor_control
def execute_turn(angle_deg):
    """Turn by angle_deg (positive = CCW, negative = CW)"""
    try:
        # Set even slower speed and longer acceleration for turns
        mdiff.ramp_up_sp = ACCELERATION_TIME_MS * 2    # Double the acceleration time
        mdiff.ramp_down_sp = ACCELERATION_TIME_MS * 2  # Double the deceleration time
        
        # Complete stop before turning
        mdiff.off(brake=True)
        time.sleep(0.2)
        
        # Execute turn (using negative speed for forward movement)
        mdiff.turn_degrees(SpeedRPM(TURN_SPEED_RPM if angle_deg > 0 else -TURN_SPEED_RPM), abs(angle_deg))
        
        # Pause after turn
        time.sleep(0.2)
        
        return True
    except Exception as e:
        print("Turn error: {}".format(e))
        mdiff.off(brake=True)
        return False

@safe_motor_control
def execute_collect(distance_cm):
    """Move forward slowly while running collector"""
    try:
        # Start collector motor with gradual speed increase
        for speed in range(0, COLLECTOR_SPEED + 1, 5):
            collector.on(speed)
            time.sleep(0.05)
        
        # Set slower acceleration for collection
        mdiff.ramp_up_sp = ACCELERATION_TIME_MS * 1.5
        mdiff.ramp_down_sp = ACCELERATION_TIME_MS * 1.5
        
        # Move forward slowly (negative speed for forward)
        mdiff.on_for_distance(SpeedRPM(SLOW_SPEED_RPM), distance_cm * 11.1)
        
        # Gradually stop collector
        for speed in range(COLLECTOR_SPEED, -1, -5):
            collector.on(speed)
            time.sleep(0.05)
        
        collector.off(brake=True)
        return True
    except Exception as e:
        print("Collection error: {}".format(e))
        collector.off(brake=True)
        return False

@safe_motor_control
def execute_collect_reverse(duration):
    """Run collector in reverse to deliver balls"""
    try:
        # Start collector motor in reverse with gradual speed increase
        for speed in range(0, -COLLECTOR_SPEED - 1, -5):
            collector.on(speed)
            time.sleep(0.05)
        
        # Run for specified duration
        time.sleep(duration)
        
        # Gradually stop collector
        for speed in range(-COLLECTOR_SPEED, 1, 5):
            collector.on(speed)
            time.sleep(0.05)
        
        collector.off(brake=True)
        return True
    except Exception as e:
        print("Reverse collection error: {}".format(e))
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
        print("Status error: {}".format(e))
        return json.dumps({"error": str(e)})

def handle_client(conn, addr):
    """
    Runs in a separate thread for each new connection.
    Receives exactly one \n-terminated JSON line and processes it
    as a movement command.
    """
    print("[TCP Server] Client connected: {}".format(addr))
    
    try:
        # Set timeout to prevent hanging
        conn.settimeout(5.0)
        
        # Read one line (ends with '\n')
        data_buffer = ""
        while True:
            chunk = conn.recv(1024)
            if not chunk:
                print("[TCP Server] Client {} disconnected before data.".format(addr))
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
            
        elif command == "COLLECT_REVERSE" and "duration" in request:
            result = execute_collect_reverse(float(request["duration"]))
            
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
        print("[TCP Server] Connection to {} timed out".format(addr))
        conn.sendall(b"ERROR\n")
    
    except json.JSONDecodeError:
        print("[TCP Server] Invalid JSON format")
        conn.sendall(b"ERROR\n")
    
    except Exception as e:
        print("[TCP Server] Error processing command: {}".format(e))
        conn.sendall(b"ERROR\n")
    
    finally:
        conn.close()

def main():
    HOST = ""      # Listen on all interfaces
    PORT = 12345   # Must match PC client's port

    print("[TCP Server] Starting on port {} …".format(PORT))
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((HOST, PORT))
    server_sock.listen(1)
    
    # Show initial battery status
    voltage = power.measured_voltage
    percentage = voltage / power.max_voltage * 100
    print("[TCP Server] Battery: {:.1f}V ({:.0f}%)".format(voltage, percentage))
    
    print("[TCP Server] Listening for incoming connections on port {}…".format(PORT))

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