import socket
import json
import time

# Connection settings
EV3_IP = "172.20.10.6"  # Update this to your EV3's IP address
EV3_PORT = 12345

def send_command(command: str, **params) -> bool:
    """Send a command to the EV3 server"""
    try:
        with socket.create_connection((EV3_IP, EV3_PORT), timeout=5) as s:
            # Prepare command
            message = {
                "command": command,
                **params
            }
            
            # Send command
            s.sendall((json.dumps(message) + "\n").encode("utf-8"))
            
            # Get response
            response = s.recv(1024).decode("utf-8").strip()
            return response == "OK"
            
    except Exception as e:
        print("Error sending command: {}".format(e))
        return False

def get_status():
    """Get current robot status"""
    try:
        with socket.create_connection((EV3_IP, EV3_PORT), timeout=5) as s:
            message = json.dumps({"command": "STATUS"}) + "\n"
            s.sendall(message.encode("utf-8"))
            response = s.recv(1024).decode("utf-8").strip()
            return json.loads(response)
    except Exception as e:
        print("Error getting status: {}".format(e))
        return {}

def drift_test():
    print("\nDrift Calibration Test")
    print("---------------------")
    print("This test will help identify and fix leftward drift.")
    print("\nSetup instructions:")
    print("1. Place the robot on a long straight line (like a piece of tape)")
    print("2. Align the CENTER of both drive wheels on this line")
    print("3. Make sure you have 2 meters of clear space ahead")
    print("4. Press Enter when ready")
    input()
    
    # First, let's check current motor speeds
    status = get_status()
    if status:
        print("\nCurrent motor configuration:")
        print("Left motor speed: {}".format(status.get("left_motor", {}).get("speed", "Unknown")))
        print("Right motor speed: {}".format(status.get("right_motor", {}).get("speed", "Unknown")))
    
    # Move forward 200cm (longer distance makes drift more apparent)
    print("\nMoving forward 200cm...")
    success = send_command("MOVE", distance=200)
    
    if success:
        print("\nMovement completed!")
        print("\nMeasurement instructions:")
        print("1. Measure how far the robot drifted from the straight line")
        print("2. Measure at the CENTER point between the drive wheels")
        print("3. If the robot drifted left, the value is positive")
        print("4. If the robot drifted right, the value is negative")
        
        drift_cm = float(input("\nEnter the drift distance (in cm, positive for left drift): "))
        
        # Calculate suggested adjustment
        # For every 1cm of drift over 200cm, we suggest a 0.5% motor speed adjustment
        adjustment_percent = abs(drift_cm) * 0.5
        
        print("\nResults:")
        print("Drift distance: {:.1f} cm".format(drift_cm))
        
        if abs(drift_cm) > 1:  # Only suggest changes if drift is significant
            print("\nRecommendation:")
            if drift_cm > 0:  # Left drift
                print("The robot drifts left, suggesting the right motor is slightly stronger.")
                print("\nIn tcp_server.py, find the motor setup section and add:")
                print("# Adjust left/right motor balance")
                print("mdiff.left_motor.duty_cycle_sp = {}  # Increase left motor power".format(int(100 + adjustment_percent)))
                print("mdiff.right_motor.duty_cycle_sp = 100  # Keep right motor at 100%")
            else:  # Right drift
                print("The robot drifts right, suggesting the left motor is slightly stronger.")
                print("\nIn tcp_server.py, find the motor setup section and add:")
                print("# Adjust left/right motor balance")
                print("mdiff.left_motor.duty_cycle_sp = 100  # Keep left motor at 100%")
                print("mdiff.right_motor.duty_cycle_sp = {}  # Increase right motor power".format(int(100 + adjustment_percent)))
        else:
            print("\nDrift is within acceptable range (±1cm). No adjustment needed.")
    else:
        print("Error: Movement command failed!")

def change_ip():
    """Change the EV3 IP address"""
    global EV3_IP
    EV3_IP = input("Enter new EV3 IP address: ")

def main():
    print("EV3 Drift Calibration Tool")
    print("=========================")
    print("This tool will help calibrate the robot's straight-line movement.")
    print("\nCurrent EV3 IP: {}".format(EV3_IP))
    print("Current EV3 Port: {}".format(EV3_PORT))
    
    while True:
        print("\nMenu:")
        print("1. Run drift test")
        print("2. Change EV3 IP address")
        print("3. Exit")
        
        choice = input("\nEnter choice (1-3): ")
        
        if choice == "1":
            drift_test()
        elif choice == "2":
            change_ip()
        elif choice == "3":
            break
        else:
            print("Invalid choice!")

if __name__ == "__main__":
    main() 