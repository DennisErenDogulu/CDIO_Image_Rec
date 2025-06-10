import socket
import json
import time
from typing import Dict

# Connection settings
EV3_IP = "172.20.10.6"  # Update this to your EV3's IP address
EV3_PORT = 12345

def send_command(command: str, **params) -> bool:
    """Send a command to the EV3 server"""
    try:
        with socket.create_connection((EV3_IP, EV3_PORT), timeout=15) as s:
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

def get_status() -> Dict:
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

def print_motor_speeds():
    """Get and print current motor speeds"""
    status = get_status()
    if status:
        left_speed = status.get("left_motor", {}).get("speed", 0)
        right_speed = status.get("right_motor", {}).get("speed", 0)
        print("\rLeft: {:4d} RPM | Right: {:4d} RPM | Diff: {:+3d} RPM".format(
            left_speed, right_speed, left_speed - right_speed), end="")
    return status

def motor_test():
    print("\nMotor Speed Test")
    print("--------------")
    print("\nTest options:")
    print("1. Forward movement test")
    print("2. Rotation test")
    print("3. Back to main menu")
    
    choice = input("\nEnter choice (1-3): ")
    
    if choice == "1":
        print("\nForward Movement Test")
        print("This test will move the robot forward and monitor motor speeds.")
        print("\nSetup instructions:")
        print("1. Place the robot on a flat surface")
        print("2. Ensure at least 1 meter of clear space ahead")
        print("3. Press Enter when ready (press Ctrl+C to stop)")
        input()
        
        print("\nMoving forward... Press Ctrl+C to stop")
        print("\nMonitoring motor speeds:")
        print("------------------------")
        
        # Start forward movement
        send_command("MOVE", distance=100)
        
        try:
            # Monitor speeds for 5 seconds
            start_time = time.time()
            while time.time() - start_time < 5:
                status = print_motor_speeds()
                time.sleep(0.1)
            
            # Stop motors
            send_command("STOP")
            
            # Final status
            print("\n\nTest completed!")
            if status:
                left_pos = status.get("left_motor", {}).get("position", 0)
                right_pos = status.get("right_motor", {}).get("position", 0)
                print("\nFinal motor positions:")
                print("Left motor: {} degrees".format(left_pos))
                print("Right motor: {} degrees".format(right_pos))
                print("Position difference: {} degrees".format(abs(left_pos - right_pos)))
                
                if abs(left_pos - right_pos) > 50:  # More than 50 degrees difference
                    print("\nSignificant position difference detected!")
                    print("Consider adjusting motor power in tcp_server.py:")
                    slower_motor = "left" if left_pos < right_pos else "right"
                    adjustment = abs(left_pos - right_pos) / max(abs(left_pos), abs(right_pos)) * 100
                    print("Increase {} motor power by approximately {:.1f}%".format(
                        slower_motor, adjustment))
        
        except KeyboardInterrupt:
            print("\n\nTest stopped by user")
            send_command("STOP")
    
    elif choice == "2":
        print("\nRotation Test")
        print("This test will rotate the robot and monitor motor speeds.")
        print("\nSetup instructions:")
        print("1. Place the robot on a flat surface")
        print("2. Ensure clear space around the robot")
        print("3. Press Enter when ready (press Ctrl+C to stop)")
        input()
        
        print("\nRotating... Press Ctrl+C to stop")
        print("\nMonitoring motor speeds:")
        print("------------------------")
        
        # Start rotation
        send_command("TURN", angle=360)
        
        try:
            # Monitor speeds for 5 seconds
            start_time = time.time()
            while time.time() - start_time < 5:
                print_motor_speeds()
                time.sleep(0.1)
            
            # Stop motors
            send_command("STOP")
            print("\n\nTest completed!")
            
        except KeyboardInterrupt:
            print("\n\nTest stopped by user")
            send_command("STOP")

def change_ip():
    """Change the EV3 IP address"""
    global EV3_IP
    EV3_IP = input("Enter new EV3 IP address: ")

def main():
    print("EV3 Motor Speed Test")
    print("===================")
    print("This tool will help test motor speed consistency.")
    print("\nCurrent EV3 IP: {}".format(EV3_IP))
    print("Current EV3 Port: {}".format(EV3_PORT))
    
    while True:
        print("\nMenu:")
        print("1. Run motor test")
        print("2. Change EV3 IP address")
        print("3. Exit")
        
        choice = input("\nEnter choice (1-3): ")
        
        if choice == "1":
            motor_test()
        elif choice == "2":
            change_ip()
        elif choice == "3":
            break
        else:
            print("Invalid choice!")

if __name__ == "__main__":
    main() 