import socket
import json
import time

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

def turn_test():
    print("\nTurn Calibration Test")
    print("-------------------")
    print("This test will verify turning accuracy.")
    print("\nTest options:")
    print("1. 90-degree turn test")
    print("2. 180-degree turn test")
    print("3. Custom angle turn test")
    print("4. Back to main menu")
    
    choice = input("\nEnter choice (1-4): ")
    
    if choice == "1":
        angle = 90
    elif choice == "2":
        angle = 180
    elif choice == "3":
        angle = float(input("\nEnter turn angle in degrees: "))
    else:
        return
    
    print("\nSetup instructions:")
    print("1. Place the robot on a flat surface")
    print("2. Mark the starting orientation (use tape or marker)")
    print("3. Press Enter when ready")
    input()
    
    # Execute turn
    print("\nTurning {} degrees...".format(angle))
    success = send_command("TURN", angle=angle)
    
    if success:
        print("\nTurn completed!")
        print("\nMeasurement instructions:")
        print("1. Use a protractor or angle measurement tool")
        print("2. Measure the actual angle turned")
        print("3. Compare with the requested {} degrees".format(angle))
        
        actual_angle = float(input("\nEnter the actual angle turned (in degrees): "))
        error = actual_angle - angle
        
        print("\nResults:")
        print("Requested angle: {} degrees".format(angle))
        print("Actual angle: {:.1f} degrees".format(actual_angle))
        print("Error: {:.1f} degrees".format(error))
        
        if abs(error) > 5:  # More than 5 degrees error
            adjustment = angle / actual_angle
            print("\nSignificant error detected!")
            print("Current configuration:")
            print("- Wheel diameter: 69 mm")
            print("- Axle track: 150 mm")
            print("\nTry adjusting the AXLE_TRACK_MM value in tcp_server.py:")
            print("Current: 150 mm")
            print("Suggested: {:.1f} mm".format(150 * adjustment))
    else:
        print("Error: Turn command failed!")

def change_ip():
    """Change the EV3 IP address"""
    global EV3_IP
    EV3_IP = input("Enter new EV3 IP address: ")

def main():
    print("EV3 Turn Calibration Tool")
    print("========================")
    print("This tool will help calibrate the robot's turning accuracy.")
    print("\nCurrent EV3 IP: {}".format(EV3_IP))
    print("Current EV3 Port: {}".format(EV3_PORT))
    
    while True:
        print("\nMenu:")
        print("1. Run turn test")
        print("2. Change EV3 IP address")
        print("3. Exit")
        
        choice = input("\nEnter choice (1-3): ")
        
        if choice == "1":
            turn_test()
        elif choice == "2":
            change_ip()
        elif choice == "3":
            break
        else:
            print("Invalid choice!")

if __name__ == "__main__":
    main() 