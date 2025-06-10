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

def calibration_test():
    print("\nMovement Calibration Test")
    print("------------------------")
    print("1. Place the robot at a starting position")
    print("2. Mark the start point at the CENTER of the drive wheels")
    print("3. Make sure you have 1.2 meters of clear space ahead")
    print("4. Press Enter when ready")
    input()
    
    # Move forward 100cm
    print("\nMoving forward 100cm...")
    success = send_command("MOVE", distance=100)
    
    if success:
        print("\nMovement completed!")
        print("Please measure the actual distance traveled from the starting line to the CENTER of the drive wheels.")
        print("\nCalculation:")
        print("If the robot moved X centimeters:")
        print("Adjustment factor = 100/X")
        
        actual_dist = float(input("\nEnter the actual distance traveled (in cm): "))
        adjustment = 100 / actual_dist
        
        print("\nResults:")
        print("Actual distance: {:.1f} cm".format(actual_dist))
        print("Current conversion factor: 10 (cm to mm)")
        print("Suggested new conversion factor: {:.1f}".format(10 * adjustment))
        
        if actual_dist < 95 or actual_dist > 105:
            print("\nRecommendation:")
            print("In tcp_server.py, update the movement conversion line to:")
            print('mdiff.on_for_distance(SpeedRPM(NORMAL_SPEED_RPM), distance_cm * {:.1f})  # adjusted cm to mm'.format(10 * adjustment))
    else:
        print("Error: Movement command failed!")

def change_ip():
    """Change the EV3 IP address"""
    global EV3_IP
    EV3_IP = input("Enter new EV3 IP address: ")

def main():
    print("EV3 Movement Calibration Tool")
    print("============================")
    print("This tool will help calibrate the robot's movement distance.")
    print("Make sure you have:")
    print("1. At least 1.2 meters of clear space")
    print("2. A measuring tape or meter stick")
    print("3. The EV3 robot running tcp_server.py")
    print("\nCurrent EV3 IP: {}".format(EV3_IP))
    print("Current EV3 Port: {}".format(EV3_PORT))
    
    while True:
        print("\nMenu:")
        print("1. Run calibration test")
        print("2. Change EV3 IP address")
        print("3. Exit")
        
        choice = input("\nEnter choice (1-3): ")
        
        if choice == "1":
            calibration_test()
        elif choice == "2":
            change_ip()
        elif choice == "3":
            break
        else:
            print("Invalid choice!")

if __name__ == "__main__":
    main() 