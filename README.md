# Robot Vision Ball Collection System

This project combines computer vision with EV3 robot control to create a system that automatically detects and collects colored balls.

## Features

- Detects orange and white balls using OpenCV
- Controls an EV3 robot to navigate toward detected balls
- Collects balls using a collection mechanism
- Returns to drop-off point once collection is complete

## Requirements

- LEGO EV3 Brick with Pybricks firmware
- Camera connected to the system (webcam or compatible camera)
- Python 3.x
- OpenCV (cv2)
- Pybricks MicroPython library

## Hardware Setup

1. Connect motors to the following ports:
   - Left drive motor: Port B
   - Right drive motor: Port C
   - Collection mechanism motor: Port A

2. Connect a camera to your system (set the correct camera index in the code)

## Running the Program

1. Connect to your EV3 brick
2. Upload the `tcp_server.py` file to your EV3 brick
3. Run the program on the EV3 brick

The robot will:
1. Scan the environment using the camera
2. Detect orange and white balls
3. Navigate toward detected balls
4. Collect up to 3 balls
5. Return to the drop-off point
6. Drop the collected balls

## Configuration

You can adjust the following parameters in the code:

- Maximum number of balls to collect
- Movement parameters (speed, turning angle)
- Camera settings (resolution, camera index)

## Troubleshooting

- If balls are not being detected correctly
- If the robot moves too quickly or slowly, adjust the movement parameters
- If the camera isn't working, check the camera index and connection 