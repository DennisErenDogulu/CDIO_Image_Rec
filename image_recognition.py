#!/usr/bin/env pybricks-micropython
from pybricks.hubs import EV3Brick
from pybricks.ev3devices import (Motor, TouchSensor, ColorSensor,
                                 InfraredSensor, UltrasonicSensor, GyroSensor)
from pybricks.parameters import Port, Stop, Direction, Button, Color
from pybricks.tools import wait, StopWatch, DataLog
from pybricks.robotics import DriveBase
from pybricks.media.ev3dev import SoundFile, ImageFile

# Initialize the EV3 brick and sensors
ev3 = EV3Brick()
left_motor = Motor(Port.B)
right_motor = Motor(Port.C)
collection_motor = Motor(Port.A)
ultrasonic_sensor = UltrasonicSensor(Port.S2)  # For obstacle detection

# Initialize the drive base
robot = DriveBase(left_motor, right_motor, wheel_diameter=55.5, axle_track=104)

# Constants for golf ball detection and navigation
WHITE_THRESHOLD = 80  # Higher threshold for white golf balls
OBSTACLE_DISTANCE = 200  # mm - distance to consider something an obstacle
GOLF_BALL_DISTANCE = 100  # mm - optimal distance for ball collection
SEARCH_SPEED = 100  # mm/s
TURN_ANGLE = 30  # degrees for obstacle avoidance
COLLECTION_TIME = 2000  # ms for ball collection

def is_white_ball():
    """
    Detect if there's a white golf ball using the color sensor
    Returns True if a white ball is detected
    """
    reflection = color_sensor.reflection()
    # White objects have high reflection values
    return reflection > WHITE_THRESHOLD

def detect_obstacle():
    """
    Check if there's an obstacle in front of the robot
    Returns True if an obstacle is detected
    """
    distance = ultrasonic_sensor.distance()
    return distance < OBSTACLE_DISTANCE

def avoid_obstacle():
    """
    Navigate around detected obstacles
    """
    ev3.speaker.beep()
    # Turn right to avoid obstacle
    robot.turn(TURN_ANGLE)
    # Move forward a bit
    robot.straight(100)
    # Turn left to resume original direction
    robot.turn(-TURN_ANGLE)

def align_with_ball():
    """
    Align the robot with the detected golf ball
    """
    attempts = 0
    max_attempts = 5
    
    while attempts < max_attempts:
        if is_white_ball():
            # Fine-tune position
            distance = ultrasonic_sensor.distance()
            if distance > GOLF_BALL_DISTANCE + 50:
                robot.straight(50)
            elif distance < GOLF_BALL_DISTANCE - 50:
                robot.straight(-50)
            else:
                return True
        else:
            # Search pattern
            robot.turn(20)
            attempts += 1
        wait(100)
    
    return False

def collect_golf_ball():
    """
    Collect a golf ball when one is detected
    """
    ev3.speaker.beep()
    # Run collection motor forward
    collection_motor.run_time(500, COLLECTION_TIME)
    wait(COLLECTION_TIME)
    # Verify collection
    if not is_white_ball():
        ev3.speaker.say("Ball collected")
        return True
    return False

def drop_golf_balls():
    """
    Drop all collected golf balls
    """
    ev3.speaker.beep()
    collection_motor.run_time(-500, 3000)
    wait(3000)
    ev3.speaker.say("Balls dropped")

def search_pattern():
    """
    Systematic search pattern for golf balls
    """
    # Drive forward while checking for balls and obstacles
    robot.drive(SEARCH_SPEED, 0)
    
    while True:
        # Check for obstacles first
        if detect_obstacle():
            robot.stop()
            avoid_obstacle()
            continue
        
        # Check for golf balls
        if is_white_ball():
            robot.stop()
            if align_with_ball():
                if collect_golf_ball():
                    return True
        
        # Small turn to scan area
        robot.turn(5)
        wait(100)
        
        # Check for emergency stop
        if Button.CENTER in ev3.buttons.pressed():
            robot.stop()
            return False

def autonomous_golf_mode():
    """
    Main autonomous mode for golf ball collection
    """
    ev3.speaker.say("Starting golf ball collection")
    balls_collected = 0
    
    while True:
        # Search for and collect balls
        if search_pattern():
            balls_collected += 1
            ev3.screen.clear()
            ev3.screen.print("Balls:", balls_collected)
            
            # Return to drop-off point
            ev3.speaker.say("Returning to base")
            robot.turn(180)
            robot.straight(500)
            
            # Drop collected balls
            drop_golf_balls()
            
            # Return to search area
            robot.turn(180)
            robot.straight(500)
        else:
            # Emergency stop was pressed
            ev3.speaker.say("Emergency stop")
            break

# Start the autonomous golf ball collection mode
autonomous_golf_mode() 