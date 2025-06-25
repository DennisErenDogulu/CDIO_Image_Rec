#!/usr/bin/env python3
"""
Configuration settings for the Ball Collection Robot
"""

import numpy as np

# Robot connection settings
EV3_IP = "172.20.10.6"
EV3_PORT = 12345

# Roboflow API settings
ROBOFLOW_API_KEY = "LdvRakmEpZizttEFtQap"
RF_WORKSPACE = "legoms3"
RF_PROJECT = "golfbot-fyxfe-etdz0"
RF_VERSION = 4

# Color detection ranges (HSV)
GREEN_LOWER = np.array([35, 50, 50])
GREEN_UPPER = np.array([85, 255, 255])

# Purple ranges for the direction marker
PURPLE_RANGES = [
    # Main purple range (violet/purple hues)
    (np.array([120, 60, 60]), np.array([150, 255, 255])),
    # Magenta-purple range (for different lighting)
    (np.array([145, 50, 50]), np.array([170, 255, 255])),
]

# Marker dimensions
GREEN_MARKER_WIDTH_CM = 20  # Width of green base sheet
PURPLE_MARKER_WIDTH_CM = 5  # Width of purple direction marker

# Wall configuration
WALL_SAFETY_MARGIN = 1  # cm, minimum distance to keep from walls

# Goal configuration
SMALL_GOAL_SIDE = "left"  # "left" or "right" - where the small goal (A) is placed
GOAL_OFFSET_CM = 6   # cm, distance to stop before goal edge (closer to goal)

# Robot configuration
ROBOT_START_X = 20  # cm from left edge
ROBOT_START_Y = 20  # cm from bottom edge
ROBOT_WIDTH = 26.5   # cm
ROBOT_LENGTH = 20  # cm
ROBOT_START_HEADING = 0  # degrees (0 = facing east)

# Physical constraints
FIELD_WIDTH_CM = 180
FIELD_HEIGHT_CM = 120
COLLECTION_DISTANCE_CM = 20  # Distance to move forward when collecting
APPROACH_DISTANCE_CM = 30    # Distance to keep from ball for approach
MAX_BALLS_PER_TRIP = 5

# Camera settings
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 15

# Performance settings
FRAME_SKIP_INTERVAL = 2  # Process every 3rd frame for better performance

# Movement settings
DELIVERY_TIME = 15.0  # Seconds to run collector in reverse

# Ignored area (center obstacle) - will be set during calibration
IGNORED_AREA = {
    "x_min": 0, "x_max": 0,
    "y_min": 0, "y_max": 0
} 