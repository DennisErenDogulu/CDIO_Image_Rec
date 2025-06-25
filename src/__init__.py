"""
Ball Collection Robot Package

A modular ball collection system using computer vision and autonomous navigation.
"""

__version__ = "1.0.0"
__author__ = "Ball Collection Team"

# Import main classes for convenience
from .vision import VisionSystem
from .navigation import NavigationSystem
from .robot_comms import RobotComms

__all__ = [
    'VisionSystem',
    'NavigationSystem', 
    'RobotComms'
] 