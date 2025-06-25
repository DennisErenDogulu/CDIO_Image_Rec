# Ball Collection Robot - Refactored Code Structure

This project has been refactored from a single monolithic file into a modular, maintainable codebase. The system implements a continuous ball collection strategy using visual markers and computer vision.

## Project Structure

```
CDIO_Image_Rec/
├── src/                          # Source code package
│   ├── __init__.py              # Package initialization
│   ├── config.py                # Configuration settings
│   ├── utils.py                 # Utility functions
│   ├── robot_comms.py           # Robot communication
│   ├── vision.py                # Computer vision system
│   ├── navigation.py            # Path planning and navigation
│   └── main.py                  # Main application class
├── run.py                       # Application entry point
├── continuous_collector.py      # Original monolithic file (legacy)
├── README_refactored.md         # This documentation
└── ... (other project files)
```

### Core Modules

#### `src/config.py`
Central configuration file containing all constants and settings:
- Robot connection parameters (IP, port)
- Roboflow API settings for ball detection
- Color detection ranges for markers
- Physical dimensions and constraints
- Camera settings
- Movement parameters

#### `src/utils.py`
Utility functions used throughout the application:
- `point_to_line_distance()` - Geometric calculations
- `check_wall_collision()` - Collision detection
- `calculate_distance()` - Distance calculations
- `calculate_angle()` - Angle calculations
- `normalize_angle()` - Angle normalization

#### `src/robot_comms.py`
Robot communication module (`RobotComms` class):
- TCP socket communication with EV3 robot
- Movement commands (move, turn, collect)
- Ball delivery functionality
- Robot status monitoring
- Error handling and timeouts

#### `src/vision.py`
Computer vision module (`VisionSystem` class):
- Camera initialization and configuration
- Marker detection (green base, purple direction)
- Ball detection using Roboflow API
- Camera calibration and homography
- Obstacle area setup
- Debug visualization

#### `src/navigation.py`
Navigation and path planning module (`NavigationSystem` class):
- Goal management (small/large goals, side selection)
- Wall and obstacle setup
- Path planning around obstacles
- Collision detection
- Movement efficiency calculations
- Goal approach path calculation

#### `src/main.py`
Main application orchestrator (`BallCollector` class):
- Coordinates all subsystems
- Main collection loop
- Path execution
- User interface and visualization
- Live tracking and status display

#### `run.py`
Application entry point script:
- Sets up Python path for src package
- Imports and runs the main application
- Provides clean entry point from project root

## Key Improvements

### Modularity
- **Separation of Concerns**: Each module handles a specific aspect of the system
- **Single Responsibility**: Classes have well-defined purposes
- **Loose Coupling**: Modules interact through clean interfaces

### Maintainability
- **Configuration Management**: All settings centralized in `config.py`
- **Code Reusability**: Common functions extracted to `utils.py`
- **Clear Structure**: Easy to locate and modify specific functionality

### Testability
- **Isolated Components**: Each module can be tested independently
- **Mock-friendly**: Clear interfaces allow for easy mocking
- **Error Handling**: Improved error handling and logging

### Scalability
- **Easy Extension**: New features can be added without affecting other modules
- **Plugin Architecture**: New vision systems or robot types can be easily integrated
- **Performance**: Optimized frame processing and movement calculations

## Usage

### Running the Application
```bash
python run.py
```

Or from the src directory:
```bash
cd src
python -m main
```

### Key Components Usage

#### Vision System
```python
from src.vision import VisionSystem
# or
from src import VisionSystem

vision = VisionSystem()
balls = vision.detect_balls()
green_center, purple_center = vision.detect_markers(frame)
```

#### Navigation System
```python
from src.navigation import NavigationSystem
# or
from src import NavigationSystem

nav = NavigationSystem()
nav.select_goal('A')  # Select small goal
path = nav.calculate_goal_approach_path(current_pos, heading)
```

#### Robot Communication
```python
from src.robot_comms import RobotComms
# or  
from src import RobotComms

robot = RobotComms()
robot.move(50)  # Move forward 50cm
robot.turn(90)  # Turn 90 degrees
robot.collect(30)  # Collect while moving forward 30cm
```

## Configuration

### Robot Settings
Edit `src/config.py` to modify:
- Robot IP address and port
- Field dimensions
- Goal positions
- Movement parameters
- Camera settings

### Visual Markers
The system uses:
- **Green marker**: Robot base position
- **Purple marker**: Robot heading direction
- **Color ranges**: Configurable in `src/config.py`

## Dependencies

- OpenCV (`cv2`)
- NumPy
- Roboflow API
- Standard Python libraries (math, json, socket, logging, time)

## Features

### Automatic Operation
- Ball detection and collection
- Obstacle avoidance
- Goal delivery
- Continuous operation

### Manual Controls
- Goal selection (A/B)
- Goal side configuration
- Debug visualization
- Path execution control

### Safety Features
- Wall proximity detection
- Collision avoidance
- Safe backup procedures
- Movement validation

## Error Handling

The refactored code includes improved error handling:
- Network communication errors
- Vision system failures
- Movement command failures
- Calibration issues

## Logging

Comprehensive logging throughout all modules:
- Info level for normal operations
- Warning level for recoverable issues
- Error level for serious problems

## Future Enhancements

The modular structure makes it easy to add:
- Multiple robot support
- Different vision systems
- Alternative communication protocols
- New navigation algorithms
- Advanced path planning
- Machine learning integration

## Migration from Original

The refactored code maintains full compatibility with the original functionality while providing:
- Better code organization
- Easier debugging
- Simplified testing
- Enhanced maintainability
- Clearer documentation 