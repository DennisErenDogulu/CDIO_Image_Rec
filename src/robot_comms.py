#!/usr/bin/env python3
"""
Robot Communication Module

Handles all communication with the EV3 robot via TCP socket.
"""

import json
import socket
import logging
from .config import EV3_IP, EV3_PORT

logger = logging.getLogger(__name__)


class RobotComms:
    """Handles communication with the EV3 robot"""
    
    def __init__(self, ip=EV3_IP, port=EV3_PORT):
        self.ip = ip
        self.port = port
    
    def send_command(self, command: str, **params) -> bool:
        """Send a command to the EV3 server"""
        try:
            with socket.create_connection((self.ip, self.port), timeout=15) as s:
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
                
        except socket.timeout:
            logger.error("Command timed out - robot might still be moving")
            return False
        except Exception as e:
            logger.error("Failed to send command: {}".format(e))
            return False
    
    def move(self, distance_cm: float) -> bool:
        """Move forward/backward by distance_cm (negative for backward)"""
        return self.send_command("MOVE", distance=distance_cm)
    
    def turn(self, angle_deg: float) -> bool:
        """Turn by angle_deg (positive = CCW)"""
        return self.send_command("TURN", angle=angle_deg)
    
    def collect(self, distance_cm: float) -> bool:
        """Move forward slowly while collecting"""
        return self.send_command("COLLECT", distance=distance_cm)
    
    def stop(self) -> bool:
        """Stop all motors"""
        return self.send_command("STOP")
    
    def deliver_balls(self, duration: float) -> bool:
        """Run collector in reverse to deliver balls"""
        try:
            logger.info(f"Sending COLLECT_REVERSE command for {duration} seconds")
            result = self.send_command("COLLECT_REVERSE", duration=duration)
            
            if result:
                logger.info("COLLECT_REVERSE command executed successfully")
            else:
                logger.warning("COLLECT_REVERSE command returned failure")
            return result
        except Exception as e:
            logger.error("Exception during ball delivery: {}".format(e))
            return False
    
    def get_status(self) -> dict:
        """Get current robot status"""
        try:
            with socket.create_connection((self.ip, self.port), timeout=5) as s:
                message = json.dumps({"command": "STATUS"}) + "\n"
                s.sendall(message.encode("utf-8"))
                response = s.recv(1024).decode("utf-8").strip()
                return json.loads(response)
        except Exception as e:
            logger.error(f"Failed to get robot status: {e}")
            return {} 