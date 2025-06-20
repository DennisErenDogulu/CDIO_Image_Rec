#!/usr/bin/env python3
"""
Model Tester for Roboflow Weights
This script compares the performance of different weight files for object detection.
"""

import cv2
import numpy as np
from roboflow import Roboflow
import time
import logging
from typing import Dict, List, Tuple

# Configuration
ROBOFLOW_API_KEY = "LdvRakmEpZizttEFtQap"
RF_WORKSPACE = "legoms3"
RF_PROJECT = "golfbot-fyxfe-etdz0"
RF_VERSION = 2

class ModelTester:
    def __init__(self):
        # Initialize logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize Roboflow
        self.rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        self.project = self.rf.workspace(RF_WORKSPACE).project(RF_PROJECT)
        
        # Initialize camera
        self.cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Performance metrics
        self.metrics = {
            'weights(1).pt': {'fps': [], 'detections': []},
            'weights (2).pt': {'fps': [], 'detections': []}
        }

    def test_model(self, weight_file: str) -> None:
        """Test a specific weight file"""
        model = self.project.version(RF_VERSION).model
        
        self.logger.info(f"Testing {weight_file}...")
        cv2.namedWindow(f"Test - {weight_file}")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # Create copy for visualization
            display = frame.copy()
            
            # Time the prediction
            start_time = time.time()
            
            # Get predictions
            predictions = model.predict(frame, confidence=30, overlap=30).json()
            
            # Calculate FPS
            fps = 1.0 / (time.time() - start_time)
            self.metrics[weight_file]['fps'].append(fps)
            
            # Store number of detections
            num_detections = len(predictions.get('predictions', []))
            self.metrics[weight_file]['detections'].append(num_detections)
            
            # Draw predictions
            for pred in predictions.get('predictions', []):
                x = pred['x']
                y = pred['y']
                width = pred['width']
                height = pred['height']
                class_name = pred['class']
                confidence = pred['confidence']
                
                # Calculate bounding box coordinates
                x1 = int(x - width/2)
                y1 = int(y - height/2)
                x2 = int(x + width/2)
                y2 = int(y + height/2)
                
                # Draw box and label
                color = (0, 255, 0) if class_name not in ['egg', 'cross'] else (0, 0, 255)
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display, f"{class_name} {confidence:.1f}%",
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, color, 2)
            
            # Draw FPS
            cv2.putText(display, f"FPS: {fps:.1f}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       1, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow(f"Test - {weight_file}", display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('n'):
                cv2.destroyWindow(f"Test - {weight_file}")
                return
    
    def print_metrics(self):
        """Print comparison metrics for both models"""
        self.logger.info("\nModel Comparison Results:")
        self.logger.info("-" * 50)
        
        for weight_file, metrics in self.metrics.items():
            avg_fps = np.mean(metrics['fps']) if metrics['fps'] else 0
            avg_detections = np.mean(metrics['detections']) if metrics['detections'] else 0
            fps_std = np.std(metrics['fps']) if metrics['fps'] else 0
            
            self.logger.info(f"\nModel: {weight_file}")
            self.logger.info(f"Average FPS: {avg_fps:.2f} ± {fps_std:.2f}")
            self.logger.info(f"Average Detections per Frame: {avg_detections:.2f}")
    
    def run(self):
        """Run tests for both weight files"""
        try:
            self.logger.info("Starting model comparison...")
            self.logger.info("Press 'q' to quit current test, 'n' for next model")
            
            # Test both weight files
            for weight_file in self.metrics.keys():
                self.test_model(weight_file)
            
            # Print comparison results
            self.print_metrics()
            
        except Exception as e:
            self.logger.error(f"Error during testing: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tester = ModelTester()
    tester.run() 