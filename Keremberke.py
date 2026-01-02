"""
EcoSort - Waste Sorting System (Sprint 2 - Enhanced Plastic Detection)
Description: Waste detection using a specialized Garbage Detection Model (YOLOv8-Medium)
             Optimized for material classification (Plastic, Paper, Metal, Glass).
Author: EcoSort Development Team
Date: January 2026
"""

import cv2
import numpy as np
import os
import time
import sys
from ultralytics import YOLO

# ---------------------------------------------------------
# Hiwonder hardware libraries setup
# ---------------------------------------------------------
try:
    sys.path.append('/home/ubuntu/ArmPi/')
    import ArmIK.ArmMoveIK as ArmIK
    import HiwonderSDK.Board as Board
    ON_ROBOT = True
    print("Robot hardware detected – running in full operation mode.")
except ImportError:
    print("Warning: Hiwonder libraries not found. Running in simulation (Mock Mode).")
    ON_ROBOT = False

    class MockBoard:
        def setBusServoPulse(self, id, pulse, time):
            pass # print(f"[Mock Hardware] Servo {id} -> Pulse {pulse}")

    class MockArmIK:
        def setPitchRangeMoving(self, target, pitch, range_min, range_max):
            print(f"[Mock IK] Moving arm to {target}")
            return True 

    Board = MockBoard()
    ArmIK = MockArmIK

# ---------------------------------------------------------
# System constants and configuration
# ---------------------------------------------------------
IMAGE_FOLDER = "./input_images"

# --- MDOEL CONFIGURATION ---
# Using the specialized garbage detection model
MODEL_PATH = "keremberke/yolov8m-garbage-detection" 

# --- SENSITIVITY SETTINGS ---
# We tell the model to report EVERYTHING above 0.15, 
# but we only act on things above 0.30.
MODEL_INTERNAL_CONF = 0.15 
CONF_THRESHOLD = 0.30      

# --- UPDATED CLASS MAPPING (keremberke model) ---
# 0: bio, 1: cardboard, 2: glass, 3: metal, 4: paper, 5: plastic
WASTE_CLASSES = {
    'plastic': 5, 
    'glass': 2,
    'metal': 3,
    'paper': 4,
    'cardboard': 1
}

# What are we looking for right now?
TARGET_WASTE_TYPE = 'plastic' 

# Recycling bin locations (cm)
BIN_COORDS = {
    'plastic': (-15, 12, 10),
    'paper': (15, 12, 10),
    'metal': (0, 15, 10)
}

SERVO_ID_GRIPPER = 1
GRIPPER_OPEN = 500
GRIPPER_CLOSE = 2000

# ---------------------------------------------------------
# Class 1: Vision System
# ---------------------------------------------------------
class WasteDetector:
    def __init__(self, model_path):
        print(f"Loading Specialized Model: {model_path}...")
        print(">> Note: First run will download approx 50MB. Please wait.")
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)

    def detect_waste(self, image_path, target_type='plastic'):
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Unable to read image {image_path}")
            return None, []

        target_id = WASTE_CLASSES.get(target_type)
        
        # Run inference with low internal threshold to catch difficult items
        results = self.model(frame, conf=MODEL_INTERNAL_CONF, verbose=False) 
        detections = []

        print(f"\n--- Analyzing {os.path.basename(image_path)} ---")
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls)
                conf = float(box.conf)
                
                # Get the name of what the model sees
                detected_name = self.model.names[cls_id]

                # DEBUG PRINT: Show everything the model sees
                print(f"DEBUG: Saw '{detected_name}' (ID: {cls_id}) with confidence {conf:.2f}")

                # Check if it matches our target AND is confident enough
                if cls_id == target_id:
                    if conf >= CONF_THRESHOLD:
                        # Success!
                        x1, y1, x2, y2 = map(int, box.xyxy)
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)

                        detections.append({
                            "bbox": (x1, y1, x2, y2),
                            "centroid": (cx, cy),
                            "confidence": conf,
                            "type": target_type
                        })

                        # Draw Green Box (Accepted)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"Target: {target_type} {conf:.2f}"
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
                    else:
                        # Found plastic, but confidence too low
                        print(f"-> Found plastic but confidence ({conf:.2f}) is below threshold ({CONF_THRESHOLD})")

        return frame, detections

# ---------------------------------------------------------
# Class 2: Coordinate Mapper
# ---------------------------------------------------------
class CoordinateMapper: 
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.x_range_cm = (-15, 15)
        self.y_range_cm = (12, 30)

    def pixel_to_world(self, u, v):
        norm_x = u / self.width
        norm_y = v / self.height
        world_x = self.x_range_cm[0] + norm_x * (self.x_range_cm[1] - self.x_range_cm[0])
        world_y = self.y_range_cm[1] - norm_y * (self.y_range_cm[1] - self.y_range_cm[0])
        return (world_x, world_y, 2.0)

# ---------------------------------------------------------
# Class 3: Robot Controller
# ---------------------------------------------------------
class RobotController:
    def __init__(self):
        self.ik = ArmIK.ArmIK() if hasattr(ArmIK, 'ArmIK') else ArmIK
        self.board = Board
        self.reset_pose()

    def reset_pose(self):
        self.ik.setPitchRangeMoving((0, 10, 15), 0, -90, 0, 1500)
        time.sleep(1.0)

    def pickup_item(self, target_coords): 
        x, y, z = target_coords
        print(f"Action: Picking up item at ({x:.1f}, {y:.1f}, {z:.1f})")
        
        self.board.setBusServoPulse(SERVO_ID_GRIPPER, GRIPPER_OPEN, 500)
        time.sleep(0.5)
        
        if not self.ik.setPitchRangeMoving((x, y, z + 6), -90, -90, 0, 1500):
            print("Error: Target unreachable!")
            return False
        time.sleep(1.5)
        
        self.ik.setPitchRangeMoving((x, y, z), -90, -90, 0, 1000)
        time.sleep(1.0)
        
        self.board.setBusServoPulse(SERVO_ID_GRIPPER, GRIPPER_CLOSE, 500)
        time.sleep(0.8)
        
        self.ik.setPitchRangeMoving((0, 10, 15), 0, -90, 0, 1500)
        time.sleep(1.5)
        return True

    def place_item(self, bin_coords):
        x, y, z = bin_coords
        print(f"Action: Placing item in bin at ({x}, {y}, {z})")
        self.ik.setPitchRangeMoving((x, y, z + 8), -90, -90, 0, 1500)
        time.sleep(1.5)
        self.board.setBusServoPulse(SERVO_ID_GRIPPER, GRIPPER_OPEN, 500)
        time.sleep(0.5)
        self.reset_pose()

# ---------------------------------------------------------
# Main Logic
# ---------------------------------------------------------
def main():
    print(f"=== EcoSort System Started ===")
    print(f"Targeting: {TARGET_WASTE_TYPE.upper()} (Class ID: {WASTE_CLASSES[TARGET_WASTE_TYPE]})")
    
    detector = WasteDetector(MODEL_PATH)
    mapper = CoordinateMapper()
    robot = RobotController()

    if not os.path.exists(IMAGE_FOLDER):
        os.makedirs(IMAGE_FOLDER)
        print(f"Created folder '{IMAGE_FOLDER}'. Please add images there.")
        return

    images = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not images:
        print(f"No images found in '{IMAGE_FOLDER}'.")
        return

    for img_name in images:
        full_path = os.path.join(IMAGE_FOLDER, img_name)
        annotated_frame, detections = detector.detect_waste(full_path, target_type=TARGET_WASTE_TYPE)

        if detections:
            # Sort by confidence
            detections.sort(key=lambda x: x['confidence'], reverse=True)
            target = detections[0]
            
            u, v = target['centroid']
            world_coords = mapper.pixel_to_world(u, v)
            
            if ON_ROBOT:
                if robot.pickup_item(world_coords):
                    bin_pos = BIN_COORDS.get(target['type'], (-15, 12, 10))
                    robot.place_item(bin_pos)
            else:
                print("[Simulation] Robot movement sequence simulated.")
            
            # Save result
            output_name = f"result_{img_name}"
            cv2.imwrite(output_name, annotated_frame)
            print(f"SUCCESS: Saved detection to '{output_name}'")
        else:
            print(f"FAIL: No {TARGET_WASTE_TYPE} detected in {img_name}")

if __name__ == "__main__":
    main()
