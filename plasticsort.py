"""
EcoSort - Waste Sorting System (Sprint 1)
Description: Waste detection from images using YOLOv8 and robotic pickup.
             Based on YOLOv8 documentation and OpenCV .
Author: EcoSort Development Team
Date: December 2025
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
    # Estimated path to robot libraries on Raspberry Pi, cheacking the connection between the code and the robot
    sys.path.append('/home/ubuntu/ArmPi/')
    import ArmIK.ArmMoveIK as ArmIK
    import HiwonderSDK.Board as Board

    ON_ROBOT = True
    print("Robot hardware detected – running in full operation mode.")
except ImportError: #if the connection to the robot failed, the code continu to run from here
    print("Warning: Hiwonder libraries not found. Running in simulation (Mock Mode).")
    ON_ROBOT = False

    # Mock classes for logic testing without physical hardware
    class MockBoard:
        def setBusServoPulse(self, id, pulse, time):
            print(f"[Mock Hardware] Servo {id} -> Pulse {pulse} in {time}ms")

        def setMotor(self, v1, v2, v3, v4):
            # print(f"[Mock Hardware] Motors set to speeds: {v1}, {v2}, {v3}, {v4}")
            pass

    class MockArmIK:
        def setPitchRangeMoving(self, target, pitch, range_min, range_max, time):
            print(f"[Mock IK] Moving arm to target {target} (x,y,z) with pitch {pitch}")
            return True 

    Board = MockBoard()
    ArmIK = MockArmIK

# ---------------------------------------------------------
# System constants and configuration
# ---------------------------------------------------------
IMAGE_FOLDER = "./input_images"
MODEL_PATH = "yolov8n.pt"  # Pre-trained COCO model 

# --- WASTE CLASSIFICATION CONFIGURATION ---
CONF_THRESHOLD = 0.5 #lower standart 

# Mapping based on COCO dataset IDs for "Sprint 1" demo.
# In future sprints, after training on Garbage Data ,
# these IDs will change to match your custom model (e.g., 0: plastic, 1: paper).
WASTE_CLASSES = {
    'plastic': 39,  # COCO ID for 'bottle'
    'paper': 73,    # COCO ID for 'book' (proxy for paper)
    'metal': 41     # COCO ID for 'cup' (proxy for can)
}

# Define which waste type we are looking for in this run
TARGET_WASTE_TYPE = 'plastic' 

# Recycling bin locations (cm)
BIN_COORDS = {
    'plastic': (-15, 12, 10),
    'paper': (15, 12, 10)
}

# Servo IDs
SERVO_ID_GRIPPER = 1
GRIPPER_OPEN = 500
GRIPPER_CLOSE = 2000

# ---------------------------------------------------------
# Class 1: Vision System
# ---------------------------------------------------------
class WasteDetector:
    def __init__(self, model_path):
        """Initialize the YOLOv8 detection model"""
        print(f"Loading YOLOv8 model from: {model_path}...")
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            print(f"Model loading error: {e}")
            sys.exit(1)

    def detect_waste(self, image_path, target_type='plastic'):#the main function, get picture and type of waste to find
        """
        Perform detection on a single image.
        """
        # Load image using OpenCV 
        frame = cv2.imread(image_path) #get the type of waste and Converts it to a number
        if frame is None:
            print(f"Error: Unable to read image {image_path}") #checking if the number is valid
            return None, []

        # Get the ID we are looking for from our config
        target_id = WASTE_CLASSES.get(target_type) 

        # Run inference
        # Note: For video stream in future, use model.track() 
        results = self.model(frame, verbose=True) 
        detections = []

        for r in results: #the loop is running on the pictures, if one of them contain waste it wukk save in the detection array
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls)
                conf = float(box.conf)

                # Filter: High confidence AND matches our target waste type
                if conf >= CONF_THRESHOLD and cls_id == target_id:
                    
                    x1, y1, x2, y2 = map(int, box.xyxy) # Convert tensor to ints

                    # Compute centroid
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    detections.append({
                        "bbox": (x1, y1, x2, y2),
                        "centroid": (cx, cy),
                        "confidence": conf,
                        "class": cls_id,
                        "type": target_type
                    })

                    # Visualization
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{target_type} ({conf:.2f})"
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        return frame, detections

# ---------------------------------------------------------
# Class 2: Coordinate Mapper
# ---------------------------------------------------------
class CoordinateMapper: #this class convert pictures in 2D to 3D object in CM
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        # Simple linear calibration for table workspace
        self.x_range_cm = (-15, 15)
        self.y_range_cm = (12, 30)

    def pixel_to_world(self, u, v):
        """Convert pixel (u,v) to robot world coords (x,y,z)"""
        norm_x = u / self.width
        norm_y = v / self.height

        world_x = self.x_range_cm[0] + norm_x * (self.x_range_cm[1] - self.x_range_cm[0])
        world_y = self.y_range_cm[1] - norm_y * (self.y_range_cm[1] - self.y_range_cm[0])
        world_z = 2.0 # Fixed height for pickup

        return (world_x, world_y, world_z)

# ---------------------------------------------------------
# Class 3: Robot Controller
# ---------------------------------------------------------
class RobotController:
    def __init__(self):
        if hasattr(ArmIK, 'ArmIK'):
            self.ik = ArmIK.ArmIK()
        else:
            self.ik = ArmIK
        self.board = Board
        self.reset_pose()

    def reset_pose(self):
        self.ik.setPitchRangeMoving((0, 10, 15), 0, -90, 0, 1500)
        time.sleep(1.0)

    def pickup_item(self, target_coords): #the order to pick up the trash
        x, y, z = target_coords
        print(f"Starting pickup at: {x:.1f}, {y:.1f}, {z:.1f}")

        # 1. Open Gripper
        self.board.setBusServoPulse(SERVO_ID_GRIPPER, GRIPPER_OPEN, 500)
        time.sleep(0.5)

        # 2. Approach (Higher Z)
        if not self.ik.setPitchRangeMoving((x, y, z + 6), -90, -90, 0, 1500):
            print("Target unreachable!")
            return False
        time.sleep(1.5)

        # 3. Descend
        self.ik.setPitchRangeMoving((x, y, z), -90, -90, 0, 1000)
        time.sleep(1.0)

        # 4. Grab
        self.board.setBusServoPulse(SERVO_ID_GRIPPER, GRIPPER_CLOSE, 500)
        time.sleep(0.8)

        # 5. Lift
        self.ik.setPitchRangeMoving((0, 10, 15), 0, -90, 0, 1500)
        time.sleep(1.5)
        return True

    def place_item(self, bin_coords):
        x, y, z = bin_coords
        print(f"Placing item at: {x}, {y}, {z}")
        self.ik.setPitchRangeMoving((x, y, z + 8), -90, -90, 0, 1500)
        time.sleep(1.5)
        self.board.setBusServoPulse(SERVO_ID_GRIPPER, GRIPPER_OPEN, 500)
        time.sleep(0.5)
        self.reset_pose()

# ---------------------------------------------------------
# Main Logic
# ---------------------------------------------------------
def main():
    try:
        detector = WasteDetector(MODEL_PATH)
        mapper = CoordinateMapper()
        robot = RobotController()
    except Exception as e:
        print(f"Init failed: {e}")
        return

    if not os.path.exists(IMAGE_FOLDER):
        os.makedirs(IMAGE_FOLDER)
        print(f"Created {IMAGE_FOLDER}. Add images and run again.")
        return

    images = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not images:
        print("No images found.")
        return

    for img_name in images:
        full_path = os.path.join(IMAGE_FOLDER, img_name)
        print(f"\n--- Processing: {img_name} ---")

        # Use the generic detect_waste function
        annotated_frame, detections = detector.detect_waste(full_path, target_type=TARGET_WASTE_TYPE)

        if detections:
            # Take the most confident detection
            detections.sort(key=lambda x: x['confidence'], reverse=True)
            target = detections[0]
            
            u, v = target['centroid']
            print(f"Found {target['type']} at ({u}, {v})")

            world_coords = mapper.pixel_to_world(u, v)
            
            if ON_ROBOT:
                if robot.pickup_item(world_coords):
                    # Get bin coordinates based on type
                    bin_pos = BIN_COORDS.get(target['type'], (-15, 12, 10))
                    robot.place_item(bin_pos)
            else:
                print("[Simulation] Pickup sequence simulated.")
            
            # Save result
            cv2.imwrite(f"result_{img_name}", annotated_frame)
        else:
            print(f"No {TARGET_WASTE_TYPE} detected.")

if __name__ == "__main__":
    main()
