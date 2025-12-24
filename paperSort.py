#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EcoSort - Waste Sorting System (Sprint 1)
Description: Paper waste detection from images using YOLOv8 and robotic pickup
             using the Hiwonder ArmPi Pro SDK.
Author: EcoSort Development Team
Date: February 2025
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
    # Estimated path to robot libraries
    sys.path.append('/home/ubuntu/ArmPi/')
    import ArmIK.ArmMoveIK as ArmIK
    import HiwonderSDK.Board as Board

    ON_ROBOT = True
    print("Robot hardware detected – running in full operation mode.")
except ImportError:
    print("Warning: Hiwonder libraries not found. Running in simulation (Mock Mode).")
    ON_ROBOT = False


    # Mock classes for logic testing without physical hardware
    class MockBoard:
        def setBusServoPulse(self, id, pulse, time):
            print(f"[Mock Hardware] Servo {id} -> Pulse {pulse} in {time}ms")

        def setMotor(self, v1, v2, v3, v4):
            print(f"[Mock Hardware] Motors set to speeds: {v1}, {v2}, {v3}, {v4}")


    class MockArmIK:
        def setPitchRangeMoving(self, target, pitch, range_min, range_max, time):
            print(f"[Mock IK] Moving arm to target {target} (x,y,z) with pitch {pitch}")
            return True  # Indicates that the movement is possible


    Board = MockBoard()
    ArmIK = MockArmIK

# ---------------------------------------------------------
# System constants and configuration
# ---------------------------------------------------------
IMAGE_FOLDER = "./input_images"  # Path to input images folder
MODEL_PATH = "yolov8n.pt"  # Nano model for fast performance

# --- CONFIGURATION FOR PAPER DETECTION ---
CONF_THRESHOLD = 0.5  # Minimum confidence threshold
# Note: In standard COCO dataset, there is no "Paper".
# We use ID 73 ('book') as a temporary proxy for testing paper handling.
# If you train a custom model, change this ID to match your 'paper' class.
CLASS_ID_Plastic = 39

# Servo IDs (based on ArmPi Pro documentation)
SERVO_ID_GRIPPER = 1
SERVO_ID_WRIST = 2
SERVO_ID_ELBOW = 3
SERVO_ID_SHOULDER = 4
SERVO_ID_BASE = 6

# Gripper states (PWM values in microseconds)
GRIPPER_OPEN = 500
GRIPPER_CLOSE = 2000

# Recycling bin location (robot base coordinate system, cm)
PAPER_BIN_COORDS = (-15, 12, 10)


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

    def detect_plastic(self, image_path):
        """
        Perform detection on a single image and return detected objects
        ONLY if they match the plastic Class ID.
        """
        frame = cv2.imread(image_path) #פונקציה שממירה תמונות לדאטא
        if frame is None:
            print(f"Error: Unable to read image {image_path}")
            return None, []

        # Run inference (verbose=True for development debugging)
        results = self.model(frame, verbose=True)
        detections = []

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Extract class ID and confidence
                cls_id = int(box.cls)
                conf = float(box.conf)

                # Filter: Must be high confidence AND match the Paper Class ID
                if conf >= CONF_THRESHOLD and cls_id == CLASS_ID_Plastic:
                    # Bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Compute centroid (grasp target)
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    detections.append({
                        "bbox": (x1, y1, x2, y2),
                        "centroid": (cx, cy),
                        "confidence": conf,
                        "class": cls_id
                    })

                    # Draw bounding box and label
                    # Note: We hardcode "Paper" label because we filtered by ID
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Paper (ID:{cls_id}) {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Draw grasp point
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        return frame, detections


# ---------------------------------------------------------
# Class 2: Coordinate Mapper
# ---------------------------------------------------------
class CoordinateMapper:
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height

        # Simple linear calibration
        self.x_range_cm = (-15, 15)  # X axis
        self.y_range_cm = (12, 30)  # Y axis

    def pixel_to_world(self, u, v):
        """Convert image pixel coordinates (u, v) to robot coordinates (X, Y, Z)."""
        # Normalize pixel coordinates
        norm_x = u / self.width
        norm_y = v / self.height

        # Linear interpolation
        world_x = self.x_range_cm[0] + norm_x * (self.x_range_cm[1] - self.x_range_cm[0])
        world_y = self.y_range_cm[1] - norm_y * (self.y_range_cm[1] - self.y_range_cm[0])

        # Assume object lies on the ground
        world_z = 2.0  # cm

        return (world_x, world_y, world_z)


# ---------------------------------------------------------
# Class 3: Robot Controller
# ---------------------------------------------------------
class RobotController:
    def __init__(self):
        # Initialize inverse kinematics module
        if hasattr(ArmIK, 'ArmIK'):
            self.ik = ArmIK.ArmIK()
        else:
            self.ik = ArmIK

        self.board = Board
        print("Initializing robot controller... Moving to home position.")
        self.reset_pose()

    def reset_pose(self):
        """Return robot arm to a safe neutral pose"""
        self.ik.setPitchRangeMoving((0, 10, 15), 0, -90, 0, 1500)
        time.sleep(1.5)

    def pickup_item(self, target_coords):
        """Execute full pickup sequence"""
        x, y, z = target_coords
        print(f"Starting pickup sequence at: {x:.2f}, {y:.2f}, {z:.2f}")

        self.board.setBusServoPulse(SERVO_ID_GRIPPER, GRIPPER_OPEN, 500)
        time.sleep(0.5)

        # Move above target
        result = self.ik.setPitchRangeMoving((x, y, z + 5), -90, -90, 0, 1500)
        if not result:
            print("Error: Target is outside robot workspace!")
            return False
        time.sleep(1.5)

        # Descend
        self.ik.setPitchRangeMoving((x, y, z), -90, -90, 0, 1000)
        time.sleep(1.0)

        # Grasp
        self.board.setBusServoPulse(SERVO_ID_GRIPPER, GRIPPER_CLOSE, 500)
        time.sleep(0.8)

        # Lift
        self.ik.setPitchRangeMoving((0, 10, 15), 0, -90, 0, 1500)
        time.sleep(1.5)

        return True

    def place_item(self, bin_coords):
        """Place the object into the recycling bin"""
        x, y, z = bin_coords
        print(f"Placing item in bin at: {x}, {y}, {z}")

        self.ik.setPitchRangeMoving((x, y, z + 5), -90, -90, 0, 1500)
        time.sleep(1.5)

        self.board.setBusServoPulse(SERVO_ID_GRIPPER, GRIPPER_OPEN, 500)
        time.sleep(0.5)

        self.reset_pose()


# ---------------------------------------------------------
# Main execution logic
# ---------------------------------------------------------
def main():
    try:
        detector = WasteDetector(MODEL_PATH)
        mapper = CoordinateMapper()
        robot = RobotController()
    except Exception as e:
        print(f"System initialization failed: {e}")
        return

    if not os.path.exists(IMAGE_FOLDER):
        print(f"Creating folder {IMAGE_FOLDER}. Please add images and rerun.")
        os.makedirs(IMAGE_FOLDER)
        return

    images = [f for f in os.listdir(IMAGE_FOLDER)
              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if len(images) == 0:
        print(f"The folder {IMAGE_FOLDER} is empty. Please add images.")
        return

    print(f"Found {len(images)} images. Starting processing...")

    for i, img_name in enumerate(images):
        full_path = os.path.join(IMAGE_FOLDER, img_name)
        print(f"\n--- Processing image {i + 1}/{len(images)}: {img_name} ---")

        annotated_frame, detections = detector.detect_paper(full_path)

        if not detections:
            print("No paper detected in this image.")
            continue

        # Sort by confidence (highest first)
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        target = detections[0]

        u, v = target['centroid']
        print(f"Paper detected at pixel ({u}, {v}) with confidence {target['confidence']:.2f}")

        world_coords = mapper.pixel_to_world(u, v)
        print(f"Computed robot coordinates (cm): {world_coords}")

        if ON_ROBOT:
            success = robot.pickup_item(world_coords)
            if success:
                robot.place_item(PAPER_BIN_COORDS)
                print("Operation completed successfully.")
            else:
                print("Pickup failed.")
        else:
            print("[Simulation] Robot would execute pickup here.")

        output_name = f"result_{img_name}"
        cv2.imwrite(output_name, annotated_frame)
        print(f"Annotated image saved as: {output_name}")

    print("\nSprint 1 execution completed.")


if __name__ == "__main__":
    main()
