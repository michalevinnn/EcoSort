import time
import random

# --- Configuration and Locations ---
LOCATIONS = {
    # Home Pose: A high, central point from which the camera sees the entire surface
    "HOME_POSE": (0, 20, 25), 
    
    # Bin Locations (Where the arm goes to release the waste)
    "BINS": {
        "plastic": (-15, 20, 10), 
        "paper": (15, 20, 10),    
        "general": (-20, 10, 10)  
    }
}

# --- Dummy Hardware Functions (To be replaced with real SDK commands) ---

def move_arm_to(coords):
    print(f"🤖 Robot moving to point: {coords}")
    time.sleep(1.5) # Simulate movement time

def open_gripper():
    print("👐 Gripper opening")

def close_gripper():
    print("✊ Gripper closing")

def detect_objects_from_camera():
    # This function mocks the computer vision model
    # It returns a list of detected objects
    if random.choice([True, False]): 
        return [{"type": "plastic", "coords": (5, 10, 2)}] # Mock detection
    else:
        return [] # Nothing detected

# --- Calibration Wizard Function ---

def run_calibration_wizard():
    print("=== Starting Location Calibration Mode ===")
    print("Instructions: The robot will move point-by-point. Place the bin/robot according to the arm's position.")
    input("Press Enter to start...")

    # 1. Calibrate Home Position
    print("\n📍 Moving to Home position...")
    move_arm_to(LOCATIONS["HOME_POSE"])
    print("=> Please ensure the robot is stable and the camera view is clear.")
    input("Is the position correct? Press Enter to continue...")

    # 2. Calibrate Bins
    print("\n🗑️ Starting bin calibration...")
    for bin_name, coords in LOCATIONS["BINS"].items():
        print(f"-> Moving to location for bin: {bin_name}")
        move_arm_to(coords)
        
        # The robot waits for you to place the bin
        print(f"🛑 STOP! Place the '{bin_name}' bin directly under the gripper.")
        input(f"Did you place the {bin_name} bin? Press Enter to continue...")
        
        # Optional: Lift arm slightly for safety before moving to next point
        print("Lifting arm for safety...")
        safe_coords = (coords[0], coords[1], coords[2] + 5)
        move_arm_to(safe_coords)

    print("\n✅ Calibration finished! Physical space is synced with the code.")
    
    # Return to Home at the end
    move_arm_to(LOCATIONS["HOME_POSE"])

# --- Main Sorting Logic ---

def run_sorting_robot():
    print("--- Starting Sorting Process ---")
    
    while True:
        # 1. Return to Home position to scan
        print("\n📍 Returning to scan position (Home)...")
        move_arm_to(LOCATIONS["HOME_POSE"])
        open_gripper() # Ensure gripper is open
        
        # 2. Capture and analyze image
        print("📷 Capturing and analyzing surface...")
        detected_objects = detect_objects_from_camera()
        
        # 3. Exit condition: If list is empty, surface is clean
        if not detected_objects:
            print("✅ Surface is clean! Waiting for new command or finishing.")
            break # Or use time.sleep() to keep waiting
            
        # 4. If waste is found - handle the FIRST item only
        target_object = detected_objects[0]
        obj_type = target_object["type"]
        obj_coords = target_object["coords"]
        
        print(f"👀 Object detected: {obj_type} at {obj_coords}")
        
        # 5. Pick up object
        move_arm_to(obj_coords) # Approach object
        close_gripper()         # Grab object
        
        # 6. Move to appropriate bin
        if obj_type in LOCATIONS["BINS"]:
            bin_coords = LOCATIONS["BINS"][obj_type]
            print(f"🗑️ Moving to {obj_type} bin...")
            move_arm_to(bin_coords)
            open_gripper() # Release waste
        else:
            print(f"⚠️ Unknown waste type: {obj_type}, skipping.")

        # 7. End of iteration - Loop returns to step 1 (Home)

# --- Main Menu ---
if __name__ == "__main__":
    print("Select Mode:")
    print("1. Calibration Wizard")
    print("2. Run Sorting Robot")
    choice = input("Enter choice (1 or 2): ")
    
    if choice == '1':
        run_calibration_wizard()
    elif choice == '2':
        run_sorting_robot()
    else:
        print("Invalid choice.")
