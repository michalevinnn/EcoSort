import time
import sys
# וודאי שהנתיב לספריות נכון (כמו בקוד הקודם שלך)
sys.path.append('/home/ubuntu/ArmPi/')
import ArmIK.ArmMoveIK as ArmIK
import HiwonderSDK.Board as Board

# אותן קואורדינטות שהגדרנו בקוד הראשי
BIN_LOCATIONS = {
    "ORANGE (Plastic)": (-15, 12, 10), 
    "BLUE (Paper)": (15, 12, 10)
}

def main():
    ik = ArmIK.ArmIK()
    board = Board
    
    # 1. איפוס: הולכים למצב בית
    print("Moving to Home position...")
    ik.setPitchRangeMoving((0, 10, 15), 0, -90, 0, 1500)
    time.sleep(2)

    for bin_name, coords in BIN_LOCATIONS.items():
        x, y, z = coords
        print(f"\n--- Calibrating {bin_name} ---")
        print(f"Moving to: X={x}, Y={y}, Z={z}")
        print("Please place the real bin under the gripper now!")

        # הזזת הרובוט למיקום הפח
        result = ik.setPitchRangeMoving((x, y, z), -90, -90, 0, 2000)
        
        if not result:
            print("Error: Unreachable point!")
        else:
            # המתנה של 10 שניות כדי שיהיה לך זמן למקם את הפח
            for i in range(10, 0, -1):
                print(f"Hold position... {i}")
                time.sleep(1)
        
        # חזרה לבית לפני הפח הבא
        ik.setPitchRangeMoving((0, 10, 15), 0, -90, 0, 1500)
        time.sleep(1)

    print("\nCalibration finished!")

if __name__ == "__main__":
    main()
