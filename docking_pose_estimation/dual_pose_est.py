#!/usr/bin/env python3
"""Docking Pose Estimation"""

import cv2
import numpy as np
import apriltag
import time
from math import atan2

from test import Motor


# -----------------------------
# Camera + Tag Configuration
# -----------------------------
TAG_SIZE = 0.06
TARGET_ID = 0

# Camera matrix from video.py
CAMERA_MATRIX = np.array([
        [628.8233032226562, 0, 646.732666015625],
        [0, 628.3672485351562, 364.4508361816406],
        [0, 0, 1]
    ], dtype=np.float32)
DIST_COEFFS =  np.array([-0.056777212768793106, 0.06796900182962418, 0.0007022436475381255, 0.0004860123444814235], dtype=np.float32)


# -----------------------------
# AprilTag Pose Estimator
# -----------------------------
class AprilTagPose:
    def __init__(self):
        self.detector = apriltag.Detector(
            apriltag.DetectorOptions(families="tag25h9")
        )

        # 3D coordinates of tag corners in tag frame
        s = TAG_SIZE / 2
        self.obj_points = np.array([
            [-s, -s, 0],
            [ s, -s, 0],
            [ s,  s, 0],
            [-s,  s, 0]
        ], dtype=np.float32)

    def detect_pose(self, gray):
        detections = self.detector.detect(gray)
        if len(detections) == 0:
            return None, None, None

        for det in detections:
            if det.tag_id == TARGET_ID:
                img_pts = np.array(det.corners, dtype=np.float32)

                success, rvec, tvec = cv2.solvePnP(
                    self.obj_points,
                    img_pts,
                    CAMERA_MATRIX,
                    DIST_COEFFS,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE
                )

                if success:
                    return det, rvec, tvec

        return None, None, None


# -----------------------------
# Main 
# -----------------------------
def main():
    print("Starting Pose-PD controller...")

    # Initialize servo
    motor = Motor(servo_ids=[1], port="/dev/ttyUSB0")
    time.sleep(0.5)

    # Random initial speed
    motor.set_motor_speed([300])
    time.sleep(1)
    motor.set_motor_speed([0])

    # Initialize AprilTag detector
    tag_estimator = AprilTagPose()
    pd = PDController(kp=2.0, kd=0.3)

    # Camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        det, rvec, tvec = tag_estimator.detect_pose(gray)

        if det is not None:
            #target roll angle
            TARGET_ROLL = 0.0
            
            # Convert rotation vector to rotation matrix
            R, _ = cv2.Rodrigues(rvec)

            # Extract roll angle from rotation matrix
            roll = atan2(R[2, 1], R[1, 1])

            # PD control: want roll → 0
            error = roll - TARGET_ROLL
            control = pd.compute(error)

            # Convert to servo speed
            speed = int(np.clip(-control * 300, -800, 800))

            motor.set_motor_speed([speed])

            cv2.putText(frame, f"Roll: {roll:.3f} rad", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(frame, f"Speed: {speed}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        else:
            motor.set_motor_speed([0])

        cv2.imshow("Pose-PD", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    motor.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()