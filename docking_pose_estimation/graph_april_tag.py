#!/usr/bin/env python3

import cv2
import apriltag
import numpy as np
from scipy.spatial.transform import Rotation
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, PointStamped

import matplotlib.pyplot as plt
plt.ion()


class AprilTagPoseEstimator(Node):
    def __init__(self, camera_matrix, dist_coeffs, tag_size, target_id):
        super().__init__('apriltag_pose_estimator')

        self.frame_id = 0
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.tag_size = tag_size
        self.target_id = target_id

        self.detector = apriltag.Detector(
            apriltag.DetectorOptions(families="tag25h9")
        )

        self.tracked_points = []
        self.bridge = CvBridge()

        # ROS publishers/subscribers
        self.pose_pub = self.create_publisher(PoseStamped, '/mavros/mocap/pose', 10)
        self.tag_center_pub = self.create_publisher(PointStamped, '/ugv/tag_center', 10)
        self.image_pub = self.create_publisher(Image, '/processed_image', 10)
        self.image_sub = self.create_subscription(Image, '/camera/color/image_raw', self.image_callback, 10)

        # Timestamp tracking for real FPS
        self.prev_stamp = None
        self.frame_intervals = []

        # Video writer (initialized later)
        self.video_out = None
        self.video_initialized = False
        self.real_fps = None

        # Single-tag object points
        self.single_tag_obj = np.array([
            [-tag_size/2, -tag_size/2, 0],
            [ tag_size/2, -tag_size/2, 0],
            [ tag_size/2,  tag_size/2, 0],
            [-tag_size/2,  tag_size/2, 0],
        ], dtype=np.float32)

        # Logging for plots
        self.time_log = []
        self.roll_log = []
        self.pitch_log = []
        self.yaw_log = []
        self.x_log = []
        self.y_log = []
        self.z_log = []

        # Create figure
        self.fig, (self.ax_rpy, self.ax_xyz) = plt.subplots(2, 1, figsize=(8, 8))
        self.fig.tight_layout()

        self.get_logger().info("AprilTag Pose Estimator Initialized")

    def init_video_writer(self, frame, fps):
        height, width, _ = frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_out = cv2.VideoWriter(
            '/home/grlopi/output.mp4',
            fourcc,
            fps,
            (width, height)
        )
        self.video_initialized = True
        self.get_logger().info(f"VideoWriter initialized at {width}x{height}, {fps:.2f} FPS")

    def update_plots(self):
        """Update live plots every few frames."""
        if len(self.time_log) < 5:
            return

        self.ax_rpy.clear()
        self.ax_xyz.clear()

        # RPY plot
        self.ax_rpy.plot(self.time_log, self.roll_log, label="Roll (deg)")
        self.ax_rpy.plot(self.time_log, self.pitch_log, label="Pitch (deg)")
        self.ax_rpy.plot(self.time_log, self.yaw_log, label="Yaw (deg)")
        self.ax_rpy.set_ylabel("Angle (deg)")
        self.ax_rpy.legend()
        self.ax_rpy.grid(True)

        # XYZ plot
        self.ax_xyz.plot(self.time_log, self.x_log, label="X (m)")
        self.ax_xyz.plot(self.time_log, self.y_log, label="Y (m)")
        self.ax_xyz.plot(self.time_log, self.z_log, label="Z (m)")
        self.ax_xyz.set_xlabel("Time (s)")
        self.ax_xyz.set_ylabel("Position (m)")
        self.ax_xyz.legend()
        self.ax_xyz.grid(True)

        plt.pause(0.001)

    def image_callback(self, msg):
        display_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        # Track timestamps for real FPS
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.prev_stamp is not None:
            dt = stamp - self.prev_stamp
            if dt > 0:
                self.frame_intervals.append(dt)
        self.prev_stamp = stamp

        # Run AprilTag detection + drawing
        gray = cv2.cvtColor(display_image, cv2.COLOR_BGR2GRAY)
        self.detect_and_plot(gray, display_image, stamp)

        # Initialize video writer once we have enough timestamps
        if not self.video_initialized and len(self.frame_intervals) > 10:
            avg_dt = sum(self.frame_intervals) / len(self.frame_intervals)
            self.real_fps = 1.0 / avg_dt
            self.init_video_writer(display_image, self.real_fps)

        # Write frame to video
        if self.video_initialized:
            self.video_out.write(display_image)

        # Update plots every 5 frames
        if self.frame_id % 5 == 0:
            self.update_plots()

        self.frame_id += 1

    def detect_and_plot(self, cv_image, display_image, stamp):
        cv_image = cv2.GaussianBlur(cv_image, (5, 5), 0)
        detections = self.detector.detect(cv_image)

        if len(detections) == 0:
            return

        detections_sorted = sorted(detections, key=lambda d: d.tag_id)

        if len(detections_sorted) == 1:
            img_points = detections_sorted[0].corners.astype(np.float32)
            self.publish_pose_single(img_points, detections_sorted[0].tag_id, display_image, stamp)

        else:
            img_points = np.vstack([
                detections_sorted[0].corners,
                detections_sorted[1].corners
            ]).astype(np.float32)
            self.publish_pose_dual(img_points, display_image, stamp)

        self.display_trajectory(display_image)

        cv2.imshow("Pose Estimation", display_image)
        cv2.waitKey(1)

    def log_pose(self, rotation_matrix, tvec, stamp):
        """Extract roll/pitch/yaw and log everything."""
        r = Rotation.from_matrix(rotation_matrix)
        roll, pitch, yaw = r.as_euler('xyz', degrees=True)

        self.time_log.append(stamp)
        self.roll_log.append(roll)
        self.pitch_log.append(pitch)
        self.yaw_log.append(yaw)
        self.x_log.append(float(tvec[0]))
        self.y_log.append(float(tvec[1]))
        self.z_log.append(float(tvec[2]))

    def publish_pose_single(self, img_points, tag_id, display_image, stamp):
        center_x = np.mean(img_points[:, 0])
        center_y = np.mean(img_points[:, 1])
        self.tracked_points.append((int(center_x), int(center_y)))

        success, rvec, tvec = cv2.solvePnP(
            self.single_tag_obj,
            img_points,
            self.camera_matrix,
            self.dist_coeffs
        )

        if not success:
            return

        rotation_matrix, _ = cv2.Rodrigues(rvec)

        # --- AXIS FLIP FIX ---
        z_axis = rotation_matrix[:, 2]
        if z_axis[2] > 0:
            R_flip = np.array([
                [1,  0,  0],
                [0, -1,  0],
                [0,  0, -1]
            ])
            rotation_matrix = rotation_matrix @ R_flip
            rvec, _ = cv2.Rodrigues(rotation_matrix)
        # ----------------------

        # Log pose for plotting
        self.log_pose(rotation_matrix, tvec, stamp)

        self.draw_axes(display_image, rvec, tvec)

        quaternion = Rotation.from_matrix(rotation_matrix).as_quat()

        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "map"
        pose_msg.pose.position.x = float(tvec[0])
        pose_msg.pose.position.y = float(tvec[1])
        pose_msg.pose.position.z = float(tvec[2])
        pose_msg.pose.orientation.x = float(quaternion[0])
        pose_msg.pose.orientation.y = float(quaternion[1])
        pose_msg.pose.orientation.z = float(quaternion[2])
        pose_msg.pose.orientation.w = float(quaternion[3])
        self.pose_pub.publish(pose_msg)

        center_msg = PointStamped()
        center_msg.header = pose_msg.header
        center_msg.point.x = float(center_x)
        center_msg.point.y = float(center_y)
        center_msg.point.z = 0.0
        self.tag_center_pub.publish(center_msg)

    def publish_pose_dual(self, img_points, display_image, stamp):
        center_x = np.mean(img_points[:, 0])
        center_y = np.mean(img_points[:, 1])
        self.tracked_points.append((int(center_x), int(center_y)))

        TAG_SIZE = 0.056
        h = TAG_SIZE / 2
        cx1, cy1 = -0.101, 0
        cx2, cy2 = 0.101, 0

        obj_points = np.array([
            [cx1 - h, cy1 - h, 0.0],
            [cx1 + h, cy1 - h, 0.0],
            [cx1 + h, cy1 + h, 0.0],
            [cx1 - h, cy1 + h, 0.0],
            [cx2 - h, cy2 - h, 0.0],
            [cx2 + h, cy2 - h, 0.0],
            [cx2 + h, cy2 + h, 0.0],
            [cx2 - h, cy2 + h, 0.0],
        ], dtype=np.float32)

        success, rvec, tvec = cv2.solvePnP(
            obj_points, img_points, self.camera_matrix, self.dist_coeffs
        )

        if not success:
            return

        rotation_matrix, _ = cv2.Rodrigues(rvec)

        # --- AXIS FLIP FIX ---
        z_axis = rotation_matrix[:, 2]
        if z_axis[2] > 0:
            R_flip = np.array([
                [1,  0,  0],
                [0, -1,  0],
                [0,  0, -1]
            ])
            rotation_matrix = rotation_matrix @ R_flip
            rvec, _ = cv2.Rodrigues(rotation_matrix)
        # ----------------------

        # Log pose for plotting
        self.log_pose(rotation_matrix, tvec, stamp)

        self.draw_axes(display_image, rvec, tvec)

        quaternion = Rotation.from_matrix(rotation_matrix).as_quat()

        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "map"
        pose_msg.pose.position.x = float(tvec[0])
        pose_msg.pose.position.y = float(tvec[1])
        pose_msg.pose.position.z = float(tvec[2])
        pose_msg.pose.orientation.x = float(quaternion[0])
        pose_msg.pose.orientation.y = float(quaternion[1])
        pose_msg.pose.orientation.z = float(quaternion[2])
        pose_msg.pose.orientation.w = float(quaternion[3])
        self.pose_pub.publish(pose_msg)

        center_msg = PointStamped()
        center_msg.header = pose_msg.header
        center_msg.point.x = float(center_x)
        center_msg.point.y = float(center_y)
        center_msg.point.z = 0.0
        self.tag_center_pub.publish(center_msg)

    def draw_axes(self, image, rvec, tvec):
        axis_length = self.tag_size * 0.5
        axis_points_3d = np.array([
            [0, 0, 0],
            [axis_length, 0, 0],
            [0, axis_length, 0],
            [0, 0, axis_length]
        ], dtype=np.float32)

        axis_points_2d, _ = cv2.projectPoints(
            axis_points_3d, rvec, tvec, self.camera_matrix, self.dist_coeffs
        )
        axis_points_2d = axis_points_2d.reshape(-1, 2)

        origin = tuple(axis_points_2d[0].astype(int))
        x_end = tuple(axis_points_2d[1].astype(int))
        y_end = tuple(axis_points_2d[2].astype(int))
        z_end = tuple(axis_points_2d[3].astype(int))

        cv2.line(image, origin, x_end, (0, 0, 255), 3)
        cv2.line(image, origin, y_end, (0, 255, 0), 3)
        cv2.line(image, origin, z_end, (255, 0, 0), 3)

    def display_trajectory(self, image):
        if len(self.tracked_points) < 2:
            return

        for i in range(1, len(self.tracked_points)):
            cv2.line(image, self.tracked_points[i - 1], self.tracked_points[i], (0, 0, 255), 4)

        image_msg = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
        self.image_pub.publish(image_msg)


def main(args=None):
    rclpy.init(args=args)

    camera_matrix = np.array([
        [628.8233, 0, 646.7327],
        [0, 628.3672, 364.4508],
        [0, 0, 1]
    ], dtype=np.float32)

    dist_coeffs = np.array([
        -0.0567772, 0.0679690, 0.0007022, 0.0004860
    ], dtype=np.float32)

    tag_size = 0.3
    target_id = {0, 1}

    estimator = AprilTagPoseEstimator(camera_matrix, dist_coeffs, tag_size, target_id)

    try:
        rclpy.spin(estimator)
    except KeyboardInterrupt:
        pass
    finally:
        if estimator.video_out is not None:
            estimator.video_out.release()

        # Save final plots
        plt.savefig("/home/grlopi/pose_plots.png")

        if len(estimator.frame_intervals) > 0:
            avg_dt = sum(estimator.frame_intervals) / len(estimator.frame_intervals)
            real_fps = 1.0 / avg_dt
            print(f"\nEstimated real FPS from rosbag: {real_fps:.3f}\n")

        estimator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()