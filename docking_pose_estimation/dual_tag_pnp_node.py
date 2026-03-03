#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import numpy as np
import cv2

class DualTagPnPNode(Node):
    def __init__(self):
        super().__init__("dual_tag_pnp")

        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image,
            "/camera/image_raw",
            self.image_callback,
            10
        )

        self.pose_pub = self.create_publisher(PoseStamped, "/docking/pose", 10)

        # Camera intrinsics 
        self.K = np.array([[600, 0, 320],
                           [0, 600, 240],
                           [0,   0,   1]], dtype=np.float32)
        self.dist = np.zeros(5)

        # AprilTag points (8 corners)
        TAG_SIZE = 56 ##mm
        h = TAG_SIZE/2
        cx1, cy1 = -101, 0
        cx2, cy2 = 101, 0
        self.obj_points = np.array([
            # Tag 1 corners
            [cx1 - h, cy1 - h, 0.0],
            [cx1 + h, cy1 - h, 0.0], 
            [cx1 + h, cy1 + h, 0.0],
            [cx1 - h, cy1 + h, 0.0], 
            
            # Tag 2 corners 
            [cx2 - h, cy2 - h, 0.0],
            [cx2 + h, cy2 - h, 0.0],
            [cx2 + h, cy2 + h, 0.0],
            [cx2 - h, cy2 + h, 0.0],
        ], dtype=np.float32)

        self.detector = cv2.apriltag.AprilTagDetector()

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detections = self.detector.detect(gray)
        if len(detections) < 2:
            return

        detections = sorted(detections, key=lambda d: d.id)

        image_points = []
        for det in detections[:2]:
            image_points.extend(det.corners)

        image_points = np.array(image_points, dtype=np.float32)

        success, rvec, tvec = cv2.solvePnP(
            self.object_points,
            image_points,
            self.K,
            self.dist,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return

        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "camera_frame"

        pose_msg.pose.position.x = float(tvec[0])
        pose_msg.pose.position.y = float(tvec[1])
        pose_msg.pose.position.z = float(tvec[2])

        R, _ = cv2.Rodrigues(rvec)
        qw = np.sqrt(1 + np.trace(R)) / 2
        qx = (R[2,1] - R[1,2]) / (4*qw)
        qy = (R[0,2] - R[2,0]) / (4*qw)
        qz = (R[1,0] - R[0,1]) / (4*qw)

        pose_msg.pose.orientation.w = qw
        pose_msg.pose.orientation.x = qx
        pose_msg.pose.orientation.y = qy
        pose_msg.pose.orientation.z = qz

        self.pose_pub.publish(pose_msg)

def main(args=None):
    rclpy.init(args=args)
    node = DualTagPnPNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
