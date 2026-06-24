#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2
import sys

class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher')
        
        # We declare a parameter for the camera index so it can be configured
        self.declare_parameter('device_index', 6) # Default to 6 for RealSense RGB
        self.declare_parameter('fps', 10.0)
        
        device_index = self.get_parameter('device_index').value
        fps = self.get_parameter('fps').value
        
        self.publisher_ = self.create_publisher(
            CompressedImage, 
            '/camera/image_raw/compressed', 
            10
        )
        
        self.get_logger().info(f"Opening video device at index {device_index}...")
        self.cap = cv2.VideoCapture(device_index)
        
        if not self.cap.isOpened():
            self.get_logger().warn(f"Failed to open device at index {device_index}. Trying index 0...")
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.get_logger().error("Failed to open any video capture device! Exiting.")
                sys.exit(1)
        
        # Set frame size to standard 640x480 for efficient image analysis processing
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        timer_period = 1.0 / fps
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.get_logger().info(f"Camera publisher node initialized. Publishing at {fps} FPS...")

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("Failed to read frame from camera.")
            return
            
        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_frame'
        msg.format = 'jpeg'
        
        # Compress the frame to JPEG to fit the CompressedImage format
        success, jpeg_data = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if success:
            msg.data = jpeg_data.tobytes()
            self.publisher_.publish(msg)
        else:
            self.get_logger().error("Failed to encode frame to JPEG.")

    def destroy_node(self):
        if self.cap.isOpened():
            self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = CameraPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
