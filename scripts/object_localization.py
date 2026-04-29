#!/usr/bin/env python3
"""
G1 Robot Camera Streaming with Object Detection

Real-time object detection and 3D positioning using SDK2 camera data.
Subscribes to camera streams from unitree_mujoco.py simulation.
"""

import sys
import time
import cv2
import numpy as np

# SDK2 for camera data subscription
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_go.msg.dds_ import AudioData_


class CameraSubscriber:
    """Subscribes to SDK2 camera data and processes for object detection"""

    def __init__(self):
        """Initialize SDK2 camera subscriber"""
        self.width, self.height = 640, 480

        # Image storage
        self.rgb_image = None
        self.depth_image = None
        self.metric_depth_image = None  # Stores metric depth in meters
        self.rgb_frame_count = 0
        self.depth_frame_count = 0

        # Depth camera parameters for metric conversion (matches publisher settings)
        self.depth_near_plane = 0.1   # Matches publisher near plane
        self.depth_far_plane = 10.0    # Matches publisher far plane (increased for less banding)

        # Setup SDK2 communication
        self.setup_sdk2_subscribers()

    def setup_sdk2_subscribers(self):
        """Setup SDK2 communication to receive camera data from simulation"""
        try:
            # Initialize SDK2 with same domain as the simulation
            ChannelFactoryInitialize(0, "lo")

            # Subscribe to camera feeds using AudioData_ messages
            self.rgb_sub = ChannelSubscriber("rt/camera/rgb", AudioData_)
            self.depth_sub = ChannelSubscriber("rt/camera/depth", AudioData_)

            self.rgb_sub.Init(self.rgb_handler, 10)
            self.depth_sub.Init(self.depth_handler, 10)

            print("✓ SDK2 camera subscribers initialized")

        except Exception as e:
            print(f"❌ SDK2 camera setup failed: {e}")
            print("Make sure unitree_mujoco.py is running with camera support")
            sys.exit(1)

    def rgb_handler(self, msg: AudioData_):
        """Handle RGB camera messages from SDK2 using AudioData_"""
        try:
            if len(msg.data) > 0:

                if self.rgb_frame_count == 0:
                    print(f"[DEBUG] RGB message received: {len(msg.data)} bytes")
                    print(f"[DEBUG] Expected: {self.width * self.height * 3} bytes")

                # Convert list of integers to numpy array (AudioData_.data is sequence[uint8])
                img_array = np.array(msg.data, dtype=np.uint8)

                # Reshape to image dimensions (height, width, channels)
                if len(img_array) >= self.height * self.width * 3:
                    self.rgb_image = img_array[:self.height * self.width * 3].reshape(
                        (self.height, self.width, 3))
                    self.rgb_frame_count += 1
                else:
                    if self.rgb_frame_count == 0:
                        print(
                            f"[WARN] RGB data size mismatch: got",
                            f"{len(img_array)}",
                            f"'need' {self.height * self.width * 3}")

        except Exception as e:
            print(f"RGB handler error: {e}")

    def depth_handler(self, msg: AudioData_):
        """Handle depth camera messages from SDK2 using AudioData_"""
        try:
            if len(msg.data) > 0:
                # Convert list of integers to numpy array (AudioData_.data is sequence[uint8])
                img_array = np.array(msg.data, dtype=np.uint8)

                # Depth is single channel
                if len(img_array) >= self.height * self.width:
                    self.depth_image = img_array[:self.height * self.width].reshape(
                        (self.height, self.width))

                    # Convert uint8 back to normalized depth [0.0, 1.0]
                    normalized_depth = self.depth_image.astype(np.float32) / 255.0

                    # Convert normalized depth to metric depth in meters
                    self.metric_depth_image = (self.depth_near_plane +
                                               (self.depth_far_plane - self.depth_near_plane) * normalized_depth)

                    self.depth_frame_count += 1

        except Exception as e:
            print(f"Depth handler error: {e}")

    def detect_object(self, image):
        """Detect red object in the image and return bounding box"""
        try:
            # Convert RGB to BGR for OpenCV
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image

            # Convert to HSV and create red color mask
            hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
            lower_red1 = np.array([0, 100, 100])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([160, 100, 100])
            upper_red2 = np.array([180, 255, 255])

            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask = cv2.bitwise_or(mask1, mask2)

            # Find contours and return largest valid one
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_contours = [c for c in contours if cv2.contourArea(c) > 100]

            if valid_contours:
                largest_contour = max(valid_contours, key=cv2.contourArea)
                return cv2.boundingRect(largest_contour)

        except Exception as e:
            print(f"Error in object detection: {e}")

        return None

    def get_object_3d_position(self, rgb_image, metric_depth_image):
        """Calculate object 3D position using RGB detection and metric depth"""
        try:
            bbox = self.detect_object(rgb_image)
            if bbox is None or metric_depth_image is None:
                return None

            x, y, w, h = bbox
            center_x = x + w // 2
            center_y = y + h // 2
            depth_meters = metric_depth_image[center_y, center_x]

            # Simple pinhole camera model
            fx = fy = 459  # Focal length approximation
            cx, cy = self.width / 2, self.height / 2

            if 0 < depth_meters < self.depth_far_plane:
                cam_x = (center_x - cx) * depth_meters / fx
                cam_y = (center_y - cy) * depth_meters / fy
                cam_z = depth_meters
                return np.array([cam_x, cam_y, cam_z])

        except Exception as e:
            print(f"Error calculating 3D position: {e}")

        return None

    def process_and_display_frame(self, image, camera_name):
        """Process frame for object detection and add annotations"""
        if image is None:
            return None

        try:
            # Convert to BGR for OpenCV display
            if len(image.shape) == 3:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                # Depth image - convert to 3-channel for display
                image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            # Detect object and draw bounding box + 3D position on both RGB and depth
            if camera_name == "RGB" and len(image.shape) == 3:
                # RGB image processing
                bbox = self.detect_object(image)
                if bbox is not None:
                    x, y, w, h = bbox
                    # Draw bounding box
                    cv2.rectangle(image_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Calculate and display 3D position on RGB image
                    if self.metric_depth_image is not None:
                        object_pos_3d = self.get_object_3d_position(image, self.metric_depth_image)
                        if object_pos_3d is not None:
                            # Display 3D coordinates on image (small font)
                            pos_text = (
                                f"3D: [{object_pos_3d[0]:.2f}, "
                                f"{object_pos_3d[1]:.2f}, "
                                f"{object_pos_3d[2]:.2f}]m"
                            )

                            cv2.putText(image_bgr, pos_text, (x, y - 25),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                            cv2.putText(image_bgr, "Object", (x, y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        else:
                            cv2.putText(image_bgr, "Object", (x, y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            elif camera_name == "Depth":
                # Depth image processing - overlay same detection from RGB
                if self.rgb_image is not None:
                    bbox = self.detect_object(self.rgb_image)
                    if bbox is not None:
                        x, y, w, h = bbox
                        # Draw bounding box on depth image
                        # Draw a + at the center of the bounding box
                        center_x, center_y = x + w // 2, y + h // 2
                        cv2.line(image_bgr, (center_x - 2, center_y), (center_x +
                                 2, center_y), (255, 255, 0), 2)  # horizontal line
                        cv2.line(image_bgr, (center_x, center_y - 2), (center_x,
                                 center_y + 2), (255, 255, 0), 2)  # vertical line

                        # Calculate and display 3D position on depth image
                        if self.metric_depth_image is not None:
                            object_pos_3d = self.get_object_3d_position(
                                self.rgb_image, self.metric_depth_image)
                            if object_pos_3d is not None:
                                # Display 3D coordinates on depth image (small font)
                                pos_text = (
                                    f"3D: [{object_pos_3d[0]:.2f}, "
                                    f"{object_pos_3d[1]:.2f}, "
                                    f"{object_pos_3d[2]:.2f}]m"
                                )

                                cv2.putText(image_bgr, pos_text, (x, y - 25),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                                cv2.putText(image_bgr, "Object", (x, y - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            return image_bgr

        except Exception as e:
            print(f"Error processing frame: {e}")
            return None

    def stream_cameras(self):
        """Main streaming loop displaying camera feeds with object detection"""
        print(f"\n🎥 Streaming camera data from simulation")
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Save current frames")
        print("  SPACE - Pause/Resume")

        # Create windows
        cv2.namedWindow("G1 Camera - RGB", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("G1 Camera - Depth", cv2.WINDOW_AUTOSIZE)

        paused = False
        save_count = 0
        last_status_time = 0
        data_received = False

        try:
            while True:
                current_time = time.time()

                # Check if we're receiving data
                if not data_received and (
                        self.rgb_image is not None or self.depth_image is not None):
                    print("✓ Camera data received from simulation")
                    data_received = True

                if not paused:
                    # Process RGB image
                    if self.rgb_image is not None:
                        display_rgb = self.process_and_display_frame(self.rgb_image, "RGB")
                        if display_rgb is not None:
                            # Add frame info
                            info_text = f"RGB | Frame: {self.rgb_frame_count}"
                            cv2.putText(display_rgb, info_text, (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            cv2.imshow("G1 Camera - RGB", display_rgb)

                    # Process depth image
                    if self.depth_image is not None:
                        display_depth = self.process_and_display_frame(self.depth_image, "Depth")
                        if display_depth is not None:
                            # Add frame info
                            info_text = f"Depth | Frame: {self.depth_frame_count}"
                            cv2.putText(display_depth, info_text, (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                            cv2.imshow("G1 Camera - Depth", display_depth)

                    # Print detection status every 5 seconds
                    if (current_time - last_status_time > 5.0 and self.rgb_image is not None):
                        bbox = self.detect_object(self.rgb_image)
                        if bbox is not None:
                            if self.metric_depth_image is not None:
                                object_pos_3d = self.get_object_3d_position(
                                    self.rgb_image, self.metric_depth_image)
                                if object_pos_3d is not None:
                                    print(f"🎯 Object Detected")
                                else:
                                    print("❌ Object detected but 3D calculation failed")
                            else:
                                print("⚠️ Object detected but no depth data available")
                        else:
                            print("🔍 No object detected")

                        last_status_time = current_time

                # Handle keyboard input
                key = cv2.waitKey(30) & 0xFF

                if key == ord('q'):
                    break
                elif key == ord(' '):  # Space bar
                    paused = not paused
                    status = "PAUSED" if paused else "RESUMED"
                    print(f"Stream {status}")
                elif key == ord('s'):  # Save frames
                    if self.rgb_image is not None:
                        filename_rgb = f"rgb_frame_{save_count:04d}.jpg"
                        cv2.imwrite(filename_rgb, cv2.cvtColor(self.rgb_image, cv2.COLOR_RGB2BGR))
                        print(f"💾 RGB frame saved: {filename_rgb}")

                    if self.depth_image is not None:
                        filename_depth = f"depth_frame_{save_count:04d}.jpg"
                        cv2.imwrite(filename_depth, self.depth_image)
                        print(f"💾 Depth frame saved: {filename_depth}")

                    save_count += 1

                time.sleep(0.01)  # Small delay

        except KeyboardInterrupt:
            print("\n⏹️  Stream stopped by user")

        cv2.destroyAllWindows()


def main():
    """Main function"""
    print("🤖 G1 Robot Camera Streaming")
    print("📡 Real-time object detection with 3D positioning")

    try:
        camera_sub = CameraSubscriber()
        camera_sub.stream_cameras()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure unitree_mujoco.py is running first")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
