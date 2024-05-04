import cv2
import numpy as np
import pyrealsense2 as rs

# Configure RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Convert depth frame to numpy array
        depth_image = np.asanyarray(depth_frame.get_data())

        # Convert color frame to OpenCV format
        color_image = np.asanyarray(color_frame.get_data())

        # Get distance at the center of the frame
        center_x, center_y = int(color_frame.width / 2), int(color_frame.height / 2)
        distance = depth_frame.get_distance(center_x, center_y)

        # Display the distance on the color image
        cv2.putText(color_image, f"Distance: {distance:.2f} meters", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Show images
        cv2.imshow('RealSense Depth and Color', color_image)

        # Break the loop on 'q' key press
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()
