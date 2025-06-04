import torch
from pathlib import Path
import cv2
import time

# Load the model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='D:\\Pothole_&_Speedbreaker_detection\\yolov5\\best.pt').half()  # Enable half precision

# Load the video
video_path = 'D:\\Pothole_&_Speedbreaker_detection\\input videos\\VID_20241202150401.mp4'
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Set parameters
skip_frames = 1  # Process every nth frame (adjust as needed)
confidence_threshold = 0.5  # Minimum confidence for detections

# Process video frames
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Video is finished

    # Skip frames based on the skip_frames parameter
    if frame_count % (skip_frames + 1) == 0:
        # Resize frame (optional)
        frame = cv2.resize(frame, (640, 480))  # Adjust size as needed

        # Perform inference on the frame
        start_time = time.time()
        results = model(frame)
        print(f"Frame processed in {time.time() - start_time:.2f} seconds")

        # Get results
        results_df = results.pandas().xyxy[0]  # Extract predictions as a DataFrame
        results_df = results_df[results_df['confidence'] > confidence_threshold]  # Filter by confidence

        # Draw bounding boxes on the frame
        frame_with_boxes = results.render()[0]

        # Show the frame with bounding boxes
        cv2.imshow('Frame', frame_with_boxes)

    frame_count += 1

    # Press 'q' to exit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
