# ---------- COMPLETE DETECTION CODE WITH QUEUED SOUND ALERTS ----------
import torch
import cv2
import os
import time
import sys
import numpy as np
from pygame import mixer  # For sound playback

def classify_obstacle(x1, y1, x2, y2, conf, frame_width):
    """Differentiate using size and aspect ratio"""
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    
    # Calculate dimensions relative to frame size
    rel_width = width / frame_width
    aspect_ratio = width / height if height != 0 else 0
    
    # Pothole: Small and square-like
    if rel_width < 0.15 and 0.7 < aspect_ratio < 1.3:
        return "pothole", conf*1.1, (0, 255, 0)  # Green
    
    # Speed breaker: Wide and large
    elif rel_width > 0.25 and aspect_ratio > 2.0:
        return "speed_breaker", conf*0.9, (0, 0, 255)  # Red
    
    else:  # Unknown
        return "pothole", conf*0.5, (0, 255, 0)  # Yellow

# Initialize device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device.upper()}")

# Initialize sound mixer
mixer.init()
sound_cooldown = 1.5  # Seconds between sound alerts
last_alert = {'pothole': 0, 'speed_breaker': 0}
sound_queue = []  # Queue for managing sound playback

# Load sound files
sound_files = {
    'pothole': 'pothole_detected.mp3',
    'speed_breaker': 'speedbreaker_detected.mp3'
}

# Load sounds as Sound objects
sound_objects = {}
for name, path in sound_files.items():
    if os.path.exists(path):
        sound_objects[name] = mixer.Sound(path)
    else:
        print(f"Warning: Sound file not found for {name} - {path}")
        sound_objects[name] = None

# Create a dedicated channel for alerts
alert_channel = mixer.Channel(0)

# Load model
model_path = r'D:\Pothole&SpeedBreakerDetection\trainedbest\best6(3).pt'
if not os.path.exists(model_path):
    print(f"Model not found at {model_path}")
    sys.exit(1)

try:
    model = torch.hub.load('ultralytics/yolov5', 'custom', 
                          path=model_path, 
                          force_reload=True).to(device).eval()
    model.conf = 0.5  # Confidence threshold
    model.iou = 0.45  # NMS IoU threshold
except Exception as e:
    print(f"Model load error: {e}")
    sys.exit(1)

# Video setup
video_path = r'D:\Pothole&SpeedBreakerDetection\Test_Videos\WhatsApp Video 2024-12-02 at 16.12.02_cd478263.mp4'
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Failed to open video: {video_path}")
    sys.exit(1)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps) if fps > 0 else 30

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Skip every 2 frames for faster processing
    if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % 3 != 0:
        continue

    # Detection
    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()

    current_time = time.time()
    detected_types = set()

    for det in detections:
        x1, y1, x2, y2, conf, _ = det.astype(float)
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        
        label, adj_conf, color = classify_obstacle(x1, y1, x2, y2, conf, frame_width)
        
        if adj_conf > 0.45:
            detected_types.add(label)
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Text background
            label_text = f"{label} {adj_conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label_text, 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1-th-10), 
                         (x1+tw, y1), color, -1)
            
            # Label
            cv2.putText(frame, label_text, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                       (255, 255, 255), 2)

    # Manage sound queue
    for obstacle in detected_types:
        if obstacle in sound_objects and sound_objects[obstacle] is not None:
            # Check cooldown and add to queue
            if current_time - last_alert[obstacle] > sound_cooldown:
                sound_queue.append(obstacle)
                last_alert[obstacle] = current_time

    # Play sounds from queue
    if not alert_channel.get_busy() and sound_queue:
        next_obstacle = sound_queue.pop(0)
        alert_channel.play(sound_objects[next_obstacle])

    # Show FPS
    cv2.imshow('Detection', frame)
    if cv2.waitKey(delay) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




