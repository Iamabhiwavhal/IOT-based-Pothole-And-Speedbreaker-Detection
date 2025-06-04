import torch
import cv2
import numpy as np
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.datasets import letterbox
from yolov5.utils.torch_utils import select_device

# ✅ Load YOLOv5 Model
MODEL_PATH = "D:/Pothole_&_Speedbreaker_Detection/yolov5/runs/train/exp11/weights/best.pt"
device = select_device('cpu')  # Use 'cuda' for GPU
model = attempt_load(MODEL_PATH, map_location=device)
stride = int(model.stride.max())
names = model.names

# ✅ Open Webcam (Change to 1 if using an external webcam)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Couldn't open webcam.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess Frame
    img = letterbox(frame, stride=stride, auto=True)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)

    # Convert to Tensor
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # ✅ Run YOLOv5 Inference
    with torch.no_grad():
        pred = model(img)[0]

    # ✅ Apply Non-Max Suppression
    pred = non_max_suppression(pred, 0.3, 0.5)

    # ✅ Process Detections
    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
