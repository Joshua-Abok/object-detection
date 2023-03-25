from ultralytics import YOLO

import cv2

model = YOLO('weights/yolov8n.pt')
results = model("images/pexels-david-skyrius-2129796.jpg", show=True)

cv2.waitKey(0)