import cv2
from ultralytics import YOLO

import cvzone
import math

# cap = cv2.VideoCapture(2)
# cap.set(3, 1280)
# cap.set(4, 720)

cap = cv2.VideoCapture()


model = YOLO("/home/abok/Documents/Fiverr/Hari/cv/Yolo-Weights/yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", 
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", 
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", 
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", 
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]


while True:
    success, img = cap.read()
    results = model(img, stream=True)  #stream=True - uses generators which will be more efficient.
    # check the individual bounding boxes
    for r in results:
        boxes = r.boxes
        for box in boxes:

            # Bounding box 
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

               #alt
            # x1, y1, w, h = box.xywh[0]
            # bbox = int(x1), int(y1), int(w), int(h)

            w, h = x2-x1, y2-y1
            cvzone.cornerRect(img, (x1, y1, w, h))
            # cv2.rectangle(img, bbox[:2], bbox[2:], (255, 0, 255), 3)
            
            # Confidence
            conf = math.ceil((box.conf[0])*100)/100
            # print(conf)

            # class name
            cls = box.cls[0]

            cvzone.putTextRect(img, f'{classNames[int(cls)]} {conf}', (max(0, x1), max(70, y1)), scale=1, thickness=1)
             


    cv2.imshow("Image", img)
    cv2.waitKey(1)
