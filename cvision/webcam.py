import cv2
from ultralytics import YOLO

import cvzone

cap = cv2.VideoCapture(2)
cap.set(3, 640)
cap.set(4, 480)

model = YOLO("Weights/yolov8n.pt")


while True:
    success, img = cap.read()
    results = model(img, stream=True)  #stream=True - uses generators which will be more efficient.
    # check the individual bounding boxes
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # x1, y1, x2, y2 = box.xyxy[0]
            # x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

               #alt
            x1, y1, w, h = box.xywh[0]
            bbox = int(x1), int(y1), int(w), int(h)
            cvzone.cornerRect(img, bbox)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
