import cv2
from ultralytics import YOLO
import numpy as np


model=YOLO('best.pt')
cap=cv2.VideoCapture('ornekvid.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #img = cv2.resize(frame, (640, 640))
    #img = np.expand_dims(img, axis=0)
   # img = img / 255.0
    results = model(img)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            confidence = box.conf[0]
            if confidence >= 0.2:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cls = int(box.cls[0])
                label = f"{model.names[cls]} {confidence:.2f}"
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.imshow('deneme',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
