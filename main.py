import cv2
import torch
from ultralytics import YOLO
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import warnings
import math
import serial
import crchesap
import pan
import codecs

import tilt

warnings.filterwarnings("ignore", category=FutureWarning)


ser = serial.Serial('COM5', 9600, timeout=1)
PTZ_COMMANDS = {
    'PAN_TİLT_OPEN': b'\x55\x10\\{x1}\xFF',
    'PAN_LEFT10': b'\x55\x01\x02\x1c\x07\x10\x00\xc3\x50\x00',
    'PAN_RIGHT': b'\x01\x01\xFF',
    'TILT_UP': b'\x01\x02\xFF',
    'TILT_DOWN': b'\x01\x03\xFF'
}

def send_ptz_command(command):
    ser.write(command)


# Kamera parametreleri
sensor_width = 7.24  # mm (sensör genişliği)
sensor_height = 5.43  # mm (sensör yüksekliği)
focal_length = 73  # mm (odak uzunluğu) 4.44-142.6mm
image_width = 1920  # piksel (sensör çözünürlüğü genişlik)
image_height = 1080  # piksel (sensör çözünürlüğü yükseklik)
iha_length=5   #kanat açıklığı metre






#im_w=2*focal_length*math.tan(math.radians(h_fov/2))
#im_h=2*focal_length*math.tan(math.radians(v_fov/2))



def calculate_speedX(x,max_speed=70):

    if x>100 or x< -100:
        speed=max_speed
    else:
        speed=(x/100)*max_speed
        if speed>max_speed:
            speed=max_speed
    byte1 = codecs.decode(pan.pan_hareket(speed * 1000).replace(r'\x', ''), 'hex')
    a=crchesap.format_data_with_crc(byte1)

    return send_ptz_command(a)


def calculate_speedY(x,max_speed=70):

    if x>100 or x< -100:
        speed=max_speed
    else:
        speed=(x/100)*max_speed
        if speed>max_speed:
            speed=max_speed
    byte2 = codecs.decode(tilt.tilt_hareket(speed * 1000).replace(r'\x', ''), 'hex')
    b=crchesap.format_data_with_crc(byte2)

    return send_ptz_command(b)




class CentroidTracker:
    def __init__(self, maxDisappeared=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1

                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        return self.objects


model = YOLO('yolov9c.pt')


video = 'ornekvid.mp4'
cap = cv2.VideoCapture(video)



fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter('C:/Users/BerkeSEVİM/video/vid.mp4', fourcc, fps, (width, height))

tracker = CentroidTracker(maxDisappeared=40)

while cap.isOpened():
    ret, frame = cap.read()
    #frame = cv2.resize(frame, (640, 640))
    if not ret:
        break

    kx=int(640/width)
    ky=int(640/height)
    results = model.predict(frame)

    rects = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            confidence = box.conf[0]
            if confidence >= 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                rects.append((int(x1), int(y1),int(x2),int(y2)))
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
                cls = int(box.cls[0])
                label = f"{model.names[cls]} "
                cv2.putText(frame, label, (x1, (y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    objects = tracker.update(rects)


    cv2.circle(frame,(320,320),4,(255,0,0),-1)

    for (objectID, centroid) in objects.items():
        text = f"ID {objectID}"
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 0, 0), -1)

        mesafeX=(320-centroid[0])#*pixel_w
        mesafeY=(320-centroid[1])#*pixel_h
        mesafe=math.sqrt(mesafeX**2+mesafeY**2)

        calculate_speedX(mesafeX)
        calculate_speedY(mesafeY)

        #distance=(iha_length*focal_length)/(x2-x1)

        v_fov = 2 * math.degrees(math.atan(sensor_height / (2 * focal_length)))
        h_fov = 2 * math.degrees(math.atan(sensor_width / (2 * focal_length)))

        # piksel değeri mm
        P_w = 2*focal_length*math.tan(math.radians(h_fov/2))/image_width
        P_h = 2*focal_length*math.tan(math.radians(v_fov/2))/image_height

        R_w=P_w*(x2-x1)#yatay uzunluk

        #distance=(image_width*focal_length/(R_w*sensor_width))/1000

        distance=(R_w*focal_length*image_width )/(1000*sensor_width) #metre cinsinden uzaklık






        #distanc_w=iha_length/(2 * math.tan(math.radians(object_angle_width / 2)))
        #distance_h=4/(2 * math.tan(math.radians(object_angle_height / 2)))

        #distance=(distance_h+distanc_w)/2



        cv2.line(frame, (320, 320), (centroid[0],centroid[1]),(255,0,0),1)
        cv2.putText(frame,f'mesafe:{int (mesafe)}',(330,320),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
        cv2.putText(frame,f'distance: {int(distance)}m',(120,120),cv2.FONT_HERSHEY_SIMPLEX,0.5,(100,255,0),1)


    cv2.imshow('frame',frame)
    #out.write(frame)
    cv2.waitKey(1)



cap.release()
out.release()
cv2.destroyAllWindows()