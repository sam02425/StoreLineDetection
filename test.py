import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import numpy as np

model=YOLO('yolov8n.pt')


def is_inside_polygon(point, polygon):
    return cv2.pointPolygonTest(np.array(polygon, np.int32), point, False) >= 0

count = 0
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap=cv2.VideoCapture('cr.mp4')

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

area1=[(544,12),(587,377),(713,372),(643,13)]
area2=[(763,17),(969,343),(1016,298),(924,16)]

confidence_threshold = 0.7  # Set the confidence threshold


while True:
    ret,frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    count += 1
    if count % 3 != 0:
        continue
    frame=cv2.resize(frame,(1020,500))
    results=model.predict(frame)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")

    count_area1 = 0
    count_area2 = 0

    for index, row in px.iterrows():
        confidence = row[4]  # Assuming the confidence score is at position 4
        if confidence < confidence_threshold:
            continue  # Skip this detection as it's below the confidence threshold

        x1, y1, x2, y2, _, d = row.astype(int)
        c = class_list[d]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if is_inside_polygon((cx, cy), area1):
            count_area1 += 1
            cvzone.cornerRect(frame, (x1, y1, x2-x1, y2-y1), 3, 2)
            cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
            cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)

        elif is_inside_polygon((cx, cy), area2):
            count_area2 += 1
            cvzone.cornerRect(frame, (x1, y1, x2-x1, y2-y1), 3, 2)
            cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
            cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)

    cv2.polylines(frame, [np.array(area1, np.int32)], True, (0, 0, 255), 2)
    cv2.polylines(frame, [np.array(area2, np.int32)], True, (0, 0, 255), 2)

    cvzone.putTextRect(frame, f'Area1 Count: {count_area1}', (50, 50), scale=1, thickness=2, colorR=(0,200,0), offset=20)
    cvzone.putTextRect(frame, f'Area2 Count: {count_area2}', (50, 80), scale=1, thickness=2, colorR=(0,200,0), offset=20)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()