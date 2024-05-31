import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import cvzone

# Initialize a global dictionary to simulate tracking IDs for each detected class
tracking_ids = {}
next_id = 0

def is_inside_polygon(point, polygon):
    """Check if a point is inside a given polygon."""
    return cv2.pointPolygonTest(np.array(polygon, np.int32), point, False) >= 0

def setup_mouse_callback(window_name):
    """Setup mouse callback to print coordinates."""
    def on_mouse_move(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            print([x, y])
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse_move)

def read_class_list(file_path):
    """Read class names from a file."""
    with open(file_path, "r") as file:
        data = file.read()
    return data.split("\n")

def assign_tracking_id(class_name):
    """Assign a new tracking ID for a detected object."""
    global next_id
    if class_name in tracking_ids:
        return tracking_ids[class_name]
    else:
        tracking_ids[class_name] = next_id
        next_id += 1
        return tracking_ids[class_name]

def process_frame(frame, model, class_list, area1, area2, confidence_threshold):
    """Process each frame for detection, filtering, and counting."""
    results = model.predict(frame)
    detections = pd.DataFrame(results[0].boxes.data).astype("float")
    count_area1, count_area2 = 0, 0

    for _, detection in detections.iterrows():
        confidence = detection[4]
        if confidence < confidence_threshold:
            continue

        x1, y1, x2, y2, _, class_id = detection.astype(int)
        class_name = class_list[class_id]
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        tracking_id = assign_tracking_id(class_name)

        if is_inside_polygon((center_x, center_y), area1) or is_inside_polygon((center_x, center_y), area2):
            cvzone.cornerRect(frame, (x1, y1, x2-x1, y2-y1), 3, 2)
            cv2.circle(frame, (center_x, center_y), 4, (255, 0, 0), -1)
            cvzone.putTextRect(frame, f'{class_name} {tracking_id}', (x1, y1), 1, 1)

            if is_inside_polygon((center_x, center_y), area1):
                count_area1 += 1
            else:
                count_area2 += 1

    return frame, count_area1, count_area2

def main():
    model = YOLO('yolov8n.pt')
    setup_mouse_callback('RGB')
    cap = cv2.VideoCapture('cr.mp4')
    class_list = read_class_list("coco.txt")
    area1 = [(544,12), (587,377), (713,372), (643,13)]
    area2 = [(763,17), (969,343), (1016,298), (924,16)]
    confidence_threshold = 0.7

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame = cv2.resize(frame, (1020, 500))
        frame, count_area1, count_area2 = process_frame(frame, model, class_list, area1, area2, confidence_threshold)

        cv2.polylines(frame, [np.array(area1, np.int32)], True, (0, 0, 255), 2)
        cv2.polylines(frame, [np.array(area2, np.int32)], True, (0, 0, 255), 2)
        cvzone.putTextRect(frame, f'Area1 Count: {count_area1}', (50, 50), scale=1, thickness=2, colorR=(0,200,0), offset=20)
        cvzone.putTextRect(frame, f'Area2 Count: {count_area2}', (50, 80), scale=1, thickness=2, colorR=(0,200,0), offset=20)

        cv2.imshow("RGB", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
