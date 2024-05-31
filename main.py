import cv2
import numpy as np
from ultralytics import YOLO
from ppdet.modeling.mot.tracker import JDETracker
from collections import defaultdict
import csv
import time

# Load the YOLOv5 model
model = YOLO('yolov8n.pt')

# Initialize the PaddleDetection object tracker
tracker = JDETracker()

# Initialize variables
heatmap = None
people_count = 0
people_data = defaultdict(dict)
frame_number = 0
video_path = 'test_videos/1.mp4'

# Open the video file
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Initialize the heatmap
heatmap = np.zeros((height, width), dtype=np.float32)

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model(frame)

    # Update the heatmap based on detected people
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            if cls == 0:  # Class 0 represents person
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                heatmap[y1:y2, x1:x2] += 1

    # Update the object tracker
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            if cls == 0:  # Class 0 represents person
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append([x1, y1, x2, y2, box.conf[0]])

    online_targets = tracker.update(detections, frame)

    # Count people and update people data
    for t in online_targets:
        track_id = t.track_id
        x1, y1, x2, y2 = map(int, t.tlbr)

        if track_id not in people_data:
            people_data[track_id]['start_frame'] = frame_number
            people_data[track_id]['start_time'] = frame_number / fps
            people_count += 1
        people_data[track_id]['end_frame'] = frame_number
        people_data[track_id]['end_time'] = frame_number / fps

    # Display the frame with bounding boxes and heatmap
    for t in online_targets:
        x1, y1, x2, y2 = map(int, t.tlbr)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, str(t.track_id), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.putText(frame, f"People Count: {people_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    heatmap_display = cv2.applyColorMap((heatmap * 255 / np.max(heatmap)).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_display = cv2.resize(heatmap_display, (width // 2, height // 2))
    frame[:height // 2, width // 2:] = heatmap_display

    cv2.imshow("Object Tracking and Heatmap", frame)

    # Handle key events
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('f'):
        frame_number += 1
    elif key == ord('b'):
        frame_number -= 1
        if frame_number < 0:
            frame_number = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    frame_number += 1

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()

# Export analytical details to CSV
with open('people_data.csv', 'w', newline='') as csvfile:
    fieldnames = ['Person ID', 'Start Frame', 'End Frame', 'Start Time', 'End Time', 'Duration']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for track_id, data in people_data.items():
        start_frame = data['start_frame']
        end_frame = data['end_frame']
        start_time = data['start_time']
        end_time = data['end_time']
        duration = end_time - start_time

        writer.writerow({
            'Person ID': track_id,
            'Start Frame': start_frame,
            'End Frame': end_frame,
            'Start Time': start_time,
            'End Time': end_time,
            'Duration': duration
        })