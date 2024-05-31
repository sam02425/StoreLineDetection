import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import cvzone
from deep_sort import DeepSort

# Initialize YOLO and DeepSORT
model = YOLO('yolov8n.pt')
deep_sort_weights = 'deep_sort/deep/checkpoint/ckpt.t7'
tracker = DeepSort(model_path=deep_sort_weights, max_age=90, use_cuda=True)

def is_inside_polygon(point, polygon):
    return cv2.pointPolygonTest(np.array(polygon, np.int32), point, False) >= 0

cap = cv2.VideoCapture('cr.mp4')
class_list = open("coco.txt", "r").read().split("\n")
area1 = [(544,12),(587,377),(713,372),(643,13)]
area2 = [(763,17),(969,343),(1016,298),(924,16)]
confidence_threshold = 0.5
unique_track_ids = set()

def preprocess_detections(detections_df, confidence_threshold):
    """Filter detections based on confidence and ensure no NaN values."""
    # Ensure detections DataFrame does not contain NaN or infinite values
    detections_df = detections_df.dropna().replace([np.inf, -np.inf], np.nan).dropna()

    # Filter detections based on confidence threshold
    filtered_detections_df = detections_df[detections_df.iloc[:, 4] >= confidence_threshold]

    return filtered_detections_df

def convert_to_tlwh_bboxes(detections):
    """Convert detections to tlwh format and ensure valid boxes."""
    tlwh_bboxes = detections[:, :4]
    tlwh_bboxes[:, 2:4] -= tlwh_bboxes[:, :2]  # Convert to width and height
    valid_boxes_mask = np.all(tlwh_bboxes[:, 2:4] > 0, axis=1)
    return tlwh_bboxes[valid_boxes_mask].astype(int), detections[valid_boxes_mask, 4]

def draw_tracking_info(frame, track, area1, area2):
    """Draw tracking information on the frame."""
    bbox = [int(coord) for coord in track[1:5]]
    track_id = track[0]
    center_x, center_y = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
    if is_inside_polygon((center_x, center_y), area1) or is_inside_polygon((center_x, center_y), area2):
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 0), 2)
        cvzone.putTextRect(frame, f'ID: {track_id}', (bbox[0], bbox[1] - 10), scale=1, thickness=2, colorR=(0, 255, 0))
        return 1
    return 0

def process_video():
    cap = cv2.VideoCapture('cr.mp4')
    model = YOLO('yolov8n.pt')
    tracker = DeepSort(model_path='deep_sort/deep/checkpoint/ckpt.t7', max_age=90, use_cuda=True)
    unique_track_ids = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (1020, 500))
        results = model.predict(frame)
        detections_df = pd.DataFrame(results[0].boxes.data).astype("float")

        # Preprocess detections while it's still a DataFrame
        detections_df = preprocess_detections(detections_df, confidence_threshold=0.5)

        # Now convert the filtered DataFrame to numpy array for further operations
        detections_np = detections_df.to_numpy()

        # Separate tlwh_bboxes and confidences from filtered detections
        tlwh_bboxes = detections_np[:, :4]
        confidences = detections_np[:, 4]

        # Subtract to convert to width and height from bottom right coordinates
        tlwh_bboxes[:, 2:4] -= tlwh_bboxes[:, :2]

        # Ensure tlwh_bboxes does not contain invalid or empty boxes
        valid_boxes_mask = np.all(tlwh_bboxes[:, 2:4] > 0, axis=1)
        tlwh_bboxes = tlwh_bboxes[valid_boxes_mask]
        confidences = confidences[valid_boxes_mask]

        # Convert tlwh_bboxes to integer
        tlwh_bboxes = np.array(tlwh_bboxes, dtype=int)

        # Proceed with DeepSORT tracker update
        tracks = tracker.update(tlwh_bboxes, confidences, frame)

        count_area1, count_area2 = 0, 0
        for track in tracks:
            # Assuming 'track' is now an array with structured information.
            # Extracting track ID and bounding box coordinates.
            track_id = track[0]
            bbox = track[1:5]

            # Convert bbox to integers if they are not already.
            bbox = [int(coord) for coord in bbox]

            # Calculating the center of the bounding box.
            center_x, center_y = int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)

            unique_track_ids.add(track_id)  # Keep track of unique track IDs.

            # Now, use 'center_x' and 'center_y' to check if the center of the track is inside the polygons.
            if is_inside_polygon((center_x, center_y), area1):
                count_area1 += 1
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 0), 2)
                cvzone.putTextRect(frame, f'ID: {track_id}', (bbox[0], bbox[1] - 10), scale=1, thickness=2,
                                   colorR=(0, 255, 0))
            elif is_inside_polygon((center_x, center_y), area2):
                count_area2 += 1
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 0), 2)
                cvzone.putTextRect(frame, f'ID: {track_id}', (bbox[0], bbox[1] - 10), scale=1, thickness=2,
                                   colorR=(0, 255, 0))
        display_tracking_info(frame, count_area1, count_area2, len(unique_track_ids))
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def display_tracking_info(frame, count_area1, count_area2, unique_ids_count):
    """Display tracking information such as area counts and unique ID count."""
    cvzone.putTextRect(frame, f'Area1 Count: {count_area1}', (50, 50), scale=1, thickness=2, colorR=(0,200,0), offset=20)
    cvzone.putTextRect(frame, f'Area2 Count: {count_area2}', (50, 80), scale=1, thickness=2, colorR=(0,200,0), offset=20)
    cvzone.putTextRect(frame, f'Unique IDs: {unique_ids_count}', (50, 110), scale=1, thickness=2, colorR=(0,200,0), offset=20)
    cv2.imshow("RGB", frame)

if __name__ == "__main__":
    process_video()