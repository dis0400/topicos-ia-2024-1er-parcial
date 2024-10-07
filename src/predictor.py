from ultralytics import YOLO
import numpy as np
import cv2
from shapely.geometry import Polygon, box
from src.models import Detection, PredictionType, Segmentation
from src.config import get_settings

SETTINGS = get_settings()


def match_gun_bbox(segment: list[list[int]], bboxes: list[list[int]], max_distance: int = 10) -> list[int] | None:
    matched_box = None
    min_distance = max_distance
    
    segment_poly = Polygon([(segment[0][0], segment[0][1]), (segment[1][0], segment[1][1]), 
                            (segment[2][0], segment[2][1]), (segment[3][0], segment[3][1])])
    
    for bbox in bboxes:
        bbox_poly = box(bbox[0], bbox[1], bbox[2], bbox[3])
        
        distance = segment_poly.distance(bbox_poly)
        
        if distance < min_distance and distance <= max_distance:
            min_distance = distance
            matched_box = bbox

    return matched_box


def annotate_detection(image_array: np.ndarray, detection: Detection) -> np.ndarray:
    ann_color = (255, 0, 0)
    annotated_img = image_array.copy()
    for label, conf, box in zip(detection.labels, detection.confidences, detection.boxes):
        x1, y1, x2, y2 = box
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), ann_color, 3)
        cv2.putText(
            annotated_img,
            f"{label}: {conf:.1f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            ann_color,
            2,
        )
    return annotated_img


def annotate_segmentation(image_array: np.ndarray, segmentation: Segmentation, draw_boxes: bool = True) -> np.ndarray:
    annotated_img = image_array.copy()
    danger_color = (0, 0, 255)
    safe_color = (0, 255, 0)
    
    for label, polygon in zip(segmentation.labels, segmentation.polygons):
        color = danger_color if label == 'danger' else safe_color

        points = np.array(polygon, np.int32)
        points = points.reshape((-1, 1, 2))

        cv2.polylines(annotated_img, [points], isClosed=True, color=color, thickness=2)

        if draw_boxes:
            x, y, w, h = cv2.boundingRect(points)
            cv2.rectangle(annotated_img, (x, y), (x + w, y + h), color, 2)
    
    return annotated_img


class GunDetector:
    def __init__(self) -> None:
        print(f"loading od model: {SETTINGS.od_model_path}")
        self.od_model = YOLO(SETTINGS.od_model_path)
        print(f"loading seg model: {SETTINGS.seg_model_path}")
        self.seg_model = YOLO(SETTINGS.seg_model_path)

    def detect_guns(self, image_array: np.ndarray, threshold: float = 0.5):
        results = self.od_model(image_array, conf=threshold)[0]
        labels = results.boxes.cls.tolist()
        indexes = [
            i for i in range(len(labels)) if labels[i] in [3, 4]
        ]  # 0 = "person"
        boxes = [
            [int(v) for v in box]
            for i, box in enumerate(results.boxes.xyxy.tolist())
            if i in indexes
        ]
        confidences = [
            c for i, c in enumerate(results.boxes.conf.tolist()) if i in indexes
        ]
        labels_txt = [
            results.names[labels[i]] for i in indexes
        ]
        return Detection(
            pred_type=PredictionType.object_detection,
            n_detections=len(boxes),
            boxes=boxes,
            labels=labels_txt,
            confidences=confidences,
        )
    
    def segment_people(self, image_array: np.ndarray, threshold: float = 0.5, max_distance: int = 10):
        results = self.seg_model(image_array, conf=threshold)[0]
        labels = results.boxes.cls.tolist()
        person_indexes = [i for i in range(len(labels)) if labels[i] == 0] 
        person_boxes = [
            [int(v) for v in box] for i, box in enumerate(results.boxes.xyxy.tolist()) if i in person_indexes
        ]
        person_polygons = [
            [[int(coord[0]), int(coord[1])] for coord in results.masks.xy[i]] for i in person_indexes
        ]
        gun_detections = self.detect_guns(image_array, threshold)
        person_labels = []
        for person_box, person_polygon in zip(person_boxes, person_polygons):
            matched_gun = match_gun_bbox(person_polygon, gun_detections.boxes, max_distance)
            if matched_gun:
                person_labels.append("danger")
            else:
                person_labels.append("safe")

        return Segmentation(
            pred_type=PredictionType.segmentation,
            n_detections=len(person_boxes),
            polygons=person_polygons,
            boxes=person_boxes,
            labels=person_labels
        )
