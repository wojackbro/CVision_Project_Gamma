from ultralytics import YOLO
import cv2
import numpy as np

class YOLODetector:
    def __init__(self, model_path='yolov8n.pt', conf_threshold=0.5):
        """
        Initialize YOLOv8 detector
        Args:
            model_path: Path to YOLOv8 model weights (default: yolov8x.pt for best accuracy)
            conf_threshold: Confidence threshold for detections
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect(self, frame):
        """
        Detect objects in the frame
        Args:
            frame: Input frame (numpy array)
        Returns:
            List of detections with bounding boxes and class information
        """
        results = self.model(frame, conf=self.conf_threshold)[0]
        detections = []
        
        for r in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            detections.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'score': score,
                'class_id': int(class_id),
                'class_name': results.names[int(class_id)]
            })
        
        return detections

    def draw_detections(self, frame, detections):
        """
        Draw bounding boxes and labels on the frame
        Args:
            frame: Input frame
            detections: List of detections
        Returns:
            Frame with drawn detections
        """
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = f"{det['class_name']} {det['score']:.2f}"
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label background
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - label_height - 10), (x1 + label_width, y1), (0, 255, 0), -1)
            
            # Draw label text
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return frame 