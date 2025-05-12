import cv2
import sys
import os
import numpy as np
from ultralytics import YOLO

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detector.yolo_detector import YOLODetector
from src.measurement.dimension_estimator import DimensionEstimator
from src.camera.camera import Camera

def main():
    # Initialize YOLO model
    model = YOLO('yolov8n.pt')
    
    # Initialize dimension estimator
    dimension_estimator = DimensionEstimator(reference_width_mm=100, focal_length=1000)
    
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    
    # Calibration flag
    calibrated = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run YOLO detection
        results = model(frame)
        
        # Process each detection
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                
                # If not calibrated, use the first detection as reference
                if not calibrated and confidence > 0.5:
                    reference_width_pixels = x2 - x1
                    dimension_estimator.calibrate(reference_width_pixels)
                    calibrated = True
                    print("Calibration complete!")
                
                # Estimate dimensions
                if calibrated:
                    dimensions = dimension_estimator.estimate_dimensions(
                        [x1, y1, x2, y2], confidence
                    )
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw dimensions
                    frame = dimension_estimator.draw_dimensions(frame, [x1, y1, x2, y2], dimensions)
                    
                    # Draw class name and confidence
                    class_name = model.names[int(box.cls[0])]
                    label = f"{class_name}: {confidence:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 40),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('Object Detection with Dimensions', frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 