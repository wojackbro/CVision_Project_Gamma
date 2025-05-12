import cv2
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detector.yolo_detector import YOLODetector
from src.measurement.dimension_estimator import DimensionEstimator
from src.camera.camera import Camera

def main():
    # Initialize components
    camera = Camera()
    detector = YOLODetector()
    dimension_estimator = DimensionEstimator()

    # Start camera
    if not camera.start():
        print("Error: Could not open camera")
        return

    print("Press 'c' to calibrate with a reference object")
    print("Press 'q' to quit")

    calibrated = False

    while True:
        # Read frame
        success, frame = camera.read()
        if not success:
            print("Error: Could not read frame")
            break

        # Detect objects
        detections = detector.detect(frame)

        # Process each detection
        for detection in detections:
            # Draw detection
            frame = detector.draw_detections(frame, [detection])

            # If calibrated, measure dimensions
            if calibrated:
                try:
                    dimensions = dimension_estimator.estimate_dimensions(detection['bbox'])
                    frame = dimension_estimator.draw_dimensions(frame, detection['bbox'], dimensions)
                except ValueError as e:
                    print(f"Error: {e}")

        # Draw FPS
        fps = camera.get_fps()
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show frame
        cv2.imshow("Object Detection and Measurement", frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            if len(detections) > 0:
                # Use the first detection as reference
                reference_width = detections[0]['bbox'][2] - detections[0]['bbox'][0]
                dimension_estimator.calibrate(reference_width)
                calibrated = True
                print("Calibration complete!")
            else:
                print("No objects detected for calibration")

    # Cleanup
    camera.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 