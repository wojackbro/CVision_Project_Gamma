import cv2
import numpy as np

class DimensionEstimator:
    def __init__(self, reference_width_mm=100):
        """
        Initialize dimension estimator
        Args:
            reference_width_mm: Reference width in millimeters for calibration
        """
        self.reference_width_mm = reference_width_mm
        self.pixels_per_mm = None

    def calibrate(self, reference_width_pixels):
        """
        Calibrate the estimator using a reference object
        Args:
            reference_width_pixels: Width of reference object in pixels
        """
        self.pixels_per_mm = reference_width_pixels / self.reference_width_mm

    def estimate_dimensions(self, bbox):
        """
        Estimate dimensions of an object using its bounding box
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
        Returns:
            Dictionary containing width and height in millimeters
        """
        if self.pixels_per_mm is None:
            raise ValueError("Calibrate the estimator first using a reference object")

        x1, y1, x2, y2 = bbox
        width_pixels = x2 - x1
        height_pixels = y2 - y1

        width_mm = width_pixels / self.pixels_per_mm
        height_mm = height_pixels / self.pixels_per_mm

        return {
            'width_mm': width_mm,
            'height_mm': height_mm
        }

    def draw_dimensions(self, frame, bbox, dimensions):
        """
        Draw dimension measurements on the frame
        Args:
            frame: Input frame
            bbox: Bounding box [x1, y1, x2, y2]
            dimensions: Dictionary containing width and height measurements
        Returns:
            Frame with drawn measurements
        """
        x1, y1, x2, y2 = bbox
        
        # Draw width measurement
        width_text = f"W: {dimensions['width_mm']:.1f}mm"
        cv2.putText(frame, width_text, (x1, y1 - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw height measurement
        height_text = f"H: {dimensions['height_mm']:.1f}mm"
        cv2.putText(frame, height_text, (x2 + 10, y1), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame 