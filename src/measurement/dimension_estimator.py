import cv2
import numpy as np

class DimensionEstimator:
    def __init__(self, reference_width_mm=100, focal_length=1000):
        """
        Initialize dimension estimator
        Args:
            reference_width_mm: Reference width in millimeters for calibration
            focal_length: Camera focal length in pixels (default: 1000)
        """
        self.reference_width_mm = reference_width_mm
        self.pixels_per_mm = None
        self.focal_length = focal_length

    def calibrate(self, reference_width_pixels, reference_distance_mm=1000):
        """
        Calibrate the estimator using a reference object
        Args:
            reference_width_pixels: Width of reference object in pixels
            reference_distance_mm: Distance to reference object in millimeters
        """
        self.pixels_per_mm = reference_width_pixels / self.reference_width_mm
        self.reference_distance = reference_distance_mm

    def estimate_dimensions(self, bbox, confidence):
        """
        Estimate dimensions of an object using its bounding box and detection confidence
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            confidence: Detection confidence score
        Returns:
            Dictionary containing width, height, and length in millimeters
        """
        if self.pixels_per_mm is None:
            raise ValueError("Calibrate the estimator first using a reference object")

        x1, y1, x2, y2 = bbox
        width_pixels = x2 - x1
        height_pixels = y2 - y1

        # Calculate width and height in millimeters
        width_mm = width_pixels / self.pixels_per_mm
        height_mm = height_pixels / self.pixels_per_mm

        # Estimate depth using the inverse square law and confidence score
        # Higher confidence means the object is more likely to be closer
        depth_factor = 1.0 / (confidence + 0.1)  # Add small constant to avoid division by zero
        length_mm = (width_mm * self.focal_length) / (width_pixels * depth_factor)

        return {
            'width_mm': width_mm,
            'height_mm': height_mm,
            'length_mm': length_mm
        }

    def draw_dimensions(self, frame, bbox, dimensions):
        """
        Draw dimension measurements on the frame
        Args:
            frame: Input frame
            bbox: Bounding box [x1, y1, x2, y2]
            dimensions: Dictionary containing width, height, and length measurements
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
        
        # Draw length (depth) measurement
        length_text = f"L: {dimensions['length_mm']:.1f}mm"
        cv2.putText(frame, length_text, (x1, y2 + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw 3D bounding box corners
        corners = np.array([
            [x1, y1], [x2, y1],  # Top corners
            [x1, y2], [x2, y2]   # Bottom corners
        ])
        
        # Draw lines connecting the corners
        for i in range(4):
            cv2.line(frame, tuple(corners[i]), tuple(corners[(i+1)%4]), (0, 255, 0), 2)
        
        return frame 