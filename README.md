# CVision Project Gamma

A computer vision project that performs real-time object detection and 3D dimension estimation using YOLOv8 and OpenCV.

## Features

### 1. Object Detection
- Real-time object detection using YOLOv8
- Supports multiple object classes
- Displays confidence scores for each detection

### 2. 3D Dimension Estimation
The system can estimate three dimensions of detected objects:
- Width (W): Horizontal dimension in millimeters
- Height (H): Vertical dimension in millimeters
- Length (L): Depth/distance dimension in millimeters

#### How Dimension Estimation Works
1. **Calibration**
   - The system automatically calibrates using the first detected object
   - Uses a reference width of 100mm by default
   - Calibration requires a high-confidence detection (>0.5)

2. **Measurement Process**
   - Width and Height: Calculated using pixel-to-millimeter ratio from calibration
   - Length (Depth): Estimated using:
     - Camera focal length (default: 1000 pixels)
     - Object width in pixels
     - Detection confidence score
     - Inverse square law for depth estimation

3. **Visualization**
   - Green bounding box around detected objects
   - Dimension labels (W, H, L) in millimeters
   - 3D bounding box corners
   - Object class and confidence score

## Recent Changes

### Enhanced Dimension Estimation (Latest Update)
1. **Added 3D Measurements**
   - Implemented length (depth) estimation
   - Added focal length parameter for better depth calculations
   - Integrated confidence scores for improved accuracy

2. **Improved Visualization**
   - Added 3D bounding box visualization
   - Display of all three dimensions (W, H, L)
   - Better label placement and formatting

3. **Automatic Calibration**
   - System now calibrates automatically with first high-confidence detection
   - No manual calibration required
   - More user-friendly operation

## Technical Details

### Dependencies
- OpenCV (cv2)
- NumPy
- Ultralytics YOLO
- Python 3.x

### Key Components
1. **DimensionEstimator Class**
   - Handles all dimension calculations
   - Manages calibration
   - Provides visualization methods

2. **Main Application**
   - Real-time video capture
   - YOLO object detection
   - Dimension estimation and display

## Usage

1. Run the application:
```bash
python src/main.py
```

2. The system will:
   - Automatically calibrate using the first detected object
   - Show real-time dimension measurements
   - Display 3D bounding boxes
   - Show object class and confidence

3. Press 'q' to quit the application

## Notes
- The depth estimation is an approximation and may need fine-tuning
- Adjust the `focal_length` parameter in `DimensionEstimator` for better accuracy
- Calibration accuracy depends on the reference object's size and distance

## Future Improvements
- Add manual calibration option
- Implement more accurate depth estimation methods
- Add support for multiple camera angles
- Improve measurement accuracy with machine learning

## Model Selection: yolov8n.pt vs yolov8x.pt

- **yolov8n.pt (Nano):**
  - Fastest and smallest model
  - Lowest accuracy
  - Suitable for devices with limited computational power (e.g., Raspberry Pi, low-end laptops)
  - Recommended if you need real-time speed and can tolerate lower detection accuracy

- **yolov8x.pt (Extra Large):**
  - Largest and most accurate model
  - Requires significantly more computational resources (preferably a powerful GPU)
  - Recommended for best detection accuracy, but may run slowly on CPUs or low-end hardware

**Tip:**
- Use `yolov8n.pt` for speed and low resource usage.
- Use `yolov8x.pt` for maximum accuracy if your hardware can handle it.
- You can change the model in `src/detector/yolo_detector.py` by setting the `model_path` argument.

## Features
- Real-time object detection using YOLOv8
- Dimension measurement of detected objects
- Support for live camera feed
- Pre-trained model support

## Setup
1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download YOLOv8 weights:
```bash
python src/download_weights.py
```

## Project Structure
```
├── src/
│   ├── detector/         # YOLOv8 object detection
│   ├── measurement/      # Dimension measurement
│   ├── camera/          # Camera interface
│   └── utils/           # Utility functions
├── models/              # Model weights
├── data/               # Training data
└── requirements.txt    # Project dependencies
```

## Training
To train the model on custom data:
1. Prepare your dataset in YOLO format
2. Place it in the `data/` directory
3. Run the training script:
```bash
python src/train.py
``` 