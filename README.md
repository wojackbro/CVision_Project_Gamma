# Computer Vision Object Detection and Measurement

This project implements real-time object detection and dimension measurement using YOLOv8 and OpenCV. It can detect objects from a live camera feed and calculate their approximate dimensions.

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

## Usage
Run the main application:
```bash
python src/main.py
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