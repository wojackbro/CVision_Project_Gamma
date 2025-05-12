from ultralytics import YOLO
import os

def download_weights():
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Download YOLOv8n weights
    model = YOLO('yolov8n.pt')
    
    # Save the model
    model.save('models/yolov8n.pt')
    print("YOLOv8 weights downloaded successfully!")

if __name__ == "__main__":
    download_weights() 