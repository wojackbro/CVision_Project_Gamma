import cv2
import time

class Camera:
    def __init__(self, camera_id=0, width=1280, height=720):
        """
        Initialize camera interface
        Args:
            camera_id: Camera device ID
            width: Frame width
            height: Frame height
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.cap = None
        self.fps = 0
        self.frame_count = 0
        self.start_time = None

    def start(self):
        """Start camera capture"""
        self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_AVFOUNDATION)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.start_time = time.time()
        return self.cap.isOpened()

    def read(self):
        """
        Read a frame from the camera
        Returns:
            Tuple of (success, frame)
        """
        if self.cap is None:
            return False, None

        success, frame = self.cap.read()
        print("Frame read success:", success, "Frame shape:", None if frame is None else frame.shape)
        if success and frame is not None:
            cv2.imwrite('test_frame.jpg', frame)  # Save a frame for inspection
        if success:
            self.frame_count += 1
            elapsed_time = time.time() - self.start_time
            if elapsed_time > 0:
                self.fps = self.frame_count / elapsed_time

        return success, frame

    def stop(self):
        """Stop camera capture"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def get_fps(self):
        """Get current FPS"""
        return self.fps

    def __del__(self):
        """Destructor to ensure camera is released"""
        self.stop() 