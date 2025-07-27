
"""
Configuration settings for the Attendance System
"""
import os
from datetime import time

class Config:
    # Directory paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    STUDENTS_DIR = os.path.join(DATA_DIR, 'students')
    ATTENDANCE_DIR = os.path.join(DATA_DIR, 'attendance')

    # Model paths
    EMOTION_MODEL_PATH = os.path.join(MODELS_DIR, 'emotion_model.h5')
    FACE_EMBEDDINGS_PATH = os.path.join(MODELS_DIR, 'face_embeddings.pkl')
    STUDENT_INFO_PATH = os.path.join(MODELS_DIR, 'student_info.pkl')

    # Time window for attendance (9:30 AM to 10:00 AM)
    ATTENDANCE_START_TIME = time(9, 30)  # 9:30 AM
    ATTENDANCE_END_TIME = time(10, 0)    # 10:00 AM

    # Face detection settings
    FACE_DETECTION_CONFIDENCE = 0.9
    FACE_RECOGNITION_THRESHOLD = 0.6

    # Emotion detection settings
    EMOTION_CONFIDENCE_THRESHOLD = 0.5

    # Camera settings
    CAMERA_INDEX = 0
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480

    # File export settings
    EXPORT_CSV = True
    EXPORT_EXCEL = True

    @staticmethod
    def create_directories():
        """Create necessary directories if they don't exist"""
        directories = [
            Config.DATA_DIR,
            Config.MODELS_DIR,
            Config.STUDENTS_DIR,
            Config.ATTENDANCE_DIR
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ Directory created/verified: {directory}")
