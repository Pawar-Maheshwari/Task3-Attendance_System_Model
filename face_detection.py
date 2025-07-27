
"""
Face Detection Module using MTCNN
"""
import cv2
import numpy as np
from mtcnn import MTCNN
from config import Config

class FaceDetector:
    def __init__(self):
        self.detector = MTCNN()
        self.confidence_threshold = Config.FACE_DETECTION_CONFIDENCE

    def detect_faces(self, image):
        """
        Detect faces in the given image

        Args:
            image: Input image (BGR format)

        Returns:
            List of detected faces with bounding boxes and landmarks
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect faces
        faces = self.detector.detect_faces(rgb_image)

        # Filter faces by confidence
        filtered_faces = [
            face for face in faces 
            if face['confidence'] >= self.confidence_threshold
        ]

        return filtered_faces

    def extract_face(self, image, face_box, margin=20):
        """
        Extract face region from image

        Args:
            image: Input image
            face_box: Face bounding box [x, y, width, height]
            margin: Additional margin around face

        Returns:
            Extracted face image
        """
        x, y, width, height = face_box

        # Add margin
        x = max(0, x - margin)
        y = max(0, y - margin)
        width = width + 2 * margin
        height = height + 2 * margin

        # Ensure boundaries are within image
        img_height, img_width = image.shape[:2]
        x2 = min(img_width, x + width)
        y2 = min(img_height, y + height)

        # Extract face
        face = image[y:y2, x:x2]
        return face

    def preprocess_face_for_emotion(self, face):
        """
        Preprocess face for emotion detection

        Args:
            face: Face image

        Returns:
            Preprocessed face (48x48 grayscale)
        """
        # Convert to grayscale
        if len(face.shape) == 3:
            gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        else:
            gray_face = face

        # Resize to 48x48
        resized_face = cv2.resize(gray_face, (48, 48))

        # Normalize pixel values
        normalized_face = resized_face.astype('float32') / 255.0

        # Add batch and channel dimensions
        processed_face = np.expand_dims(normalized_face, axis=[0, -1])

        return processed_face

    def draw_face_box(self, image, face, label=None, color=(0, 255, 0)):
        """
        Draw bounding box around detected face

        Args:
            image: Input image
            face: Face detection result
            label: Optional label text
            color: Box color (BGR)
        """
        x, y, width, height = face['box']
        confidence = face['confidence']

        # Draw rectangle
        cv2.rectangle(image, (x, y), (x + width, y + height), color, 2)

        # Draw confidence
        cv2.putText(image, f'{confidence:.2f}', (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw label if provided
        if label:
            cv2.putText(image, label, (x, y + height + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return image
