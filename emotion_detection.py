
"""
Emotion Detection Module using trained CNN
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from config import Config

class EmotionDetector:
    def __init__(self):
        self.model = None
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.confidence_threshold = Config.EMOTION_CONFIDENCE_THRESHOLD
        self.load_model()

    def load_model(self):
        """Load trained emotion detection model"""
        try:
            self.model = keras.models.load_model(Config.EMOTION_MODEL_PATH)
            print(" Emotion detection model loaded successfully")
        except Exception as e:
            print(f" Could not load emotion model: {e}")
            self.model = None

    def preprocess_face(self, face_image):
        """
        Preprocess face image for emotion detection

        Args:
            face_image: Input face image

        Returns:
            Preprocessed image ready for model input
        """
        # Convert to grayscale if needed
        if len(face_image.shape) == 3:
            gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_face = face_image

        # Resize to 48x48 (model input size)
        resized_face = cv2.resize(gray_face, (48, 48))

        # Normalize pixel values to [0, 1]
        normalized_face = resized_face.astype('float32') / 255.0

        # Add batch and channel dimensions: (1, 48, 48, 1)
        processed_face = np.expand_dims(normalized_face, axis=[0, -1])

        return processed_face

    def detect_emotion(self, face_image):
        """
        Detect emotion from face image

        Args:
            face_image: Input face image

        Returns:
            Dictionary with emotion prediction results
        """
        if self.model is None:
            return {
                'emotion': 'neutral',
                'confidence': 0.0,
                'all_emotions': {}
            }

        # Preprocess the face
        processed_face = self.preprocess_face(face_image)

        try:
            # Make prediction
            predictions = self.model.predict(processed_face, verbose=0)

            # Get emotion probabilities
            emotion_probabilities = predictions[0]

            # Find the emotion with highest probability
            max_emotion_idx = np.argmax(emotion_probabilities)
            predicted_emotion = self.emotion_labels[max_emotion_idx]
            confidence = float(emotion_probabilities[max_emotion_idx])

            # Create dictionary of all emotions and their probabilities
            all_emotions = {
                emotion: float(prob) 
                for emotion, prob in zip(self.emotion_labels, emotion_probabilities)
            }

            return {
                'emotion': predicted_emotion,
                'confidence': confidence,
                'all_emotions': all_emotions
            }

        except Exception as e:
            print(f"Error in emotion detection: {e}")
            return {
                'emotion': 'neutral',
                'confidence': 0.0,
                'all_emotions': {}
            }

    def is_confident_prediction(self, confidence):
        """Check if prediction confidence meets threshold"""
        return confidence >= self.confidence_threshold

    def get_emotion_color(self, emotion):
        """Get color for emotion visualization"""
        emotion_colors = {
            'happy': (0, 255, 0),      # Green
            'sad': (255, 0, 0),        # Blue
            'angry': (0, 0, 255),      # Red
            'fear': (128, 0, 128),     # Purple
            'surprise': (0, 255, 255), # Cyan
            'disgust': (0, 128, 0),    # Dark Green
            'neutral': (128, 128, 128) # Gray
        }
        return emotion_colors.get(emotion, (255, 255, 255))  # Default white

    def draw_emotion_info(self, image, emotion_result, position):
        """
        Draw emotion information on image

        Args:
            image: Input image
            emotion_result: Emotion detection result
            position: (x, y) position to draw text
        """
        emotion = emotion_result['emotion']
        confidence = emotion_result['confidence']

        # Get color for this emotion
        color = self.get_emotion_color(emotion)

        # Draw emotion text
        text = f"{emotion}: {confidence:.2f}"
        cv2.putText(image, text, position,
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return image

    def create_emotion_bar_chart(self, emotion_result, width=300, height=200):
        """
        Create a bar chart visualization of emotion probabilities

        Args:
            emotion_result: Emotion detection result
            width: Chart width
            height: Chart height

        Returns:
            Bar chart image
        """
        # Create blank image
        chart = np.zeros((height, width, 3), dtype=np.uint8)

        if not emotion_result['all_emotions']:
            return chart

        # Get emotions and their probabilities
        emotions = list(emotion_result['all_emotions'].keys())
        probabilities = list(emotion_result['all_emotions'].values())

        # Calculate bar dimensions
        bar_width = width // len(emotions)
        max_bar_height = height - 40  # Leave space for labels

        # Draw bars
        for i, (emotion, prob) in enumerate(zip(emotions, probabilities)):
            # Calculate bar height
            bar_height = int(prob * max_bar_height)

            # Calculate bar position
            x1 = i * bar_width + 5
            x2 = x1 + bar_width - 10
            y1 = height - 20
            y2 = y1 - bar_height

            # Get color for this emotion
            color = self.get_emotion_color(emotion)

            # Draw bar
            cv2.rectangle(chart, (x1, y2), (x2, y1), color, -1)

            # Draw emotion label
            cv2.putText(chart, emotion[:3], (x1, y1 + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        return chart
