
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import cv2
import os

class EmotionModelTrainer:
    def __init__(self):
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        self.img_size = 48

    def load_fer2013_data(self, csv_path):
        """Load FER2013 dataset from CSV file"""
        print("Loading FER2013 dataset...")

        data = pd.read_csv(csv_path)
        pixels = data['pixels'].tolist()
        emotions = data['emotion'].tolist()

        # Convert pixel strings to numpy arrays
        faces = []
        for pixel_sequence in pixels:
            face = np.fromstring(pixel_sequence, dtype=int, sep=' ')
            face = face.reshape((self.img_size, self.img_size))
            faces.append(face)

        faces = np.array(faces)
        emotions = np.array(emotions)

        # Normalize pixel values
        faces = faces / 255.0

        # Reshape for CNN input
        faces = faces.reshape(-1, self.img_size, self.img_size, 1)

        # Convert emotions to categorical
        emotions = to_categorical(emotions, num_classes=len(self.emotions))

        print(f"Dataset loaded: {faces.shape[0]} samples")
        return faces, emotions

    def create_model(self):
        """Create CNN model for emotion detection"""
        model = Sequential([
            # First convolutional block
            Conv2D(32, (3, 3), activation='relu', input_shape=(self.img_size, self.img_size, 1)),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            # Second convolutional block
            Conv2D(64, (3, 3), activation='relu'),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            # Third convolutional block
            Conv2D(128, (3, 3), activation='relu'),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            # Fully connected layers
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(len(self.emotions), activation='softmax')
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def train_model(self, csv_path, epochs=50, batch_size=64):
        """Train the emotion detection model"""
        # Load data
        X, y = self.load_fer2013_data(csv_path)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y.argmax(axis=1)
        )

        # Create model
        model = self.create_model()

        print("Model architecture:")
        model.summary()

        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.0001
        )

        # Train model
        print("Starting training...")
        history = model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )

        # Evaluate model
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"\nTest Accuracy: {test_accuracy:.4f}")

        # Save model
        model.save('models/emotion_model.h5')
        print("Model saved to models/emotion_model.h5")

        return model, history

# Usage example
if __name__ == "__main__":
    trainer = EmotionModelTrainer()

    # Download FER2013 dataset from: https://www.kaggle.com/datasets/msambare/fer2013
    # Place the fer2013.csv file in data/ directory

    csv_path = "data/fer2013.csv"
    if os.path.exists(csv_path):
        model, history = trainer.train_model(csv_path)
    else:
        print(f"Please download FER2013 dataset and place it at {csv_path}")
        print("Download from: https://www.kaggle.com/datasets/msambare/fer2013")
