
import cv2
import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

class DataPreparator:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def extract_faces_from_images(self, input_dir, output_dir):
        """Extract faces from images in a directory"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        face_count = 0

        for filename in os.listdir(input_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(input_dir, filename)
                img = cv2.imread(img_path)

                if img is None:
                    continue

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

                for i, (x, y, w, h) in enumerate(faces):
                    face_roi = gray[y:y+h, x:x+w]
                    face_resized = cv2.resize(face_roi, (100, 100))

                    output_filename = f"face_{face_count:04d}.jpg"
                    output_path = os.path.join(output_dir, output_filename)

                    cv2.imwrite(output_path, face_resized)
                    face_count += 1

        print(f"Extracted {face_count} faces from {input_dir}")
        return face_count

    def prepare_training_data(self, students_dir):
        """Prepare training data from student directories"""
        X = []
        y = []
        student_labels = {}
        label_counter = 0

        for student_folder in os.listdir(students_dir):
            student_path = os.path.join(students_dir, student_folder)

            if not os.path.isdir(student_path):
                continue

            student_id = student_folder.split('_')[0]
            student_labels[label_counter] = student_id

            for img_file in os.listdir(student_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(student_path, img_file)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                    if img is not None:
                        img_resized = cv2.resize(img, (100, 100))
                        X.append(img_resized.flatten())
                        y.append(label_counter)

            label_counter += 1

        X = np.array(X)
        y = np.array(y)

        # Save prepared data
        with open('data/training_data.pkl', 'wb') as f:
            pickle.dump({
                'X': X,
                'y': y,
                'labels': student_labels
            }, f)

        print(f"Training data prepared: {X.shape[0]} samples from {len(student_labels)} students")
        return X, y, student_labels

    def augment_data(self, X, y, augment_factor=2):
        """Simple data augmentation"""
        augmented_X = []
        augmented_y = []

        for i in range(len(X)):
            img = X[i].reshape(100, 100)
            label = y[i]

            # Original image
            augmented_X.append(X[i])
            augmented_y.append(label)

            # Augmented versions
            for _ in range(augment_factor):
                # Random rotation
                angle = np.random.uniform(-15, 15)
                M = cv2.getRotationMatrix2D((50, 50), angle, 1)
                rotated = cv2.warpAffine(img, M, (100, 100))

                # Random brightness
                brightness = np.random.uniform(0.8, 1.2)
                bright_img = np.clip(img * brightness, 0, 255).astype(np.uint8)

                augmented_X.append(rotated.flatten())
                augmented_y.append(label)

                augmented_X.append(bright_img.flatten())
                augmented_y.append(label)

        return np.array(augmented_X), np.array(augmented_y)

# Usage example
if __name__ == "__main__":
    prep = DataPreparator()

    # Extract faces from raw images
    # prep.extract_faces_from_images('data/raw_images', 'data/extracted_faces')

    # Prepare training data from student images
    X, y, labels = prep.prepare_training_data('data/student_images')

    # Augment data
    X_aug, y_aug = prep.augment_data(X, y)

    print(f"Final dataset: {X_aug.shape[0]} samples")
