
import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime, time as dt_time
import pickle
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from deepface import DeepFace
import time
import json

class SmartAttendanceSystem:
    def __init__(self):
        """
        Initialize the Smart Attendance System with Face Recognition and Emotion Detection
        """
        self.face_model = None
        self.emotion_model = None
        self.students_database = {}
        self.attendance_records = []
        self.allowed_start_time = dt_time(9, 30)  # 9:30 AM
        self.allowed_end_time = dt_time(10, 0)    # 10:00 AM
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

        # Load or create models
        self.setup_models()

    def check_time_window(self):
        """Check if current time is within allowed attendance window"""
        current_time = datetime.now().time()
        is_allowed = self.allowed_start_time <= current_time <= self.allowed_end_time

        if not is_allowed:
            print(f"Attendance system is only active between {self.allowed_start_time} and {self.allowed_end_time}")
            print(f"Current time: {current_time}")

        return is_allowed

    def create_face_recognition_model(self):
        """Create KNN model for face recognition"""
        return KNeighborsClassifier(n_neighbors=5, weights='distance', metric='euclidean')

    def create_emotion_model(self):
        """Create CNN model for emotion detection"""
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(7, activation='softmax')  # 7 emotions
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def setup_models(self):
        """Initialize or load existing models"""
        try:
            self.load_models()
        except:
            print("Creating new models...")
            self.face_model = self.create_face_recognition_model()
            self.emotion_model = self.create_emotion_model()
            print("New models created successfully!")

    def detect_faces(self, image):
        """Detect faces in the given image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        return faces, gray

    def preprocess_face(self, face_roi, target_size=(100, 100)):
        """Preprocess face region for recognition"""
        face_resized = cv2.resize(face_roi, target_size)
        face_flattened = face_resized.flatten()
        return face_flattened

    def preprocess_emotion(self, face_roi, target_size=(48, 48)):
        """Preprocess face region for emotion detection"""
        face_resized = cv2.resize(face_roi, target_size)
        face_normalized = face_resized / 255.0
        face_expanded = np.expand_dims(face_normalized, axis=[0, -1])
        return face_expanded

    def register_student(self, student_id, student_name, num_images=50):
        """Register a new student by capturing multiple face images"""
        if not self.check_time_window():
            return False

        print(f"Registering student: {student_name} (ID: {student_id})")
        print(f"Please look at the camera. Capturing {num_images} images...")

        cap = cv2.VideoCapture(0)
        faces_data = []
        count = 0

        student_folder = f'data/student_images/{student_id}_{student_name}'
        os.makedirs(student_folder, exist_ok=True)

        while count < num_images:
            ret, frame = cap.read()
            if not ret:
                continue

            faces, gray = self.detect_faces(frame)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

                face_roi = gray[y:y+h, x:x+w]
                face_processed = self.preprocess_face(face_roi)
                faces_data.append(face_processed)

                # Save individual face image
                cv2.imwrite(f'{student_folder}/face_{count}.jpg', face_roi)
                count += 1

                cv2.putText(frame, f'Captured: {count}/{num_images}', 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                if count >= num_images:
                    break

            cv2.imshow('Registering Student', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        # Store student data
        self.students_database[student_id] = {
            'name': student_name,
            'face_data': faces_data,
            'registered_on': datetime.now().isoformat()
        }

        # Retrain the face recognition model
        self.train_face_model()

        print(f"Student {student_name} registered successfully!")
        return True

    def train_face_model(self):
        """Train the face recognition model with registered students"""
        if not self.students_database:
            print("No students registered yet!")
            return False

        X = []
        y = []

        for student_id, data in self.students_database.items():
            for face_data in data['face_data']:
                X.append(face_data)
                y.append(student_id)

        X = np.array(X)
        y = np.array(y)

        self.face_model.fit(X, y)
        print(f"Face recognition model trained with {len(X)} samples from {len(self.students_database)} students")

        self.save_models()
        return True

    def train_emotion_model(self, train_data_path=None):
        """Train the emotion detection model"""
        # This would typically use FER2013 dataset or similar
        # For demo purposes, we'll create a simple training loop
        print("Training emotion detection model...")

        # In a real implementation, you would load FER2013 dataset here
        # For now, we'll just compile the model
        if self.emotion_model is None:
            self.emotion_model = self.create_emotion_model()

        print("Emotion model ready for training!")
        # Note: In real implementation, add actual training data loading and training loop
        return True

    def recognize_face(self, face_roi):
        """Recognize a face using the trained model"""
        if self.face_model is None:
            return None, 0.0

        face_processed = self.preprocess_face(face_roi)
        face_reshaped = face_processed.reshape(1, -1)

        # Get prediction and confidence
        prediction = self.face_model.predict(face_reshaped)[0]
        distances, indices = self.face_model.kneighbors(face_reshaped)
        confidence = 1.0 / (1.0 + distances[0][0])  # Convert distance to confidence

        return prediction, confidence

    def detect_emotion(self, face_roi):
        """Detect emotion in the face"""
        if self.emotion_model is None:
            # Fallback to DeepFace for emotion detection
            try:
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                dominant_emotion = result[0]['dominant_emotion']
                confidence = result[0]['emotion'][dominant_emotion] / 100.0
                return dominant_emotion, confidence
            except:
                return 'neutral', 0.5

        face_processed = self.preprocess_emotion(face_roi)
        prediction = self.emotion_model.predict(face_processed)[0]
        emotion_idx = np.argmax(prediction)
        confidence = prediction[emotion_idx]

        return self.emotion_labels[emotion_idx], confidence

    def mark_attendance(self):
        """Main function to mark attendance using camera"""
        if not self.check_time_window():
            return False

        if not self.students_database:
            print("No students registered! Please register students first.")
            return False

        cap = cv2.VideoCapture(0)
        print("Attendance system started. Press 'q' to quit.")

        marked_today = set()  # Track students already marked today

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            # Check time window periodically
            if not self.check_time_window():
                break

            faces, gray = self.detect_faces(frame)

            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]

                # Recognize face
                student_id, face_confidence = self.recognize_face(face_roi)

                # Detect emotion
                emotion, emotion_confidence = self.detect_emotion(face_roi)

                # Draw rectangle around face
                color = (0, 255, 0) if face_confidence > 0.6 else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

                # Display information
                if student_id and face_confidence > 0.6:
                    student_name = self.students_database[student_id]['name']

                    # Mark attendance if not already marked today
                    today = datetime.now().date()
                    if student_id not in marked_today:
                        self.record_attendance(student_id, student_name, emotion, emotion_confidence)
                        marked_today.add(student_id)
                        print(f"Attendance marked for {student_name} with emotion: {emotion}")

                    # Display info on frame
                    cv2.putText(frame, f'{student_name} ({emotion})', 
                               (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.putText(frame, f'Conf: {face_confidence:.2f}', 
                               (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                else:
                    cv2.putText(frame, f'Unknown ({emotion})', 
                               (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow('Smart Attendance System', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        print("Attendance session completed!")
        return True

    def record_attendance(self, student_id, student_name, emotion, emotion_confidence):
        """Record attendance with timestamp and emotion"""
        timestamp = datetime.now()

        attendance_record = {
            'student_id': student_id,
            'student_name': student_name,
            'date': timestamp.date().isoformat(),
            'time': timestamp.time().isoformat(),
            'timestamp': timestamp.isoformat(),
            'emotion': emotion,
            'emotion_confidence': emotion_confidence,
            'status': 'Present'
        }

        self.attendance_records.append(attendance_record)

        # Save to CSV
        self.save_attendance_to_csv()

        return True

    def save_attendance_to_csv(self):
        """Save attendance records to CSV file"""
        if not self.attendance_records:
            return

        today = datetime.now().date().isoformat()
        filename = f'attendance_records/attendance_{today}.csv'

        df = pd.DataFrame(self.attendance_records)
        df.to_csv(filename, index=False)

        print(f"Attendance saved to {filename}")

        # Also save to Excel
        excel_filename = f'attendance_records/attendance_{today}.xlsx'
        df.to_excel(excel_filename, index=False)

        return filename

    def generate_attendance_report(self, start_date=None, end_date=None):
        """Generate comprehensive attendance report"""
        if not self.attendance_records:
            print("No attendance records found!")
            return None

        df = pd.DataFrame(self.attendance_records)

        if start_date:
            df = df[df['date'] >= start_date]
        if end_date:
            df = df[df['date'] <= end_date]

        # Summary statistics
        print("\n=== ATTENDANCE REPORT ===")
        print(f"Total Records: {len(df)}")
        print(f"Unique Students: {df['student_id'].nunique()}")
        print(f"Date Range: {df['date'].min()} to {df['date'].max()}")

        # Emotion analysis
        print("\n=== EMOTION ANALYSIS ===")
        emotion_counts = df['emotion'].value_counts()
        for emotion, count in emotion_counts.items():
            percentage = (count / len(df)) * 100
            print(f"{emotion.capitalize()}: {count} ({percentage:.1f}%)")

        # Student-wise attendance
        print("\n=== STUDENT-WISE ATTENDANCE ===")
        student_attendance = df.groupby('student_name').agg({
            'date': 'nunique',
            'emotion': lambda x: x.mode().iloc[0] if not x.empty else 'neutral'
        }).rename(columns={'date': 'days_present', 'emotion': 'most_common_emotion'})

        for student, data in student_attendance.iterrows():
            print(f"{student}: {data['days_present']} days, Most common emotion: {data['most_common_emotion']}")

        return df

    def save_models(self):
        """Save trained models"""
        try:
            if self.face_model:
                with open('models/face_recognition_model.pkl', 'wb') as f:
                    pickle.dump(self.face_model, f)

            if self.emotion_model:
                self.emotion_model.save('models/emotion_model.h5')

            # Save students database
            with open('models/students_database.json', 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                db_copy = {}
                for k, v in self.students_database.items():
                    db_copy[k] = {
                        'name': v['name'],
                        'registered_on': v['registered_on'],
                        'face_data': [face.tolist() for face in v['face_data']]
                    }
                json.dump(db_copy, f, indent=2)

            print("Models and database saved successfully!")
            return True
        except Exception as e:
            print(f"Error saving models: {e}")
            return False

    def load_models(self):
        """Load pre-trained models"""
        try:
            # Load face recognition model
            with open('models/face_recognition_model.pkl', 'rb') as f:
                self.face_model = pickle.load(f)

            # Load emotion model
            self.emotion_model = tf.keras.models.load_model('models/emotion_model.h5')

            # Load students database
            with open('models/students_database.json', 'r') as f:
                db_data = json.load(f)
                self.students_database = {}
                for k, v in db_data.items():
                    self.students_database[k] = {
                        'name': v['name'],
                        'registered_on': v['registered_on'],
                        'face_data': [np.array(face) for face in v['face_data']]
                    }

            print("Models and database loaded successfully!")
            return True
        except FileNotFoundError:
            print("Models not found. Please train the models first.")
            return False
        except Exception as e:
            print(f"Error loading models: {e}")
            return False

# Usage example and main execution
def main():
    """Main function to demonstrate the attendance system"""

    # Initialize the system
    system = SmartAttendanceSystem()

    print("\n=== SMART ATTENDANCE SYSTEM ===")
    print("1. Register new student")
    print("2. Start attendance marking")
    print("3. Generate attendance report")
    print("4. Exit")

    while True:
        choice = input("\nEnter your choice (1-4): ").strip()

        if choice == '1':
            student_id = input("Enter student ID: ").strip()
            student_name = input("Enter student name: ").strip()

            if student_id and student_name:
                system.register_student(student_id, student_name)
            else:
                print("Please provide valid student ID and name.")

        elif choice == '2':
            print("Starting attendance marking...")
            system.mark_attendance()

        elif choice == '3':
            system.generate_attendance_report()

        elif choice == '4':
            print("Saving models and exiting...")
            system.save_models()
            print("Goodbye!")
            break

        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
