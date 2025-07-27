
"""
Face Recognition Module using face_recognition library
"""
import pickle
import numpy as np
import face_recognition
import cv2
from config import Config

class FaceRecognizer:
    def __init__(self):
        self.face_embeddings = {}
        self.student_info = {}
        self.threshold = Config.FACE_RECOGNITION_THRESHOLD
        self.load_embeddings()

    def load_embeddings(self):
        """Load face embeddings and student info from files"""
        try:
            with open(Config.FACE_EMBEDDINGS_PATH, 'rb') as f:
                self.face_embeddings = pickle.load(f)

            with open(Config.STUDENT_INFO_PATH, 'rb') as f:
                self.student_info = pickle.load(f)

            print(f"✅ Loaded embeddings for {len(self.face_embeddings)} students")
        except FileNotFoundError:
            print("⚠️ No existing embeddings found. Please train the model first.")

    def save_embeddings(self):
        """Save face embeddings and student info to files"""
        with open(Config.FACE_EMBEDDINGS_PATH, 'wb') as f:
            pickle.dump(self.face_embeddings, f)

        with open(Config.STUDENT_INFO_PATH, 'wb') as f:
            pickle.dump(self.student_info, f)

        print("✅ Embeddings saved successfully")

    def add_student(self, student_id, student_name, images):
        """
        Add a new student to the database

        Args:
            student_id: Unique student identifier
            student_name: Student's full name
            images: List of image paths or numpy arrays
        """
        embeddings = []

        for image in images:
            if isinstance(image, str):
                # Load image from path
                img = face_recognition.load_image_file(image)
            else:
                # Use numpy array directly
                img = image

            # Get face encodings
            face_encodings = face_recognition.face_encodings(img)

            if face_encodings:
                embeddings.append(face_encodings[0])
            else:
                print(f"⚠️ No face found in image")

        if embeddings:
            # Store average embedding
            avg_embedding = np.mean(embeddings, axis=0)
            student_key = f"{student_id}_{student_name.replace(' ', '_')}"

            self.face_embeddings[student_key] = avg_embedding
            self.student_info[student_key] = {
                'id': student_id,
                'name': student_name
            }

            self.save_embeddings()
            print(f"✅ Student {student_name} ({student_id}) added successfully")
        else:
            print(f"❌ Failed to add student {student_name} - no valid faces found")

    def recognize_face(self, face_image):
        """
        Recognize a face from the given image

        Args:
            face_image: Face image (numpy array)

        Returns:
            Dictionary with recognition results
        """
        if not self.face_embeddings:
            return {
                'recognized': False,
                'student_id': None,
                'student_name': 'Unknown',
                'confidence': 0.0
            }

        # Get face encoding
        face_encodings = face_recognition.face_encodings(face_image)

        if not face_encodings:
            return {
                'recognized': False,
                'student_id': None,
                'student_name': 'No Face Detected',
                'confidence': 0.0
            }

        face_encoding = face_encodings[0]

        # Compare with known faces
        best_match = None
        best_distance = float('inf')

        for student_key, known_encoding in self.face_embeddings.items():
            # Calculate face distance
            distance = face_recognition.face_distance([known_encoding], face_encoding)[0]

            if distance < best_distance:
                best_distance = distance
                best_match = student_key

        # Check if match is within threshold
        if best_match and best_distance < self.threshold:
            student_info = self.student_info[best_match]
            confidence = 1.0 - best_distance  # Convert distance to confidence

            return {
                'recognized': True,
                'student_id': student_info['id'],
                'student_name': student_info['name'],
                'confidence': confidence
            }
        else:
            return {
                'recognized': False,
                'student_id': None,
                'student_name': 'Unknown',
                'confidence': 0.0
            }

    def get_all_students(self):
        """Get list of all registered students"""
        students = []
        for student_key, info in self.student_info.items():
            students.append({
                'key': student_key,
                'id': info['id'],
                'name': info['name']
            })
        return students

    def remove_student(self, student_id):
        """Remove a student from the database"""
        to_remove = None
        for student_key, info in self.student_info.items():
            if info['id'] == student_id:
                to_remove = student_key
                break

        if to_remove:
            del self.face_embeddings[to_remove]
            del self.student_info[to_remove]
            self.save_embeddings()
            print(f"✅ Student {student_id} removed successfully")
        else:
            print(f"❌ Student {student_id} not found")
