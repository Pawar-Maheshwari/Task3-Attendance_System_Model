
"""
Main Attendance System Module
Integrates face detection, recognition, and emotion detection
"""
import cv2
import pandas as pd
import numpy as np
from datetime import datetime, time
import os
from config import Config
from face_detection import FaceDetector
from face_recognition_module import FaceRecognizer
from emotion_detection import EmotionDetector

class AttendanceSystem:
    def __init__(self):
        # Initialize components
        self.face_detector = FaceDetector()
        self.face_recognizer = FaceRecognizer()
        self.emotion_detector = EmotionDetector()

        # Initialize attendance data
        self.attendance_records = []
        self.detected_students = set()  # Track already detected students

        # Create directories
        Config.create_directories()

        print(" Attendance System initialized")

    def is_attendance_time(self):
        """Check if current time is within attendance window"""
        current_time = datetime.now().time()
        return Config.ATTENDANCE_START_TIME <= current_time <= Config.ATTENDANCE_END_TIME

    def process_frame(self, frame):
        """
        Process a single frame for attendance

        Args:
            frame: Input video frame

        Returns:
            Processed frame with annotations
        """
        # Check if it's attendance time
        if not self.is_attendance_time():
            cv2.putText(frame, "Outside attendance hours", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return frame

        # Detect faces
        faces = self.face_detector.detect_faces(frame)

        if not faces:
            cv2.putText(frame, "No faces detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            return frame

        # Process each detected face
        for face in faces:
            # Extract face region
            face_box = face['box']
            face_image = self.face_detector.extract_face(frame, face_box)

            if face_image.size == 0:
                continue

            # Recognize face
            recognition_result = self.face_recognizer.recognize_face(face_image)

            # Detect emotion
            emotion_result = self.emotion_detector.detect_emotion(face_image)

            # Draw face box
            color = (0, 255, 0) if recognition_result['recognized'] else (0, 0, 255)
            self.face_detector.draw_face_box(frame, face, color=color)

            # Prepare display text
            if recognition_result['recognized']:
                student_name = recognition_result['student_name']
                student_id = recognition_result['student_id']
                face_conf = recognition_result['confidence']

                # Check if student already marked present
                if student_id not in self.detected_students:
                    # Mark attendance
                    self.mark_attendance(
                        student_id=student_id,
                        student_name=student_name,
                        emotion=emotion_result['emotion'],
                        emotion_confidence=emotion_result['confidence'],
                        face_confidence=face_conf
                    )
                    self.detected_students.add(student_id)

                # Draw student info
                text = f"{student_name} ({student_id})"
                status = "PRESENT"
            else:
                text = "Unknown Student"
                status = "NOT RECOGNIZED"

            # Draw text information
            x, y, w, h = face_box
            cv2.putText(frame, text, (x, y - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, status, (x, y - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Draw emotion info
            emotion_text = f"{emotion_result['emotion']}: {emotion_result['confidence']:.2f}"
            emotion_color = self.emotion_detector.get_emotion_color(emotion_result['emotion'])
            cv2.putText(frame, emotion_text, (x, y + h + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, emotion_color, 2)

        # Add timestamp and system info
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        detected_count = len(self.detected_students)
        cv2.putText(frame, f"Present: {detected_count}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return frame

    def mark_attendance(self, student_id, student_name, emotion, emotion_confidence, face_confidence):
        """
        Mark attendance for a student

        Args:
            student_id: Student ID
            student_name: Student name
            emotion: Detected emotion
            emotion_confidence: Emotion detection confidence
            face_confidence: Face recognition confidence
        """
        current_time = datetime.now()

        attendance_record = {
            'student_id': student_id,
            'name': student_name,
            'status': 'Present',
            'time': current_time.strftime("%H:%M:%S"),
            'date': current_time.strftime("%Y-%m-%d"),
            'emotion': emotion,
            'emotion_confidence': round(emotion_confidence, 3),
            'face_confidence': round(face_confidence, 3),
            'timestamp': current_time
        }

        self.attendance_records.append(attendance_record)
        print(f"✅ Attendance marked: {student_name} ({student_id}) - {emotion}")

    def mark_absent_students(self):
        """Mark absent students at the end of attendance period"""
        all_students = self.face_recognizer.get_all_students()
        current_date = datetime.now()

        for student in all_students:
            if student['id'] not in self.detected_students:
                # Mark as absent
                attendance_record = {
                    'student_id': student['id'],
                    'name': student['name'],
                    'status': 'Absent',
                    'time': '',
                    'date': current_date.strftime("%Y-%m-%d"),
                    'emotion': '',
                    'emotion_confidence': 0.0,
                    'face_confidence': 0.0,
                    'timestamp': current_date
                }
                self.attendance_records.append(attendance_record)

        print(f" Marked {len(all_students) - len(self.detected_students)} students as absent")

    def export_attendance(self, filename=None):
        """
        Export attendance records to CSV and Excel

        Args:
            filename: Optional custom filename
        """
        if not self.attendance_records:
            print(" No attendance records to export")
            return

        # Create DataFrame
        df = pd.DataFrame(self.attendance_records)

        # Generate filename if not provided
        if filename is None:
            current_date = datetime.now().strftime("%Y-%m-%d")
            filename = f"attendance_{current_date}"

        # Export to CSV
        if Config.EXPORT_CSV:
            csv_path = os.path.join(Config.ATTENDANCE_DIR, f"{filename}.csv")
            df.to_csv(csv_path, index=False)
            print(f" Attendance exported to: {csv_path}")

        # Export to Excel
        if Config.EXPORT_EXCEL:
            excel_path = os.path.join(Config.ATTENDANCE_DIR, f"{filename}.xlsx")
            df.to_excel(excel_path, index=False)
            print(f" Attendance exported to: {excel_path}")

        return df

    def run_camera_attendance(self, display=True):
        """
        Run attendance system with camera input

        Args:
            display: Whether to display video feed
        """
        # Initialize camera
        cap = cv2.VideoCapture(Config.CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)

        if not cap.isOpened():
            print(" Could not open camera")
            return

        print(" Camera attendance started. Press 'q' to quit.")
        print(f" Attendance window: {Config.ATTENDANCE_START_TIME} - {Config.ATTENDANCE_END_TIME}")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print(" Could not read frame from camera")
                    break

                # Process frame
                processed_frame = self.process_frame(frame)

                # Display frame if requested
                if display:
                    cv2.imshow('Attendance System', processed_frame)

                # Check for quit command
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # Auto-quit if outside attendance hours
                if not self.is_attendance_time() and self.attendance_records:
                    print("⏰ Attendance period ended. Exporting data...")
                    break

        except KeyboardInterrupt:
            print("\n Attendance interrupted by user")

        finally:
            # Cleanup
            cap.release()
            if display:
                cv2.destroyAllWindows()

            # Mark absent students and export data
            self.mark_absent_students()
            self.export_attendance()

    def get_attendance_summary(self):
        """Get summary of attendance records"""
        if not self.attendance_records:
            return "No attendance records available"

        df = pd.DataFrame(self.attendance_records)

        summary = f"""
        === ATTENDANCE SUMMARY ===
        Total Students Processed: {len(df)}
        Present: {len(df[df['status'] == 'Present'])}
        Absent: {len(df[df['status'] == 'Absent'])}

        Emotion Distribution:
        {df['emotion'].value_counts().to_string() if 'emotion' in df.columns else 'No emotion data'}
        """

        return summary
