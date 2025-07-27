
"""
Main Application File for Attendance System
Provides command-line interface for system operations
"""
import argparse
import sys
import os
import cv2
from datetime import datetime, timedelta
from attendance_system import AttendanceSystem
from face_recognition_module import FaceRecognizer
from config import Config


def setup_directories():
    """Setup required directories"""
    Config.create_directories()
    print(" Directory setup completed")

def add_student_interactive():
    """Interactive student addition with camera"""
    recognizer = FaceRecognizer()

    # Get student info
    student_id = input("Enter Student ID: ")
    student_name = input("Enter Student Name: ")

    print(f"\n Adding student: {student_name} ({student_id})")
    print("Position your face in front of the camera and press SPACE to capture")
    print("Press 'q' to quit, 'r' to retake current photo")

    # Initialize camera
    cap = cv2.VideoCapture(Config.CAMERA_INDEX)

    if not cap.isOpened():
        print(" Could not open camera")
        return

    captured_images = []
    photo_count = 0
    max_photos = 5

    try:
        while photo_count < max_photos:
            ret, frame = cap.read()
            if not ret:
                break

            # Display frame
            display_frame = frame.copy()
            cv2.putText(display_frame, f"Photo {photo_count + 1}/{max_photos}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, "SPACE: Capture, Q: Quit, R: Retake", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow('Add Student', display_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):  # Space to capture
                captured_images.append(frame.copy())
                photo_count += 1
                print(f" Photo {photo_count} captured")

            elif key == ord('r') and captured_images:  # R to retake
                captured_images.pop()
                photo_count -= 1
                print("↩ Last photo removed")

            elif key == ord('q'):  # Q to quit
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    # Add student if photos were captured
    if captured_images:
        recognizer.add_student(student_id, student_name, captured_images)
        print(f" Student {student_name} added with {len(captured_images)} photos")
    else:
        print(" No photos captured. Student not added.")

def add_student_from_files(student_id, student_name, image_paths):
    """Add student from existing image files"""
    recognizer = FaceRecognizer()

    # Verify image files exist
    valid_images = []
    for path in image_paths:
        if os.path.exists(path):
            valid_images.append(path)
        else:
            print(f" Image not found: {path}")

    if valid_images:
        recognizer.add_student(student_id, student_name, valid_images)
        print(f" Student {student_name} added with {len(valid_images)} images")
    else:
        print(" No valid images found. Student not added.")

def list_students():
    """List all registered students"""
    recognizer = FaceRecognizer()
    students = recognizer.get_all_students()

    if not students:
        print(" No students registered")
        return

    print(f"\n Registered Students ({len(students)}):")
    print("-" * 50)
    for student in students:
        print(f"ID: {student['id']:<10} Name: {student['name']}")
    print("-" * 50)

def remove_student(student_id):
    """Remove a student from the database"""
    recognizer = FaceRecognizer()
    recognizer.remove_student(student_id)

def run_attendance(no_display=False):
    """Run the attendance system"""
    system = AttendanceSystem()

    print(" Starting Attendance System...")
    print(f" Attendance window: {Config.ATTENDANCE_START_TIME} - {Config.ATTENDANCE_END_TIME}")
    print("Press 'q' to quit or Ctrl+C to interrupt")

    try:
        system.run_camera_attendance(display=not no_display)

        # Show summary
        print(system.get_attendance_summary())

    except KeyboardInterrupt:
        print("\n Attendance system interrupted")

def generate_report(start_date, end_date, student_id=None, output_file=None):
    """Generate attendance report for date range"""
    # This would read from saved attendance files
    attendance_dir = Config.ATTENDANCE_DIR

    if not os.path.exists(attendance_dir):
        print(" No attendance data found")
        return

    print(f" Generating report from {start_date} to {end_date}")

    # Implementation would read CSV files and generate report
    # For now, just list available files
    files = [f for f in os.listdir(attendance_dir) if f.endswith('.csv')]

    if files:
        print(f" Found {len(files)} attendance files:")
        for file in sorted(files):
            print(f"  - {file}")
    else:
        print(" No attendance files found")

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description='Student Attendance System')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Setup directories')

    # Add student commands
    add_parser = subparsers.add_parser('add-student', help='Add new student')
    add_parser.add_argument('student_id', help='Student ID')
    add_parser.add_argument('student_name', help='Student name')
    add_parser.add_argument('--images', help='Comma-separated image paths')
    add_parser.add_argument('--interactive', action='store_true', 
                          help='Use camera for interactive capture')

    # List students command
    list_parser = subparsers.add_parser('list-students', help='List registered students')

    # Remove student command
    remove_parser = subparsers.add_parser('remove-student', help='Remove student')
    remove_parser.add_argument('student_id', help='Student ID to remove')

    # Run attendance command
    run_parser = subparsers.add_parser('run', help='Run attendance system')
    run_parser.add_argument('--no-display', action='store_true',
                          help='Run without video display')

    # Generate report command
    report_parser = subparsers.add_parser('report', help='Generate attendance report')
    report_parser.add_argument('start_date', help='Start date (YYYY-MM-DD)')
    report_parser.add_argument('end_date', help='End date (YYYY-MM-DD)')
    report_parser.add_argument('--student-id', help='Filter by student ID')
    report_parser.add_argument('--output', help='Output file path')

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Execute commands
    try:
        if args.command == 'setup':
            setup_directories()

        elif args.command == 'add-student':
            if args.interactive or not args.images:
                add_student_interactive()
            else:
                image_paths = args.images.split(',')
                add_student_from_files(args.student_id, args.student_name, image_paths)

        elif args.command == 'list-students':
            list_students()

        elif args.command == 'remove-student':
            remove_student(args.student_id)

        elif args.command == 'run':
            run_attendance(args.no_display)

        elif args.command == 'report':
            generate_report(args.start_date, args.end_date, 
                          args.student_id, args.output)

    except KeyboardInterrupt:
        print("\n Operation interrupted by user")
    except Exception as e:
        print(f" Error: {e}")

if __name__ == "__main__":
    main()
