
"""
Setup script for Attendance System
"""
import os
import sys
import subprocess
from config import Config

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False
    return True

def create_directories():
    """Create necessary directories"""
    print("📁 Creating project directories...")
    Config.create_directories()

def download_instructions():
    """Display download instructions for large files"""
    print("\n📋 DOWNLOAD INSTRUCTIONS FOR LARGE FILES:")
    print("-" * 50)
    print("If model files are too large for GitHub, download from Google Drive:")
    print("")
    print("1. FER2013 Dataset:")
    print("   - Download from: https://www.kaggle.com/datasets/msambare/fer2013")
    print("   - Place fer2013.csv in: data/")
    print("")
    print("2. Pre-trained Models (if provided):")
    print("   - emotion_model.h5 → models/")
    print("   - face_embeddings.pkl → models/")
    print("   - student_info.pkl → models/")
    print("")
    print("3. If Google Drive links are provided in README:")
    print("   - Click the links to download")
    print("   - Place files in their respective directories")
    print("-" * 50)

def setup_sample_data():
    """Create sample data structure"""
    print("\n📁 Creating sample data structure...")

    # Create sample student directory structure
    sample_student_dir = os.path.join(Config.STUDENTS_DIR, "STU001_John_Doe")
    os.makedirs(sample_student_dir, exist_ok=True)

    # Create a sample info file
    sample_info = '''Sample Student Directory Structure:

students/
├── STU001_John_Doe/
│   ├── photo1.jpg
│   ├── photo2.jpg
│   └── photo3.jpg
└── STU002_Jane_Smith/
    ├── image1.png
    └── image2.png

Instructions:
1. Create a directory for each student using format: StudentID_StudentName
2. Add 3-5 clear photos of the student's face
3. Supported formats: .jpg, .jpeg, .png, .bmp
4. Ensure good lighting and clear face visibility
'''

    with open(os.path.join(Config.STUDENTS_DIR, "README_STUDENT_SETUP.txt"), "w") as f:
        f.write(sample_info)

    print("✅ Sample data structure created")

def run_tests():
    """Run basic system tests"""
    print("\n🧪 Running basic system tests...")

    # Test imports
    try:
        import cv2
        import tensorflow as tf
        import mtcnn
        import face_recognition
        print("✅ All required libraries imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

    # Test camera
    try:
        cap = cv2.VideoCapture(Config.CAMERA_INDEX)
        if cap.isOpened():
            print("✅ Camera test passed")
            cap.release()
        else:
            print("⚠️ Camera not available - check camera index in config.py")
    except Exception as e:
        print(f"⚠️ Camera test failed: {e}")

    return True

def main():
    """Main setup function"""
    print("🚀 ATTENDANCE SYSTEM SETUP")
    print("=" * 40)

    # Install requirements
    if not install_requirements():
        print("❌ Setup failed during package installation")
        return

    # Create directories
    create_directories()

    # Setup sample data
    setup_sample_data()

    # Run tests
    if not run_tests():
        print("⚠️ Some tests failed, but setup can continue")

    # Show download instructions
    download_instructions()

    print("\n🎯 NEXT STEPS:")
    print("-" * 30)
    print("1. Download FER2013 dataset and place in data/")
    print("2. Run: jupyter notebook model_training.ipynb")
    print("3. Add student photos to data/students/")
    print("4. Run: python main.py add-student <ID> <name> --interactive")
    print("5. Run: python main.py run")
    print("\nOptional:")
    print("- GUI version: python gui.py")
    print("- Help: python main.py --help")

    print("\n✅ Setup completed successfully!")

if __name__ == "__main__":
    main()
