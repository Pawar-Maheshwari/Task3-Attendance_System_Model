# Smart Attendance System with Face Recognition and Emotion Detection

A comprehensive machine learning-based attendance system that uses face recognition to identify students and emotion detection to analyze their emotional state during attendance marking.

## Features

- **Face Recognition**: Identifies registered students using trained ML models
- **Emotion Detection**: Analyzes student emotions (happy, sad, angry, neutral, etc.)
- **Time-Based Operation**: Only works during specified time windows (9:30 AM - 10:00 AM)
- **Data Storage**: Saves attendance records in CSV and Excel formats
- **Real-time Processing**: Live camera feed processing
- **Comprehensive Reporting**: Generates detailed attendance and emotion analysis reports

## System Requirements

- Python 3.7+
- Webcam or camera
- Minimum 4GB RAM
- 2GB free disk space

## Installation

1. Clone or download the project files
2. Run the setup script:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

3. Or install manually:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Running the System

```bash
python smart_attendance_system.py
```

### 2. Registering Students

- Select option 1 from the menu
- Enter student ID and name
- Look at the camera to capture face images
- System will automatically train the recognition model

### 3. Taking Attendance

- Select option 2 from the menu
- Students look at the camera
- System identifies faces and detects emotions
- Attendance is automatically recorded with timestamps

### 4. Generating Reports

- Select option 3 from the menu
- View comprehensive attendance and emotion analysis
- Reports are saved in CSV/Excel format

## Model Training

### Face Recognition Model

The system uses a K-Nearest Neighbors (KNN) classifier for face recognition:
- Trained on student face images during registration
- Uses facial feature vectors for identification
- Confidence threshold filtering for accuracy

### Emotion Detection Model

Two options for emotion detection:

1. **Custom CNN Model** (train_emotion_model.py):
   - Download FER2013 dataset
   - Run training script
   - Achieves ~70% accuracy on 7 emotions

2. **DeepFace Library** (fallback):
   - Pre-trained models
   - Real-time emotion analysis
   - No training required

## File Structure

```
├── smart_attendance_system.py    # Main system file
├── train_emotion_model.py        # Emotion model training
├── prepare_data.py              # Data preparation utilities
├── requirements.txt             # Python dependencies
├── setup.sh                     # Installation script
├── config/
│   └── system_config.json       # System configuration
├── models/                      # Trained models
├── data/                        # Training and student data
├── attendance_records/          # CSV/Excel attendance files
└── logs/                        # System logs
```

## Configuration

Edit `config/system_config.json` to modify:
- Attendance time window
- Recognition thresholds
- Emotion detection settings
- Data storage options

## Time-Based Operation

The system only operates during configured hours:
- Default: 9:30 AM - 10:00 AM
- Prevents unauthorized access
- Configurable in system settings

## Data Storage

Attendance records include:
- Student ID and name
- Date and time of attendance
- Detected emotion and confidence
- Attendance status

Formats supported:
- CSV files (daily)
- Excel files (daily)
- JSON exports (optional)

## Troubleshooting

### Common Issues

1. **Camera not detected**:
   - Check camera permissions
   - Ensure no other applications using camera

2. **Low face recognition accuracy**:
   - Register with more face images
   - Ensure good lighting conditions
   - Check camera quality

3. **Emotion detection errors**:
   - Install DeepFace: `pip install deepface`
   - Check model file integrity
   - Verify TensorFlow installation

### Performance Optimization

- Use GPU for faster emotion detection
- Optimize image resolution for speed
- Reduce number of detection frames per second

## Technical Details

### Face Recognition Pipeline

1. **Face Detection**: Haar Cascade Classifier
2. **Preprocessing**: Resize to 100x100 pixels
3. **Feature Extraction**: Flatten image data
4. **Classification**: KNN with k=5
5. **Confidence Filtering**: Threshold = 0.6

### Emotion Detection Pipeline

1. **Face Extraction**: From detected face region
2. **Preprocessing**: Resize to 48x48, normalize
3. **CNN Inference**: 7-class emotion classification
4. **Confidence Scoring**: Softmax probabilities

### Machine Learning Models

- **Face Recognition**: Scikit-learn KNN
- **Emotion Detection**: TensorFlow/Keras CNN
- **Training Data**: FER2013 + Student images
- **Performance**: ~90% face recognition, ~70% emotion accuracy

## Future Enhancements

- Multi-face simultaneous detection
- Advanced emotion categories
- Integration with student management systems
- Mobile app interface
- Cloud-based deployment
- Anti-spoofing mechanisms

## License

This project is for educational purposes. Please ensure compliance with privacy laws and institutional policies when deploying.

## Support

For technical support or questions:
1. Check troubleshooting section
2. Review system logs in `logs/` directory
3. Verify all dependencies are installed correctly

## Contributing

Contributions welcome! Areas for improvement:
- Better emotion detection models
- UI/UX enhancements
- Performance optimizations
- Additional security features
