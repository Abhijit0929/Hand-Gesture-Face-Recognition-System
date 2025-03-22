# Hand Gesture & Face Recognition System

## Overview
This project integrates real-time hand gesture recognition, face recognition, and object detection using OpenCV, Mediapipe, and YOLO. The system can:
- Detect and recognize known faces
- Identify hand gestures such as Thumbs Up, Thumbs Down, Open Palm, Fist, Peace Sign, and Pointing Up
- Perform object detection using the YOLO model
- Provide voice feedback using text-to-speech

## Installation
### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd hand-gesture-recognition
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage
### 1. Run the Application
```bash
python src/main.py
```

### 2. Hand Gestures Recognized
- 👍 Thumbs Up
- 👎 Thumbs Down
- ✋ Open Palm
- ✊ Fist
- ✌ Peace Sign
- ☝ Pointing Up

### 3. Face Recognition
- Loads default face from `abhi.jpg`
- Detects new faces and allows adding new ones dynamically

### 4. Object Detection (Optional)
- Requires YOLO model files: `yolov4.weights`, `yolov4.cfg`, `coco.names`

## File Structure
```
hand-gesture-recognition/
│── models/                 # Pre-trained models
│── src/                    # Source code
│── configs/                # Configuration files
│── static/                 # Image storage
│── requirements.txt        # Dependencies
│── README.md               # Documentation
```

## Notes
- Ensure your webcam is properly connected

## Author
- GitHub: [Abhijit Dhumale ](https://github.com/Abhijit0929)
