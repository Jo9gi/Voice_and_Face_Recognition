# Voice and Face Recognition System

A comprehensive dual-factor biometric authentication system combining voice recognition (GMM) and face recognition (FaceNet) for secure access control.

## 🚀 Features

- **Dual Authentication**: Voice + Face recognition for enhanced security
- **Real-time Processing**: Live camera feed with instant recognition
- **Multiple Users**: Support for multiple user profiles
- **High Accuracy**: FaceNet-based face recognition with MFCC voice features
- **Easy Setup**: Simple installation and user registration process

## 📋 Requirements

- **Python 3.7.x** (Critical - newer versions incompatible with TensorFlow 1.14)
- **Webcam** for face recognition
- **Microphone** for voice authentication
- **Windows/Linux/macOS**
- **4GB+ RAM** recommended

## 🛠️ Installation

### Step 1: Install Python 3.7
```bash
# Download from: https://www.python.org/downloads/release/python-379/
# Or use conda:
conda create -n voice_face_env python=3.7
conda activate voice_face_env
```

### Step 2: Clone Repository
```bash
git clone https://github.com/Jo9gi/Voice_and_Face_Recognition.git
cd Voice_and_Face_Recognition
```

### Step 3: Install Dependencies
```bash
# Install main requirements
pip install -r requirement.txt

# Install audio dependencies
pip install sounddevice soundfile

# Fix protobuf compatibility (if needed)
pip install protobuf==3.20.3
```

### Step 4: Verify Installation
```bash
# Test face recognition
python test_face_only.py
```

## 🎯 Quick Start Guide

### 1. Register Your First User

**Add Face Data:**
```bash
python add_face_only.py
# Enter your name when prompted
# Position face in camera frame
# Wait for "Face captured!" message
```

**Add Voice Data:**
```bash
python add_voice_only.py
# Enter the SAME name as face registration
# Speak clearly when prompted (3 seconds)
# Repeat 8 times for better accuracy
```

**Train Voice Model:**
```bash
python train_voice_model.py
# Trains GMM model for voice recognition
# Creates .gmm file in gmm_models/ folder
```

### 2. Test Individual Systems

**Test Face Recognition:**
```bash
python test_face_only.py
# Shows live camera feed with recognition
# Press 'q' to quit
```

**Test Full Authentication:**
```bash
python full_recognition.py
# Step 1: Voice authentication (speak your name)
# Step 2: Face authentication (show your face)
# Both must match for access
```

## 📁 Project Structure

```
Voice_and_Face_Recognition/
├── 📄 Core Files
│   ├── face_functions.py          # FaceNet model integration
│   ├── voice_functions.py         # Audio processing & MFCC extraction
│   ├── full_recognition.py        # Main dual-auth system
│   ├── add_face_only.py          # Face registration utility
│   ├── add_voice_only.py         # Voice recording utility
│   ├── train_voice_model.py      # GMM model training
│   └── test_face_only.py         # Face recognition testing
│
├── 🤖 Models & Data
│   ├── facenet_model/            # Pre-trained FaceNet model
│   │   └── model.h5              # 128D face embedding model
│   ├── haarcascades/             # Face detection classifier
│   │   └── haarcascade_frontalface_default.xml
│   ├── face_database/            # 🔒 Face embeddings (auto-created)
│   │   └── embeddings.pickle     # Stored face encodings
│   ├── voice_database/           # 🔒 Voice samples (auto-created)
│   │   └── [username]/           # User voice samples
│   └── gmm_models/              # 🔒 Trained voice models (auto-created)
│       └── [username].gmm       # User voice model
│
├── 📋 Configuration
│   ├── requirement.txt          # Python dependencies
│   ├── .gitignore              # Git ignore rules
│   └── README.md               # This file
│
└── 📁 Temp Folders (auto-created)
    └── saved_image/            # Temporary image storage
```

## 🔧 How It Works

### Face Recognition Pipeline
1. **Detection**: Haar Cascade detects faces in camera feed
2. **Preprocessing**: Resize to 160x160, normalize pixels
3. **Encoding**: FaceNet generates 128-dimensional embedding
4. **Matching**: Compare with stored embeddings using Euclidean distance
5. **Threshold**: Distance < 10.0 = Match

### Voice Recognition Pipeline
1. **Recording**: Capture 3-second audio samples
2. **Feature Extraction**: Extract 40-dimensional MFCC features
3. **Training**: Train Gaussian Mixture Model (GMM) per user
4. **Recognition**: Score audio against all user models
5. **Decision**: Highest scoring model = Recognized user

### Dual Authentication Flow
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Voice     │───▶│    Face     │───▶│   Access    │
│ Recognition │    │ Recognition │    │  Decision   │
└─────────────┘    └─────────────┘    └─────────────┘
      ✓                   ✓                 ✅ GRANT
      ✓                   ✗                 ❌ DENY
      ✗                   ✓                 ❌ DENY
      ✗                   ✗                 ❌ DENY
```

## 🎛️ Configuration & Tuning

### Face Recognition Threshold
```python
# In test_face_only.py and full_recognition.py
if min_dist <= 10.0:  # Adjust this value
    # Lower = stricter matching
    # Higher = more lenient matching
```

### Voice Model Training
```python
# In train_voice_model.py
n_components=16  # Number of Gaussian components
# More components = better accuracy, slower training
```

## 🚨 Troubleshooting

### Common Issues & Solutions

**❌ "ModuleNotFoundError: No module named 'cv2'"**
```bash
pip install opencv-python==4.1.2.30
```

**❌ "ModuleNotFoundError: No module named 'sounddevice'"**
```bash
pip install sounddevice soundfile
```

**❌ "Descriptors cannot be created directly"**
```bash
pip install protobuf==3.20.3
```

**❌ "Camera not working"**
- Close other applications using camera
- Check camera permissions
- Try different camera index: `cv2.VideoCapture(1)`

**❌ "Face shows as Unknown"**
- Ensure good lighting during registration
- Re-register face with `python add_face_only.py`
- Adjust threshold in code if needed

**❌ "Voice not recognized"**
- Speak clearly and consistently
- Record in quiet environment
- Ensure same name for face and voice registration

### Performance Tips

- **Better Face Recognition**: Register face in good lighting, front-facing
- **Better Voice Recognition**: Record 8+ samples, speak consistently
- **Faster Processing**: Close unnecessary applications
- **Multiple Users**: Use distinct names, avoid similar voices

## 🔒 Security Features

- **Dual-Factor Authentication**: Both biometrics required
- **Liveness Detection**: Real-time camera feed prevents photo attacks
- **Encrypted Storage**: Biometric data stored as mathematical embeddings
- **Configurable Thresholds**: Adjustable security levels
- **Multi-User Support**: Isolated user profiles

## 📊 Technical Specifications

| Component | Technology | Details |
|-----------|------------|----------|
| Face Recognition | FaceNet | 128D embeddings, Triplet loss |
| Voice Recognition | GMM + MFCC | 40D features, Gaussian mixture |
| Face Detection | Haar Cascade | OpenCV implementation |
| Deep Learning | TensorFlow 1.14 | Keras 2.2.4 |
| Audio Processing | SoundDevice | Real-time recording |
| Image Processing | OpenCV 4.1.2 | Computer vision |

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [FaceNet Paper](https://arxiv.org/pdf/1503.03832.pdf) - Face recognition architecture
- [Keras OpenFace](https://github.com/iwantooxxoox/Keras-OpenFace) - Implementation reference
- [DeepLearning.ai](https://www.coursera.org/learn/convolutional-neural-networks) - Educational foundation

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify Python 3.7.x installation
3. Ensure all dependencies are installed
4. Create an issue on GitHub with error details

---

**⭐ Star this repository if it helped you!**