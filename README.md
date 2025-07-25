# ISL Hand Sign Recognition System

A real-time **Indian Sign Language (ISL) to Text Converter** that uses computer vision and deep learning to recognize hand gestures and convert them into readable text. This web-based application provides an accessible solution for ISL communication through advanced hand tracking and gesture recognition technology.

## 🚀 Features

### Core Functionality
- **Real-time Hand Detection**: Advanced hand tracking using MediaPipe with 21-point hand landmark detection
- **ISL Gesture Recognition**: Recognizes 36 different ISL gestures (A-Z alphabets and 0-9 numbers)
- **Live Video Feed**: Real-time camera integration with hand gesture visualization
- **Instant Text Conversion**: Single-shot and continuous prediction modes

### User Interface
- **Hand Sign to Text**: Capture and predict individual hand gestures
- **Real-time Text Mode**: Continuous gesture recognition with automatic text composition
- **Prediction Confidence**: Visual confidence meter showing prediction accuracy
- **Prediction History**: Track recent predictions with timestamps
- **Composed Text Area**: Accumulate predictions into readable text with copy functionality
- **Visual Hand Landmarks**: Real-time hand skeleton overlay on video feed

### Technical Features
- **Enhanced Accuracy**: Multi-frame prediction averaging for improved accuracy
- **Image Enhancement**: Automatic contrast and brightness optimization
- **Responsive Design**: Modern web interface with Font Awesome icons
- **Error Handling**: Comprehensive error management and user feedback
- **Background Processing**: Non-blocking prediction processing

## 🛠️ Tech Stack

### Backend
- **Framework**: Flask 2.3.3 (Python web framework)
- **Deep Learning**: TensorFlow 2.13.0 with Keras
- **Computer Vision**: OpenCV 4.8.1.78 for image processing
- **Hand Detection**: MediaPipe for real-time hand tracking
- **Model Architecture**: MobileNetV2 (pre-trained CNN for efficient inference)

### Frontend
- **Web Technologies**: HTML5, CSS3, JavaScript (ES6+)
- **Styling**: Custom CSS with Font Awesome 6.0.0 icons
- **Real-time Communication**: Fetch API for backend communication
- **Video Handling**: HTML5 Canvas and Video APIs


## 📋 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Webcam or camera device
- Modern web browser (Chrome, Firefox, Safari, Edge)

### Step 1: Clone the Repository
```bash
git clone https://github.com/aslam-03/ISL-Conversion.git
cd ISL-Conversion
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the Application
```bash
python app.py
```

### Step 4: Access the Web Interface
Open your browser and navigate to:
```
http://localhost:5000
```

## 🗂️ Project Structure

```
ISL-Conversion/
│
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── README.md                      # Project documentation
│
├── app/                           # Web application files
│   ├── templates/
│   │   └── index.html            # Main web interface
│   └── static/
│       ├── css/
│       │   └── style.css         # Frontend styling
│       └── js/
│           └── app.js            # Frontend JavaScript logic
│
├── model/                         # Trained models
│   ├── isl_mobilenetv2.h5       # Main MobileNetV2 model
│   └── trained_model.h5         # Alternative trained model
│
├── dataset/                       # Training and testing data
│   ├── original_images/          # Training dataset
│   │   ├── A/, B/, C/, ..., Z/   # Alphabet gesture folders
│   │   └── 0/, 1/, 2/, ..., 9/   # Number gesture folders
│   └── test_images/              # Testing dataset
│       └── [same structure as original_images]
│
├── src/                          # Source code for model training
│   ├── model.py                  # Model architecture and training
│   ├── process2.py               # Data preprocessing
│   ├── test.py                   # Model testing utilities
│   └── coversiton2.py            # Conversion utilities
│
└── .vscode/                      # VS Code configuration
```

## 🎯 How to Use

1. **Start the Application**: Run `python app.py` and open `http://localhost:5000`

2. **Camera Setup**: Click "Start Camera" to activate your webcam

3. **Single Prediction**: 
   - Show your hand gesture to the camera
   - Click "Hand Sign to Text" for individual predictions

4. **Real-time Mode**:
   - Click "Real-time Text" for continuous gesture recognition
   - Watch predictions appear automatically in the text area

5. **View Results**:
   - Check the prediction confidence meter
   - View prediction history with timestamps
   - Copy composed text using the "Copy" button

## 🔧 Model Information

- **Architecture**: MobileNetV2-based transfer learning
- **Input Size**: 250×250×3 RGB images
- **Output Classes**: 36 categories (A-Z, 0-9)
- **Training Data**: Custom ISL gesture dataset
- **Accuracy**: Optimized for real-time performance with confidence thresholds

## 📞 Contact Information

**Developer**: Mohamed Aslam I

- **Email**: [aslamachu8558@gmail.com](mailto:aslamachu8558@gmail.com)
- **LinkedIn**: [linkedin.com/in/mohammed-aslam](www.linkedin.com/in/mohamed-aslam-i)
- **GitHub**: [github.com/aslam-03](https://github.com/aslam-03)

---
