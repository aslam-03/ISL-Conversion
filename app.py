from flask import Flask, render_template, request, jsonify, Response
import cv2
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
import base64
import io
from PIL import Image
import threading
import os
import mediapipe as mp
import time

app = Flask(__name__, 
            template_folder=os.path.abspath('frontend/templates'), 
            static_folder=os.path.abspath('frontend/static'))

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'backend', 'model', 'isl_mobilenetv2.h5')
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Define categories (same as during training)
categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
              'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# Initialize MediaPipe hands with better accuracy settings
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.8,  # Increased for better accuracy
    min_tracking_confidence=0.7    # Increased for better tracking
)

def extract_hand_region(frame):
    """Extract hand region using MediaPipe for focused prediction"""
    try:
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame with MediaPipe
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            # Get the first hand landmarks
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Get frame dimensions
            h, w, _ = frame.shape
            
            # Calculate bounding box around hand with improved accuracy
            x_coords = [landmark.x for landmark in hand_landmarks.landmark]
            y_coords = [landmark.y for landmark in hand_landmarks.landmark]
            
            # Use smaller padding for better hand focus
            padding = 0.15
            x_min = max(0, int((min(x_coords) - padding) * w))
            x_max = min(w, int((max(x_coords) + padding) * w))
            y_min = max(0, int((min(y_coords) - padding) * h))
            y_max = min(h, int((max(y_coords) + padding) * h))
            
            # Ensure we have a square region for consistent model input
            width = x_max - x_min
            height = y_max - y_min
            size = max(width, height)
            
            # Make sure the square is not too small
            min_size = min(w, h) // 4
            size = max(size, min_size)
            
            # Center the square region
            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2
            
            x_min = max(0, center_x - size // 2)
            x_max = min(w, center_x + size // 2)
            y_min = max(0, center_y - size // 2)
            y_max = min(h, center_y + size // 2)
            
            # Extract hand region
            hand_region = frame[y_min:y_max, x_min:x_max]
            
            # Resize to model input size with high quality interpolation
            if hand_region.size > 0:
                hand_region = cv2.resize(hand_region, (250, 250), interpolation=cv2.INTER_CUBIC)
                return hand_region, True
        
        # If no hand detected, return original frame resized
        return cv2.resize(frame, (250, 250), interpolation=cv2.INTER_CUBIC), False
        
    except Exception as e:
        print(f"Error extracting hand region: {e}")
        return cv2.resize(frame, (250, 250), interpolation=cv2.INTER_CUBIC), False

def preprocess_frame(frame):
    """Preprocess the frame for model prediction"""
    try:
        # Resize to the same size used in training
        img = cv2.resize(frame, (250, 250))
        # Normalize pixel values
        img = img / 255.0
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        return img
    except Exception as e:
        print(f"Error preprocessing frame: {e}")
        return None

def draw_hand_landmarks(frame):
    """Draw hand landmarks and bounding box on frame like the reference code"""
    try:
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame with MediaPipe
        results = hands.process(rgb_frame)
        
        # Create a copy of the frame to draw on
        annotated_frame = frame.copy()
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks and connections
                mp_drawing.draw_landmarks(
                    annotated_frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
                
                # Get frame dimensions
                h, w, _ = frame.shape
                
                # Calculate bounding box around hand
                x_coords = [landmark.x for landmark in hand_landmarks.landmark]
                y_coords = [landmark.y for landmark in hand_landmarks.landmark]
                
                # Add padding around hand
                padding = 0.1
                x_min = max(0, int((min(x_coords) - padding) * w))
                x_max = min(w, int((max(x_coords) + padding) * w))
                y_min = max(0, int((min(y_coords) - padding) * h))
                y_max = min(h, int((max(y_coords) + padding) * h))
                
                # Draw bounding rectangle around hand
                cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
                
                # Add text label
                cv2.putText(annotated_frame, "Hand Detected", (x_min, y_min - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                return annotated_frame, True
        
        # If no hand detected, add text indication
        cv2.putText(annotated_frame, "Show your hand to the camera", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        return annotated_frame, False
        
    except Exception as e:
        print(f"Error drawing hand landmarks: {e}")
        return frame, False

def enhance_image_for_prediction(frame):
    """Enhanced image preprocessing for better accuracy"""
    try:
        # Convert to grayscale for better hand detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization for better contrast
        gray = cv2.equalizeHist(gray)
        
        # Convert back to BGR for consistency with training
        enhanced = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # Apply Gaussian blur to reduce noise
        enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        # Sharpen the image
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        return enhanced
    except Exception as e:
        print(f"Error enhancing image: {e}")
        return frame

def predict_sign_enhanced(frame):
    """Enhanced prediction with MediaPipe hand detection for better accuracy"""
    if model is None:
        return "Model not loaded", 0.0, False
    
    try:
        # Extract hand region using MediaPipe
        hand_region, hand_detected = extract_hand_region(frame)
        
        if not hand_detected:
            return "No hand detected", 0.0, False
        
        # Apply image enhancement to hand region
        enhanced_frame = enhance_image_for_prediction(hand_region)
        
        # Standard preprocessing
        processed_frame = preprocess_frame(enhanced_frame)
        if processed_frame is None:
            return "Error processing frame", 0.0, False
        
        # Make prediction with confidence boosting
        predictions = []
        
        # Original prediction
        pred1 = model.predict(processed_frame, verbose=0)
        predictions.append(pred1)
        
        # Apply slight transformations for robustness (only if confidence is low)
        base_confidence = np.max(pred1)
        if base_confidence < 0.85:  # Only if we need more confidence
            # Slightly adjust brightness
            brightened = cv2.convertScaleAbs(enhanced_frame, alpha=1.1, beta=10)
            processed_bright = preprocess_frame(brightened)
            if processed_bright is not None:
                pred_bright = model.predict(processed_bright, verbose=0)
                predictions.append(pred_bright)
            
            # Slightly adjust contrast
            contrasted = cv2.convertScaleAbs(enhanced_frame, alpha=1.2, beta=0)
            processed_contrast = preprocess_frame(contrasted)
            if processed_contrast is not None:
                pred_contrast = model.predict(processed_contrast, verbose=0)
                predictions.append(pred_contrast)
        
        # Average the predictions for better accuracy
        if len(predictions) > 1:
            avg_prediction = np.mean(predictions, axis=0)
        else:
            avg_prediction = predictions[0]
            
        class_idx = np.argmax(avg_prediction)
        confidence = float(np.max(avg_prediction))
        predicted_label = categories[class_idx]
        
        # Apply confidence threshold for better accuracy
        min_confidence = 0.6
        if confidence < min_confidence:
            return "Low confidence", confidence, hand_detected
        
        return predicted_label, confidence, hand_detected
    except Exception as e:
        print(f"Error making prediction: {e}")
        return "Error", 0.0, False

def rotate_image(image, angle):
    """Rotate image by given angle"""
    try:
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
        return rotated
    except:
        return image

def predict_sign(frame):
    """Original prediction function for backward compatibility"""
    if model is None:
        return "Model not loaded", 0.0
    
    try:
        processed_frame = preprocess_frame(frame)
        if processed_frame is None:
            return "Error processing frame", 0.0
        
        # Make prediction
        prediction = model.predict(processed_frame, verbose=0)
        class_idx = np.argmax(prediction)
        confidence = float(np.max(prediction))
        predicted_label = categories[class_idx]
        
        return predicted_label, confidence
    except Exception as e:
        print(f"Error making prediction: {e}")
        return "Error", 0.0

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

# Global camera variable to control when camera is active
camera_active = False
current_camera = None
realtime_mode = None
realtime_active = False
latest_prediction = None  # Store latest prediction for real-time mode

@app.route('/video_feed')
def video_feed():
    """Video streaming route with hand detection overlay"""
    global camera_active
    if not camera_active:
        # Return a placeholder response when camera is not active
        return Response("Camera not active", mimetype='text/plain')
    
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    """Generate video frames with hand detection overlay"""
    global current_camera, camera_active, realtime_mode, realtime_active
    
    current_camera = cv2.VideoCapture(0)
    
    if not current_camera.isOpened():
        print("Error: Could not open camera")
        camera_active = False
        return
    
    frame_count = 0
    prediction_interval = 15  # Process prediction every 15 frames for real-time
    
    try:
        while camera_active:
            success, frame = current_camera.read()
            if not success:
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Draw hand landmarks and detection
            annotated_frame, hand_detected = draw_hand_landmarks(frame)
            
            # Real-time prediction processing
            if realtime_active and frame_count % prediction_interval == 0:
                try:
                    predicted_label, confidence, hand_detected_pred = predict_sign_enhanced(frame)
                    
                    # Store latest prediction globally for frontend access
                    global latest_prediction
                    latest_prediction = {
                        'prediction': predicted_label,
                        'confidence': round(confidence * 100, 2),
                        'hand_detected': hand_detected_pred,
                        'mode': realtime_mode,
                        'timestamp': time.time()
                    }
                    
                    # Add prediction text to frame if confidence is good
                    if hand_detected_pred and confidence > 0.65:  # Lower threshold for real-time display
                        prediction_text = f"Prediction: {predicted_label} ({confidence*100:.1f}%)"
                        cv2.putText(annotated_frame, prediction_text, (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    else:
                        # Show message when hand is not detected clearly
                        if not hand_detected_pred:
                            cv2.putText(annotated_frame, "Show your hand clearly", (10, 30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        else:
                            cv2.putText(annotated_frame, "Keep hand steady", (10, 30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)
                    
                except Exception as e:
                    print(f"Error in real-time prediction: {e}")
            
            # Add status text
            status_text = "Hand Detected: YES" if hand_detected else "Hand Detected: NO"
            status_color = (0, 255, 0) if hand_detected else (0, 0, 255)
            cv2.putText(annotated_frame, status_text, (10, annotated_frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            # Add real-time mode indicator
            if realtime_active:
                mode_text = f"Real-time {realtime_mode.upper()} Mode: ON"
                cv2.putText(annotated_frame, mode_text, (10, annotated_frame.shape[0] - 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            if not ret:
                continue
                
            frame_bytes = buffer.tobytes()
            
            frame_count += 1
            
            # Yield frame in byte format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    except Exception as e:
        print(f"Error in video feed: {e}")
    finally:
        if current_camera:
            current_camera.release()
            current_camera = None
        camera_active = False
        realtime_active = False

@app.route('/start_video_feed', methods=['POST'])
def start_video_feed():
    """Start the video feed"""
    global camera_active
    camera_active = True
    return jsonify({'success': True, 'message': 'Video feed started'})

@app.route('/stop_video_feed', methods=['POST'])
def stop_video_feed():
    """Stop the video feed"""
    global camera_active, current_camera, realtime_active, latest_prediction
    camera_active = False
    realtime_active = False
    latest_prediction = None  # Clear latest prediction
    if current_camera:
        current_camera.release()
        current_camera = None
    return jsonify({'success': True, 'message': 'Video feed stopped'})

@app.route('/capture_frame', methods=['POST'])
def capture_frame():
    """Capture a single frame from the camera for prediction"""
    global current_camera
    
    if not current_camera or not current_camera.isOpened():
        return jsonify({
            'success': False,
            'error': 'Camera not available'
        })
    
    try:
        # Capture frame
        success, frame = current_camera.read()
        if not success:
            return jsonify({
                'success': False,
                'error': 'Failed to capture frame'
            })
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Encode frame to base64
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            return jsonify({
                'success': False,
                'error': 'Failed to encode frame'
            })
        
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'image': f'data:image/jpeg;base64,{frame_base64}'
        })
        
    except Exception as e:
        print(f"Error capturing frame: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/start_realtime', methods=['POST'])
def start_realtime():
    """Start real-time prediction"""
    global realtime_active, realtime_mode
    data = request.get_json()
    mode = data.get('mode', 'text')
    
    realtime_active = True
    realtime_mode = mode
    
    return jsonify({
        'success': True, 
        'message': f'Real-time {mode} prediction started'
    })

@app.route('/stop_realtime', methods=['POST'])
def stop_realtime():
    """Stop real-time prediction"""
    global realtime_active, realtime_mode, latest_prediction
    
    realtime_active = False
    realtime_mode = None
    latest_prediction = None  # Clear latest prediction
    
    return jsonify({
        'success': True, 
        'message': 'Real-time prediction stopped'
    })

@app.route('/get_realtime_prediction', methods=['GET'])
def get_realtime_prediction():
    """Get the latest real-time prediction"""
    global latest_prediction
    
    if latest_prediction and realtime_active:
        return jsonify({
            'success': True,
            **latest_prediction
        })
    else:
        return jsonify({
            'success': False,
            'message': 'No real-time prediction available'
        })

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for hand sign to text prediction with MediaPipe hand detection"""
    try:
        # Get the image data from request
        data = request.get_json()
        image_data = data['image']
        mode = data.get('mode', 'text')  # 'text' or 'speech'
        
        # Remove data URL prefix
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert PIL image to OpenCV format
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Make prediction with MediaPipe hand detection for better accuracy
        predicted_label, confidence, hand_detected = predict_sign_enhanced(frame)
        
        response_data = {
            'success': True,
            'prediction': predicted_label,
            'confidence': round(confidence * 100, 2),
            'mode': mode,
            'hand_detected': hand_detected
        }
        
        # Only proceed if hand is detected and confidence is reasonable
        if hand_detected and confidence > 0.5:
            response_data['spoken'] = False
        else:
            response_data['spoken'] = False
            if not hand_detected:
                response_data['message'] = 'Please show your hand clearly in the camera'
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in prediction endpoint: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/predict_realtime', methods=['POST'])
def predict_realtime():
    """API endpoint for real-time hand sign prediction"""
    try:
        # Get the image data from request
        data = request.get_json()
        image_data = data['image']
        mode = data.get('mode', 'text')  # 'text' or 'speech'
        
        # Remove data URL prefix
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert PIL image to OpenCV format
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Make prediction with MediaPipe hand detection
        predicted_label, confidence, hand_detected = predict_sign_enhanced(frame)
        
        response_data = {
            'success': True,
            'prediction': predicted_label,
            'confidence': round(confidence * 100, 2),
            'mode': mode,
            'hand_detected': hand_detected,
            'timestamp': time.time()
        }
        
        # Only proceed if hand is detected and confidence is high enough
        if hand_detected and confidence > 0.75:  # Increased threshold for real-time
            response_data['spoken'] = False
        else:
            response_data['spoken'] = False
            if not hand_detected:
                response_data['message'] = 'Show your hand clearly'
            elif confidence <= 0.75:
                response_data['message'] = 'Keep hand steady for better recognition'
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in real-time prediction endpoint: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'categories_count': len(categories)
    })

if __name__ == '__main__':
    print("Starting ISL Conversion Flask App...")
    print(f"Model loaded: {model is not None}")
    print(f"Categories: {len(categories)}")
    app.run(debug=True, host='0.0.0.0', port=5000)
