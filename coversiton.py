import cv2
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
import pyttsx3  # For text-to-speech
from tkinter import Tk, Label, Button, Text, Canvas
from PIL import Image, ImageTk

# Load the trained model
model = load_model("C:\\Users\\ASLAM\\Desktop\\ISL\\isl_mobilenetv2.h5")

# Define categories (same as during training)
categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
              'U', 'V', 'W', 'X', 'Y', 'Z', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Function to convert text to speech
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Function to preprocess the frame
def preprocess_frame(frame):
    img = cv2.resize(frame, (250, 250))  # Resize to the same size used in training
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to update the webcam feed
def update_frame():
    ret, frame = cap.read()
    if ret:
        # Preprocess the frame
        processed_frame = preprocess_frame(frame)
        prediction = model.predict(processed_frame)
        class_idx = np.argmax(prediction)
        predicted_label = categories[class_idx]

        # Display prediction
        text_box.delete(1.0, "end")
        text_box.insert("end", predicted_label)

        # Convert the frame for displaying in Tkinter
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        webcam_label.imgtk = imgtk
        webcam_label.configure(image=imgtk)

    root.after(10, update_frame)

# Initialize GUI
root = Tk()
root.title("Real-Time ISL to Text and Speech")
root.geometry("800x600")

# Webcam feed
webcam_label = Label(root)
webcam_label.pack(side="left", padx=20, pady=20)

# Text box for displaying predictions
text_box = Text(root, height=1, width=30, font=("Helvetica", 18))
text_box.pack(side="top", padx=20, pady=20)

# Button to trigger speech conversion
speak_button = Button(root, text="Speak", command=lambda: speak(text_box.get("1.0", "end-1c")), font=("Helvetica", 14))
speak_button.pack(side="bottom", pady=20)

# Open webcam
cap = cv2.VideoCapture(0)

# Start updating the frame
update_frame()

# Start the GUI event loop
root.mainloop()

# Release resources
cap.release()
cv2.destroyAllWindows()
