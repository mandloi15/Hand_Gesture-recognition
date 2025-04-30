import streamlit as st
import cv2
from PIL import Image
import torch
import numpy as np
import mediapipe as mp
import time
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Page config
st.set_page_config(page_title="Hand Gesture Recognition", layout="wide")
st.title("Real-time Hand Gesture Recognition")

# Initialize session state
if 'camera_on' not in st.session_state:
    st.session_state.camera_on = False
if 'fps' not in st.session_state:
    st.session_state.fps = 30

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    selected_fps = st.slider("Select FPS", min_value=1, max_value=60, value=30)
    st.session_state.fps = selected_fps

# Load model and processor
@st.cache_resource
def load_model():
    model_name = "dima806/hand_gestures_image_detection"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    return processor, model

processor, model = load_model()

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Create columns for buttons and stats
col1, col2, col3 = st.columns(3)
with col1:
    start_button = st.button("Start Camera")
with col2:
    stop_button = st.button("Stop Camera")
with col3:
    fps_display = st.empty()

# Create a placeholder for the webcam feed
video_placeholder = st.empty()

# Camera control logic
if start_button:
    st.session_state.camera_on = True
if stop_button:
    st.session_state.camera_on = False

# Initialize webcam
cap = cv2.VideoCapture(0)

# Set camera properties for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, st.session_state.fps)

# FPS calculation variables
frame_count = 0
start_time = time.time()

while st.session_state.camera_on:
    # FPS control
    frame_start_time = time.time()
    
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to access webcam")
        break

    # Convert BGR to RGB for MediaPipe
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect hand landmarks
    results = hands.process(img_rgb)
    
    # Draw landmarks if hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
    
    img_pil = Image.fromarray(img_rgb)

    # Resize and preprocess
    inputs = processor(images=img_pil, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax(-1).item()
    label = model.config.id2label[predicted_class]

    # Display result
    cv2.putText(frame, f'Gesture: {label}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Calculate and display FPS
    frame_count += 1
    if frame_count % 30 == 0:
        current_time = time.time()
        fps = frame_count / (current_time - start_time)
        fps_display.text(f"FPS: {fps:.2f}")
        frame_count = 0
        start_time = current_time

    # Control frame rate
    elapsed_time = time.time() - frame_start_time
    sleep_time = max(0, 1.0/st.session_state.fps - elapsed_time)
    time.sleep(sleep_time)

    # Display the frame in Streamlit
    video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

hands.close()
cap.release()

# Display message when camera is off
if not st.session_state.camera_on:
    video_placeholder.empty()
    st.info("Camera is turned off. Click 'Start Camera' to begin.")