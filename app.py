import cv2
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Load model and processor
model_name = "dima806/hand_gestures_image_detection"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    # Resize and preprocess
    inputs = processor(images=img_pil, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax(-1).item()
    label = model.config.id2label[predicted_class]

    # Display result
    cv2.putText(frame, f'Gesture: {label}', (10, 30), cv2.
                +FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)
    cv2.imshow("Webcam Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
