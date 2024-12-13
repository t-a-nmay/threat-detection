import cv2
import time
import torch
from ultralytics import YOLO
import smtplib
from email.mime.text import MIMEText
from flask import Flask, Response, render_template
from inference_sdk import InferenceHTTPClient

# Initialize Flask app
app = Flask(__name__)

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Initialize YOLO model
model = YOLO("yolo-Weights/yolov8n.pt").to(device)

# Initialize Roboflow Inference Client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="8ufkhwwGgmhkzNovrf8r"
)
MODEL_ID = "gun-detection-s5poj/1"

# OpenCV Video Capture
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height

# Object classes for YOLO
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

alert_triggered = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def generate_frames():
        global alert_triggered
        while True:
            success, frame = cap.read()
            if not success:
                break

            # Convert frame to RGB for Roboflow
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Save frame to a temporary file for Roboflow inference
            temp_image_path = "temp_frame.jpg"
            cv2.imwrite(temp_image_path, frame_rgb)

            # Perform inference with Roboflow
            roboflow_results = CLIENT.infer(temp_image_path, model_id=MODEL_ID)

            # Perform inference with YOLO
            results = model(frame, verbose=False)

            # Parse Roboflow results
            if roboflow_results and roboflow_results.get("predictions"):
                for pred in roboflow_results["predictions"]:
                    x1, y1 = int(pred["x"] - pred["width"] / 2), int(pred["y"] - pred["height"] / 2)
                    x2, y2 = int(pred["x"] + pred["width"] / 2), int(pred["y"] + pred["height"] / 2)
                    label = pred["class"]
                    confidence = pred["confidence"]

                    # Draw bounding box and label for Roboflow results
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} ({confidence:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Trigger alert for specific conditions
                    if label == "gun" and confidence > 0.5 and not alert_triggered:
                        print("Gun detected! Alert triggered by Roboflow.")
                        alert_triggered = True

            # Parse YOLO results
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    label = f"{classNames[cls]}"
                    confidence = box.conf[0]

                    # Draw bounding box and label for YOLO results
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, f"{label} ({confidence:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    # Trigger alert for specific conditions
                    if label == "cell phone" and confidence > 0.5 and not alert_triggered:
                        print("Cell phone detected! Alert triggered by YOLO.")
                        alert_triggered = True

            # Reset alert if no relevant objects are detected
            if alert_triggered and not any(pred.get("class") == "gun" for pred in roboflow_results["predictions"]):
                alert_triggered = False

            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
