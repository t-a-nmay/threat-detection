import cv2
import torch
from ultralytics import YOLO
import smtplib
from email.mime.text import MIMEText
import time
from flask import Flask, Response, render_template
import atexit

# Flask application setup
app = Flask(__name__)

# Global variables
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load YOLO models
model1 = YOLO("C:/Users/priya/objdec/yolo-Weights/best1.pt").to(device)
model2 = YOLO("C:/Users/priya/objdec/yolo-Weights/best2.pt").to(device)

cap = cv2.VideoCapture(0)
cap.set(3, 320)  # Width
cap.set(4, 240)  # Height

# Object classes
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
detected_objects = {}
last_email_time = 0  # For cooldown

# Email alert function
def send_email_alert():
    global last_email_time
    current_time = time.time()
    cooldown = 300  # 5 minutes cooldown

    if current_time - last_email_time < cooldown:
        print("Email alert suppressed (cooldown).")
        return

    sender_email = "ganeshakabhagwaan@gmail.com"
    receiver_email = "tanmaypandita111@gmail.com"
    password = "rozc zguq arav lhqp"

    message = MIMEText("Intruder detected!")
    message['Subject'] = "Security Alert"
    message['From'] = sender_email
    message['To'] = receiver_email

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, message.as_string())
            print("Email alert sent!")
            last_email_time = current_time
    except Exception as e:
        print(f"Failed to send email: {e}")

# Event logging function
def log_event(frame, object_name, frame_time):
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime(frame_time))
    file_name = f"suspicious_event_{object_name}_{timestamp}.jpg"
    cv2.imwrite(file_name, frame)

    with open("event_log.txt", "a") as log_file:
        log_file.write(f"Suspicious {object_name} detected at {time.ctime(frame_time)}\n")
    print(f"Logged event for {object_name}")

# Behavior logging function
def log_behavior(frame, cls_name, frame_time):
    current_time = time.time()
    if cls_name not in detected_objects:
        detected_objects[cls_name] = []

    detected_objects[cls_name].append(frame_time)

    # Keep a time window of 5 minutes (300 seconds)
    detected_objects[cls_name] = [t for t in detected_objects[cls_name] if current_time - t < 300]

    if len(detected_objects[cls_name]) > 2:
        print(f"Suspicious behavior detected: {cls_name} appears frequently.")
        log_event(frame, cls_name, current_time)

# Flask routes
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

            # Run both models
            results1 = model1(frame, verbose=False)
            results2 = model2(frame, verbose=False)

            # Process results from both models
            for results in [results1, results2]:
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls = int(box.cls.item())

                        label = f"{classNames[cls]}"
                        color = (0, 255, 0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                        if classNames[cls] == 'cell phone' and not alert_triggered:
                            log_behavior(frame, classNames[cls], time.time())
                            send_email_alert()
                            alert_triggered = True

            # Reset alert_triggered if no "cell phone" detected
            if alert_triggered and not any(
                int(box.cls.item()) == classNames.index('cell phone') for results in [results1, results2] for r in results for box in r.boxes
            ):
                alert_triggered = False

            # Encode frame
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Release camera on exit
@atexit.register
def release_camera():
    if cap.isOpened():
        cap.release()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
