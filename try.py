import cv2
import torch
from ultralytics import YOLO
import smtplib
from email.mime.text import MIMEText
import os
import time
from flask import Flask, Response, render_template


app = Flask(__name__)


# 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
print(f"Using device: {device}")


model = YOLO("yolo-Weights/yolov8n.pt").to(device)


cap = cv2.VideoCapture(0)
cap.set(3, 640)  
cap.set(4, 480)  

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
              "teddy bear", "hair drier", "toothbrush"
              ]
alert_triggered = False

detected_objects = {}

def send_email_alert():
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
    except Exception as e:
        print(f"Failed to send email: {e}")


def log_event(frame, object_name, frame_time):
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime(frame_time))
    file_name = f"suspicious_event_{object_name}_{timestamp}.jpg"
    cv2.imwrite(file_name, frame)  

    with open("event_log.txt", "a") as log_file:
        log_file.write(f"Suspicious {object_name} detected at {time.ctime(frame_time)}\n")
    print(f"Logged event for {object_name}")

def log_behavior(frame, cls_name, frame_time):
    """ Log suspicious behavior based on the frequency and occurrence of objects. """
    current_time = time.time()
    if cls_name not in detected_objects:
        detected_objects[cls_name] = []

    detected_objects[cls_name].append(frame_time)
    
    # Keep a time window of 5 minutes (300 seconds)
    detected_objects[cls_name] = [t for t in detected_objects[cls_name] if current_time - t < 300]
    
    if len(detected_objects[cls_name]) > 2:
        print(f"Suspicious behavior detected: {cls_name} appears frequently.")
        log_event(frame, cls_name, current_time)  # Log event with frame and class name

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

            results = model(frame,verbose=False) 

            for r in results:
                boxes = r.boxes
                for box in boxes:
                 
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])

               
                    label = f"{classNames[cls]}"
                    color = (0, 255, 0) 
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                  
                    if classNames[cls] == 'cell phone' and not alert_triggered:
                        log_behavior(frame, classNames[cls], time.time())
                        send_email_alert()
                        alert_triggered = True  

           
            if alert_triggered and not any(box.cls[0] == classNames.index('cell phone') for r in results for box in r.boxes):
                alert_triggered = False

           
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

