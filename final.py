import cv2
import torch
from ultralytics import YOLO
from inference_sdk import InferenceHTTPClient
import smtplib
from email.mime.text import MIMEText
import time

# Device configuration
device = 'cpu'
print(f"Using device: {device}")

# Initialize YOLO model
model = YOLO("C:/Users/tanma/Desktop/Object Detection 2/yolo-Weights/yolov8n.pt")

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
detected_objects = {}

# Email alert function
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
    if cls_name not in ["gun", "knife", "pistol"]:
        return

    current_time = time.time()
    if cls_name not in detected_objects:
        detected_objects[cls_name] = []

    detected_objects[cls_name].append(frame_time)
    
    # Keep a time window of 5 minutes (300 seconds)
    detected_objects[cls_name] = [t for t in detected_objects[cls_name] if current_time - t < 300]
    
    if len(detected_objects[cls_name]) > 2:
        print(f"Suspicious behavior detected: {cls_name} appears frequently.")
        log_event(frame, cls_name, current_time)  # Log event with frame and class name

# Main function
def main():
    global alert_triggered

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to read frame from camera.")
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
                if label in ["gun", "pistol"] and confidence > 0.8 and not alert_triggered:
                    print("Gun detected! Alert triggered by Roboflow.")
                    send_email_alert()
                    log_event(frame, label, time.time())
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
                if label in ["knife", "pistol", "gun"] and confidence > 0.8 and not alert_triggered:
                    print("Suspicious object detected! Alert triggered by YOLO.")
                    send_email_alert()
                    log_event(frame, label, time.time())
                    alert_triggered = True

                # Log suspicious behavior
                log_behavior(frame, label, time.time())

        # Reset alert if no relevant objects are detected
        if alert_triggered and not any(pred.get("class") in ["gun", "pistol"] for pred in roboflow_results.get("predictions", [])):
            alert_triggered = False

        # Display the frame
        cv2.imshow("Object Detection", frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
