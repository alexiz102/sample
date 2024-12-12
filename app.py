from flask import Flask, render_template, request, Response
from ultralytics import YOLO
import cv2

cap = cv2.VideoCapture(0) # Initialize webcam

model = YOLO("yolov8n.pt") # Initialize model

app = Flask(__name__) # Initialize webserver

#function for retrieveing frames and detection
def generate_frames(): 
    while True:
        ret, frame = cap.read() # Get Frames
        if not ret:
            break
        results = model.predict(frame) # Make prediction
        frame = results[0].plot() # Plot predictions
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', frame)[1].tobytes() + b'\r\n') # Send Frames
        
@app.route('/')
def index():
    return render_template('index.html')

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
