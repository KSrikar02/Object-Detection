import cv2
import numpy as np

# Replace with the path to your YOLOv3 configuration and weights files
yolov3_cfg = 'yolov4.cfg'
yolov3_weights = 'yolov4.weights'
yolov3_names = 'yolov3.names'  # Path to the YOLOv3 class names file

# Load YOLOv3 model
net = cv2.dnn.readNet(yolov3_weights, yolov3_cfg)

# Load class names
with open(yolov3_names, 'r') as f:
    class_names = f.read().strip().split('\n')

# Replace 'your_rtsp_url' with the actual RTSP URL of your IP camera
rtsp_url = f"rtsp://admin:admin@192.168.1.216:554/stream1"

# Create a VideoCapture object to connect to the camera
cap = cv2.VideoCapture(rtsp_url)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Unable to connect to the camera.")
    exit()

# Set the desired frame rate (FPS)
cap.set(cv2.CAP_PROP_FPS, 10)  # Set to 10 frames per second

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to retrieve frame from the camera.")
        break

    # Prepare the frame for YOLOv3
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Get the output layer names
    output_layer_names = net.getUnconnectedOutLayersNames()

    # Forward pass
    detections = net.forward(output_layer_names)

    # Process the detections
    # Process the detections
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:  # You can adjust the confidence threshold as needed
               center_x, center_y, width, height = obj[0], obj[1], obj[2], obj[3]

            # Calculate the top-left corner coordinates of the bounding box
               x = int((center_x - width / 2) * frame.shape[1])
               y = int((center_y - height / 2) * frame.shape[0])
               width = int(width * frame.shape[1])
               height = int(height * frame.shape[0])

            # Draw a bounding box and label
               cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
               label = f"{class_names[class_id]}: {confidence:.2f}"
               cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with object detection
    cv2.imshow('YOLOv3 Object Detection', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()


