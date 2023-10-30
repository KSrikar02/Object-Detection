import cv2

# Replace 'your_rtsp_url' with the actual RTSP URL of your IP camera
rtsp_url = f"rtsp://admin:admin@192.168.1.216:554/stream1"


# Create a VideoCapture object to connect to the camera
cap = cv2.VideoCapture(rtsp_url)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Unable to connect to the camera.")
    exit()

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to retrieve frame from the camera.")
        break

    # Display the frame
    cv2.imshow('IP Camera Feed', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
