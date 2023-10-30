import cv2
import numpy as np

# This is to pull the information about what each object is called
classNames = []
classFile = "coco.names"
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

# This is to pull the information about what each object should look like
configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "frozen_inference_graph.pb"

# This is some setup values to get good results
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Variables to track ROI selection
roi_selected = False
roi_start = (0, 0)
roi_end = (0, 0)

# This function handles mouse events for ROI selection
def selectROI(event, x, y, flags, param):
    global roi_selected, roi_start, roi_end

    if event == cv2.EVENT_LBUTTONDOWN:
        roi_selected = False
        roi_start = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        roi_end = (x, y)
        roi_selected = True

# This is to set up what the drawn box size/color is and the font/size/color of the name tag and confidence label
def getObjects(img, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nms)

    if len(objects) == 0:
        objects = classNames
    objectInfo = []
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box, className])
                if draw:
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    return img, objectInfo

# Below is the never-ending loop that determines what will happen when an object is identified.
if __name__ == "__main__":
    # Replace 'rtsp://your_camera_ip_address:port/' with your camera's RTSP URL
    rtsp_url = f"rtsp://admin:admin@192.168.1.216:554/stream1"


    cap = cv2.VideoCapture(rtsp_url)

    # Set the desired frame rate (FPS)
    cap.set(cv2.CAP_PROP_FPS, 10)  # Set to 10 frames per second

    # Below determines the size of the live feed window that will be displayed
    cap.set(3, 640)
    cap.set(4, 480)

    cv2.namedWindow("Output")
    cv2.setMouseCallback("Output", selectROI)

    while True:
        success, img = cap.read()

        if roi_selected:
            roi_img = img[roi_start[1]:roi_end[1], roi_start[0]:roi_end[0]]
            result, objectInfo = getObjects(roi_img, 0.45, 0.2)
            cv2.imshow("Output", result)
        else:
            result, objectInfo = getObjects(img, 0.45, 0.2)
            cv2.imshow("Output", result)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

