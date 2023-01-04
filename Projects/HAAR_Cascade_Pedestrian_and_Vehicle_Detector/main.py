import cv2
import numpy as np
import cv_functions as cf

# Path to a video file
path = r'C:\Users\gostapenko\PycharmProjects\Computer_Vision_with_OpenCV_PyTorch_Tensorflow_and_Keras\Data\Images\walking.avi'

# Create a video capture object
video_capture = cv2.VideoCapture(path)

# Get some parameters for future use
frame_width = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
frame_size = (int(frame_width), int(frame_height))
fps = 10
codec_mp4 = cv2.VideoWriter_fourcc(*"FMP4")
codec_avi = cv2.VideoWriter_fourcc(*"MJPG")

video_name = "Pedestrian_Detector"

# Check if it works
if video_capture.isOpened():
    print('Success.')
else:
    print("Unable to open a video file.")

# Create a video writer object
video_writer = cv2.VideoWriter(f"{video_name}.mp4", codec_mp4, fps, frame_size)

# Create a HAAR cascade object
pedestrian_detector = cv2.CascadeClassifier("Haarcascades//haarcascade_fullbody.xml")

# Main loop
while True:
    # Read the web camera stream
    has_frames, frame = video_capture.read()
    if not has_frames:
        break

    # Convert a frame to grayscale
    frame_gray = cf.color(frame, 'bgr', 'gray')

    # Detect pedestrians
    pedestrians = pedestrian_detector.detectMultiScale(frame_gray, minNeighbors=7)
    bodies_found = len(pedestrians)

    for x, y, w, h in pedestrians:
        cv2.rectangle(frame, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 255),
                      thickness=2, lineType=cv2.LINE_4)

    cv2.putText(frame, "Pedestrians: " + str(bodies_found), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)

    # Define keyboard keys to control the camera stream
    key = cv2.waitKey(1)

    if key == 27 or key == ord('Q') or key == ord('q'):
        break

    cv2.imshow(video_name, frame)

    video_writer.write(frame)

# Release all objects
cv2.destroyAllWindows()
video_capture.release()
video_writer.release()
