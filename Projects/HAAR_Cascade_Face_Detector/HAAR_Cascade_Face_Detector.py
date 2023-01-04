import cv2
import numpy as np
import cv_functions as cf

# Create a web camera reader object
video_capture = cv2.VideoCapture(0)

# Get some parameters for future use
frame_width = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
frame_size = (int(frame_width), int(frame_height))
fps = 15
codec_mp4 = cv2.VideoWriter_fourcc(*"FMP4")
codec_avi = cv2.VideoWriter_fourcc(*"MJPG")

web_camera_name = "HAAR_Cascade_Face_Detector"

# Check if it works okay
if video_capture.isOpened():
    print('Success.')
else:
    print("Unable to open a web camera.")

# Create a video writer object
video_writer = cv2.VideoWriter(f"{web_camera_name}.avi", codec_avi, fps, frame_size)

# Define web camera stream modes
STREAM = True
RECORDING = False
DETECT_FACES = False
DETECT_EYES = False
DETECT_SMILES = False

# Create detector objects
face_detector = cv2.CascadeClassifier("Haarcascades//haarcascade_frontalface_default.xml")
eye_detector = cv2.CascadeClassifier("Haarcascades//haarcascade_eye.xml")
smile_detector = cv2.CascadeClassifier("Haarcascades//haarcascade_smile.xml")


# Main loop
while True:
    # Read the web camera stream
    has_frames, frame = video_capture.read()
    if not has_frames:
        break

    # Mirror the frame for convenience
    frame = cv2.flip(frame, 1)

    # Convert the frame to grayscale
    frame_gray = cf.color(frame, 'bgr', 'gray')

    # Detect faces, eyes, and smiles in the frame
    faces = face_detector.detectMultiScale(frame_gray, scaleFactor=1.3, minNeighbors=10)
    eyes = eye_detector.detectMultiScale(frame_gray, scaleFactor=1.3, minNeighbors=10)
    smiles = smile_detector.detectMultiScale(frame_gray, scaleFactor=4, minNeighbors=20)

    # Mark detected faces
    if DETECT_FACES:
        for x, y, w, h in faces:
            cv2.rectangle(frame, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 255),
                          thickness=3, lineType=cv2.LINE_8)
            # Draw a rectangular bubble
            cv2.rectangle(frame, (x, y - 20), (x + 35, y ), color=(0, 0, 0),
                          thickness=-1, lineType=cv2.LINE_8)
            # Put text
            cv2.putText(frame, 'face', (x, y - 5), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                        color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

    # Mark detected eyes
    if DETECT_EYES:
        for x, y, w, h in eyes:
            cv2.rectangle(frame, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 255),
                          thickness=1, lineType=cv2.LINE_8)
            # Put text
            cv2.putText(frame, 'eye', (x, y - 5), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                        color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

    # Mark detected smiles
    if DETECT_SMILES:
        for x, y, w, h in smiles:
            cv2.rectangle(frame, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0),
                          thickness=1, lineType=cv2.LINE_AA)
            # Draw a rectangular bubble
            cv2.rectangle(frame, (x, y - 20), (x + 45, y ), color=(0, 0, 0),
                          thickness=-1, lineType=cv2.LINE_8)
            # Put text
            cv2.putText(frame, 'smile', (x, y - 5), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                        color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

    # Define keyboard keys to control the camera stream
    key = cv2.waitKey(1)

    if key == 27 or key == ord('Q') or key == ord('q'):
        break

    # Define web camera modes behavior
    if RECORDING:
        video_writer.write(frame)

    if key == ord('S'):
        # Stop the recording
        RECORDING = False
        video_writer.release()

    if key == ord('R'):
        # Start the recording
        RECORDING = True

    if key == ord('f'):
        # Face detection ON
        DETECT_FACES = True

    if key == ord('e'):
        # Eye detection ON
        DETECT_EYES = True

    if key == ord('s'):
        # Smile detection ON
        DETECT_SMILES = True

    if key == ord('o'):
        # All detection modes OFF
        DETECT_FACES = False
        DETECT_EYES = False
        DETECT_SMILES = False

    cv2.imshow(web_camera_name, frame)

# Release the web camera objects
cv2.destroyAllWindows()
video_capture.release()