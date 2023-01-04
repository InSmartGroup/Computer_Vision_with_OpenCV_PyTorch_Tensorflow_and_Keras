import cv2
import numpy as np

# Create a web camera reader object
video_capture = cv2.VideoCapture(0)

# Get some parameters for future use
frame_width = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
frame_size = (int(frame_width), int(frame_height))
fps = 30
codec_mp4 = cv2.VideoWriter_fourcc(*"FMP4")
codec_avi = cv2.VideoWriter_fourcc(*"MJPG")

web_camera_name = "Web_camera_template"

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

# Main loop
while True:
    # Read the web camera stream
    has_frames, frame = video_capture.read()
    if not has_frames:
        break

    # Mirror the frame for convenience
    frame = cv2.flip(frame, 1)

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

    cv2.imshow(web_camera_name, frame)

# Release the web camera objects
cv2.destroyAllWindows()
video_capture.release()