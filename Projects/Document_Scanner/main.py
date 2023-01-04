import cv2
import numpy as np
import cv_functions as cf

# Create a video reader object
video_capture = cv2.VideoCapture(0)

# Create 2 windows
cv2.namedWindow("Input video stream", cv2.WINDOW_NORMAL)
cv2.namedWindow("Scanned document")

# Get some parameters for future use
frame_width = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(frame_width, frame_height)
frame_size = (int(frame_width), int(frame_height))
fps = 10
codec_mp4 = cv2.VideoWriter_fourcc(*"FMP4")
codec_avi = cv2.VideoWriter_fourcc(*"MJPG")

video_name = "Document_Scanner"

# Check if it works okay
if video_capture.isOpened():
    print('Success.')
else:
    print("Unable to open a web camera.")

# Create a video writer object
video_writer = cv2.VideoWriter(f"{video_name}.mp4", codec_mp4, fps, frame_size)

# Define web camera stream modes
STREAM = True
RECORDING = False
DRAW_CONTOURS = False

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

    # Threshold the grayscale frame
    retval, frame_thresh = cv2.threshold(frame_gray, 230, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, hierarchy = cv2.findContours(frame_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by perimeter to find the largest one
    if len(contours) > 0:
        contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
        contours_sorted = list(contours_sorted)
        largest_contour = contours_sorted.pop(0)
        if DRAW_CONTOURS:
            cv2.drawContours(frame, largest_contour, -1, (0, 255, 255), 3, cv2.LINE_AA)

        # Find 4 keypoints of the largest contour
        contour_points = cv2.approxPolyDP(largest_contour,
                                          epsilon=cv2.arcLength(largest_contour, closed=True) * 0.05,
                                          closed=True)
        print(f"Contour points: {len(contour_points)}")

        # Define the size of the transformed image
        frame_transformed_size = (500, 800)

        # Define input and output keypoints
        if len(contour_points) == 4:
            input_points = np.float32(contour_points)

            output_points = np.float32([[0, 0],
                                        [0, 800],
                                        [500, 800],
                                        [500, 0]])

            # Calculate the perspective transformation matrix
            matrix = cv2.getPerspectiveTransform(input_points, output_points)

            # Transform and mirror the frame
            frame_transformed = cv2.warpPerspective(frame, matrix, frame_transformed_size)
            frame_transformed = cv2.flip(frame_transformed, 1)

            cv2.imshow("Scanned document", frame_transformed)
    else:
        pass

    # Define keyboard keys to control the camera stream
    key = cv2.waitKey(1)

    if key == 27 or key == ord('Q') or key == ord('q'):
        break

    # Define web camera modes behavior
    if RECORDING:
        video_writer.write(frame)

    if key == ord('R'):
        # Start the recording
        RECORDING = True

    if key == ord('S'):
        # Stop the recording
        RECORDING = False
        video_writer.release()

    if key == ord('D'):
        DRAW_CONTOURS = True

    elif key == ord('d'):
        DRAW_CONTOURS = False

    cv2.imshow("Input video stream", frame)

# Release the web camera objects
cv2.destroyAllWindows()
video_capture.release()
