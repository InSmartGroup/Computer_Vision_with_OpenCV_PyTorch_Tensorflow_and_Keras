import cv2
import numpy as np
import convenience_functions as cf

# Read an image
image_bgr = cv2.imread('..//Data//Images//obama.jpg', 1)
print(f"Image shape: {image_bgr.shape}")

# Resize an image and apply different interpolation methods to check the output quality difference
image_resized_linear = cv2.resize(image_bgr, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_LINEAR)
# cf.show_cv2(image_resized_linear)
image_resized_cubic = cv2.resize(image_bgr, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
# cf.show_cv2(image_resized_cubic)
image_resized_lanczos = cv2.resize(image_bgr, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_LANCZOS4)
# cf.show_cv2(image_resized_lanczos)
image_resized_nearest = cv2.resize(image_bgr, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_NEAREST)
# cf.show_cv2(image_resized_nearest)

# Display the image using 'plt'
# cf.show_mat(image_bgr)

# Define the region of interest (ROI)
roi = image_bgr[40:220, 100:260, :]
# cf.show_cv2(roi)

# Draw a rectangular frame around the image borders
roi_rectangle = cv2.rectangle(roi, (0, 0), (roi.shape[1], roi.shape[0]),
                              color=(255, 0, 255), thickness=5, lineType=cv2.LINE_8)
# cf.show_cv2(roi)
