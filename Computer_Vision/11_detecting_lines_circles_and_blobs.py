import cv2
import numpy as np
import convenience_functions as cf

# Read an image
# source_image = cv2.imread('..//Data//Images//soduku.jpg', 1)
# print(f"Source image shape: {source_image.shape}")
# cf.show_cv2(source_image)

# image_gray = cf.color(source_image, 'bgr', 'gray')  # convert to grayscale

# image_thresh = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                      cv2.THRESH_BINARY_INV, 31, 27)  # adaptive threshold

# image_edges = cv2.Canny(image_gray, 200, 255, apertureSize=3)  # edge detection
# cf.show_cv2(image_edges)

# Detect lines using 'cv2.HoughLines()'
# lines = cv2.HoughLines(image_edges, 1, np.pi / 180, 255)

# Draw lines
# for line in lines:
#     rho, theta = line[0]
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a * rho
#     y0 = b * rho
#     x1 = int(x0 + 1000 * (-b))
#     y1 = int(y0 + 1000 * a)
#     x2 = int(x0 - 1000 * (-b))
#     y2 = int(y0 - 1000 * a)
#     cv2.line(source_image, (x1, y1), (x2, y2), (255, 255, 0), 2, lineType=cv2.LINE_AA)
#
# cf.show_cv2(source_image)

# Detect lines using 'cv2.HoughLinesP()'
# lines = cv2.HoughLinesP(image_thresh, rho=1, theta=np.pi / 180, threshold=255,
#                         minLineLength=3, maxLineGap=50)

# Draw the lines
# for index, line in enumerate(lines):
#     x1, y1, x2, y2 = line[0]
#     cv2.line(source_image, (x1, y1), (x2, y2), (255, 0, 255), 2, lineType=cv2.LINE_8)

# cf.show_cv2(source_image)

# Blob detection
source_image = cv2.imread('..//Data//Images//Sunflowers.jpg', 1)

# Create a blob detector
detector = cv2.SimpleBlobDetector_create()

# Detect keypoint in the image
keypoints = detector.detect(source_image)

# Draw keypoints
blank_image = np.zeros((1, 1))
image_keypoints = cv2.drawKeypoints(source_image, keypoints, blank_image,
                                    (255, 0, 255), cv2.DRAW_MATCHES_FLAGS_DEFAULT)
cf.show_cv2(image_keypoints)
