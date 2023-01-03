import cv2
import numpy as np
import convenience_functions as cf

# Read an image
image_bgr = cv2.imread("..//Data//Images//LP.jpg", 1)
# print(f"Source image shape: {image_bgr.shape}")
# cf.show_cv2(image_bgr)

# Convert to grayscale
image_gray = cf.color(image_bgr, 'bgr', 'gray')

# Threshold
# val, image_thresh = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# cf.show_cv2(image_thresh)

# Find contours
# contours, hierarchy = cv2.findContours(image_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# print(len(contours))

# Find the area and perimeter for each contour in the list
# for index, contour in enumerate(contours):
#     area = cv2.contourArea(contour)
#     perimeter = cv2.arcLength(contour, closed=True)
#     print(f"Contour {index} has area {area}, perimeter {round(perimeter, 3)}")

# Draw contours in the source image
# image_contours = cv2.drawContours(image_bgr, contours, contourIdx=-1,
#                                   color=(0, 255, 255), thickness=2, lineType=cv2.LINE_AA)

# cf.show_cv2(image_contours)

# Draw bounding boxes around the contours
# for contour in contours:
#     x, y, w, h = cv2.boundingRect(contour)
#     cv2.rectangle(image_bgr, (x, y), (x + w, y + h), (255, 255, 0), thickness=1, lineType=cv2.LINE_8)

# cf.show_cv2(image_bgr)

# Calculate the center of mass of each contour
# for index, contour in enumerate(contours):
#     moments = cv2.moments(contour)
#     x = int(round(moments["m10"] / moments["m00"]))
#     y = int(round(moments["m01"] / moments["m00"]))
#     cv2.circle(image_bgr, (x, y), 5, (255, 255, 255), -1)

# cf.show_cv2(image_bgr)

# We can also use Canny instead of thresholding
image_canny = cv2.Canny(image_gray, 200, 255)
# cf.show_cv2(image_canny)

# Find contours in the Canny image
contours, hierarchy = cv2.findContours(image_canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
print(f"Number of contours found: {len(contours)}")

# Find contour area and perimeter
for index, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, closed=True)
    print(f"Contour {index} has area {area}, perimeter {round(perimeter, 3)}")

# Draw bounding boxes around found contours
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(image_bgr, (x, y), (x + w, y + h), (255, 255, 0), 2, lineType=cv2.LINE_8)

cf.show_cv2(image_bgr)
