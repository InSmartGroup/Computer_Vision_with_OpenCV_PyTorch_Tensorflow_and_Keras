import cv2
import numpy as np
import convenience_functions as cf

# Read an image
source_image = cv2.imread('..//Data//Images//bunchofshapes.jpg', 1)
# print(f"Source image shape: {source_image.shape}")
image_resized = cf.resize(source_image, 0.5)
# cf.show_cv2(image_resized)

# Convert to grayscale
image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

# Threshold
val, image_thresh = cv2.threshold(image_gray, 200, 255, cv2.THRESH_BINARY)
# cf.show_cv2(image_thresh)

# Find contours
contours, hierarchy = cv2.findContours(image_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
# print(f"Number of contours: {len(contours)}")

# Calculate contours area and perimeter
for index, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, closed=True)
    # print(f"Contour {index}: {area} area, {round(perimeter, 3)} perimeter")

# Remove the largest contour from the contours list
contours = list(contours)
largest_contour = contours.pop(4)

# Calculate the center of mass for each contour and mark the center
for index, contour in enumerate(contours):
    m = cv2.moments(contour)
    x = int(round(m["m10"] / m['m00']))
    y = int(round(m["m01"] / m['m00']))
    # print(f"Contour {index}: centroid {x, y}")
    cv2.circle(image_resized, (x, y), 2, (255, 0, 0), -1, lineType=cv2.LINE_AA)
# print()

# Draw contours
image_contours = cv2.drawContours(image_resized, contours, -1, (0, 255, 0), 2, cv2.LINE_AA)
# cf.show_cv2(image_contours)

# Convex Hull
source_image = cv2.imread('..//Data//Images//hand.jpg', 1)  # read an image
image_gray = cf.color(source_image, 'bgr', 'gray')  # convert to grayscale
val, image_thresh = cv2.threshold(image_gray, 175, 255, cv2.THRESH_BINARY)  # threshold
# cf.show_cv2(image_thresh)

contours, hierarchy = cv2.findContours(image_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)  # find contours

areas, perimeters = cf.contours_area_perimeter(contours, verbose=True)  # find all contour areas and perims
contours = list(contours)  # convert the tuple to a list
largest_contour = contours.pop(2)  # pop the largest contour (around the whole image)

for contour in contours:  # draw all the remaining contours using the 'cv2.convexHull()' (it should be a list)
    cv2.drawContours(source_image, [cv2.convexHull(contour)], -1, (0, 0, 255), 3, lineType=cv2.LINE_AA)
# cf.show_cv2(source_image)

